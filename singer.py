"""SingerTranslator render orchestrator.

Public surface: render(syllables) -> path-to-mixed-wav.

Pipeline:
  syllables -> phonemes (g2p_en, with double-plosive recipe on first slot)
            -> target metadata (lyric/phoneme override on top of MJ chorus
               template, preserving notes/durations/melismas)
            -> SoulX inference (subprocess, MJ verse as voice prompt)
            -> ffmpeg mix with Thriller chorus accompaniment
            -> wav file
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Sequence

from g2p_en import G2p

# ---------------- paths -----------------------------------------------------

ROOT = Path(__file__).parent.resolve()
ASSETS = ROOT / "assets"
# Default to the bundled submodule. Local dev can override via SOULX_ROOT to
# reuse an existing install (its venv + downloaded weights).
SOULX_ROOT = Path(os.environ.get("SOULX_ROOT", str(ROOT / "vendor" / "SoulX-Singer")))
# Use Python's tempdir by default — Gradio's _check_allowed accepts it on
# macOS (where /tmp is a symlink that Gradio rejects).
WORK = Path(os.environ.get("SINGER_WORK_DIR", os.path.join(tempfile.gettempdir(), "singer_renders")))
WORK.mkdir(parents=True, exist_ok=True)

PROMPT_WAV   = ASSETS / "prompt.wav"
PROMPT_META  = ASSETS / "prompt.json"
TEMPLATE     = ASSETS / "chorus_target.json"   # notes/timings/types from MJ chorus
ACCOMP_WAV   = ASSETS / "accompaniment.wav"

PLOSIVES = {"B", "P", "T", "K", "D", "G"}

# Manual phoneme overrides for known syllables where g2p_en's English-only
# pronunciation would be wrong (mostly Spanish, but extensible).
SYLLABLE_OVERRIDES: dict[str, str] = {
    # buenos dias (Spanish) — g2p_en mispronounces 'nos' and 'as'
    "bue": "en_B-W-EH1",
    "nos": "en_N-OW1-S",
    "di":  "en_D-IY1",
    "as":  "en_AA1-S",
    # michael jackson — g2p_en treats 'chael' as standalone -> "CHAYL".
    # Real second syllable of "michael" is K-AH0-L ("kul"). Force it.
    "mi":    "en_M-IY1",
    "chael": "en_K-AH0-L",
    "jack":  "en_JH-AE1-K",
    "son":   "en_S-AH0-N",
    # happy birthday — g2p_en gives 'hap' -> HH-AE1-P, which combined with
    # the following 'pee' (P-IY1) produces a double-P that doesn't sound
    # like "happy" (/ˈhæpi/ has one P, owned by the second syllable).
    "hap":   "en_HH-AE1",
    # llueve mucho (Spanish) — g2p_en treats these as English which loses
    # the Castilian phonology entirely:
    #   'llue' -> L-UW1 (= "loo") drops the palatal — should be /jwe/
    #   've'   -> V-IY1 (= "vee") uses English /v/ — Spanish v=/b/
    # 'mu' and 'cho' happen to round-trip OK via g2p_en (M-UW1, CH-OW1).
    "llue": "en_Y-W-EH1",   # Y = English /j/ — closest ARPABET to Castilian /jwe/
    "ve":   "en_B-EH1",     # Spanish /be/
}

# ---------------- phoneme helpers ------------------------------------------

_g2p: G2p | None = None

def _g2p_instance() -> G2p:
    global _g2p
    if _g2p is None:
        _g2p = G2p()
    return _g2p


def syllable_to_phoneme(syllable: str) -> str:
    """Convert one syllable to SoulX 'en_X-Y-Z' phoneme string.

    Checks SYLLABLE_OVERRIDES first (case-insensitive), falls back to g2p_en.
    """
    key = syllable.lower()
    if key in SYLLABLE_OVERRIDES:
        return SYLLABLE_OVERRIDES[key]
    phones = [p for p in _g2p_instance()(syllable) if p.strip()]
    if not phones:
        raise ValueError(f"g2p_en returned no phones for syllable {syllable!r}")
    return "en_" + "-".join(phones)


def held_form(phoneme: str) -> str:
    """Vowel-only form for melismas. en_B-W-EH1 -> en_EH1.

    Strategy: pick the LAST stressed vowel (ends in 0/1/2). If none, fall back
    to the last part.
    """
    parts = phoneme.removeprefix("en_").split("-")
    for p in reversed(parts):
        if p and p[-1] in "012":
            return f"en_{p}"
    return f"en_{parts[-1]}"


def double_plosive_if_needed(phoneme: str) -> str:
    """For phonemes whose first phone is a plosive (B/P/T/K/D/G), double it.

    Mirrors the milestone-13 recipe (en_B-W-EH1 -> en_B-B-W-EH1) which made
    the leading consonant survive on short, high notes.
    """
    parts = phoneme.removeprefix("en_").split("-")
    if not parts or parts[0] not in PLOSIVES:
        return phoneme
    return "en_" + "-".join([parts[0]] + parts)

# ---------------- metadata builder -----------------------------------------

OFF_MELISMA_MIN_DUR = 0.20         # in 'off' mode, slots shorter than this
                                   # stay as melismas — too short for the model
                                   # to articulate a fresh syllable cleanly
DEFAULT_AUTO_MELISMA_DUR = 0.30    # in 'default' mode, also auto-melisma slots
                                   # below this — protects against the smearing
                                   # effect on fast slots while keeping the
                                   # metadata's natural note_type=3 melismas.


def build_target_metadata(syllables: Sequence[str], out_path: Path,
                          melisma_mode: str = "default") -> Path:
    """Build a SoulX target_metadata.json by cyclically mapping `syllables`
    onto the chorus template.

    melisma_mode:
      'off'     — fresh syllable on every slot >= OFF_MELISMA_MIN_DUR (0.20s);
                  very short slots still get the held vowel because the model
                  can't articulate a fresh syllable that fast. More lyric
                  clarity, less smooth.
      'default' — honour the metadata's note_type=3 as a held vowel, AND
                  auto-melisma any slot below DEFAULT_AUTO_MELISMA_DUR (0.25s).

    The leading-plosive recipe is applied only to slot-1 (first sung note),
    not to every recurrence — that's where short+high articulation is hardest.
    """
    if len(syllables) == 0:
        raise ValueError("need at least one syllable")
    if melisma_mode not in ("off", "default"):
        raise ValueError(f"unknown melisma_mode {melisma_mode!r}")

    template = json.loads(TEMPLATE.read_text())
    item = template[0]

    def _split(field):
        v = item[field]
        return v.split() if isinstance(v, str) else list(v)

    note_pitch = _split("note_pitch")
    note_type  = _split("note_type")
    durations  = [float(d) for d in _split("duration")]

    phonemes = [syllable_to_phoneme(s) for s in syllables]
    held     = [held_form(p) for p in phonemes]

    new_text, new_phon = [], []
    new_ntype = list(note_type)
    cycle_idx = 0
    last_idx: int | None = None
    first_fresh_done = False

    for i, (pitch_s, ntype_s) in enumerate(zip(note_pitch, note_type)):
        pitch = int(pitch_s); ntype = int(ntype_s)

        if pitch == 0 or ntype == 1:
            new_text.append("<SP>")
            new_phon.append("<SP>")
            new_ntype[i] = "1"
            last_idx = None
            continue

        # Decide whether this slot is a melisma (held vowel of the previous
        # syllable) or a fresh syllable, based on melisma_mode.
        force_melisma = (
            ntype == 3 and melisma_mode != "off" and last_idx is not None
        ) or (
            # 'off' mode still keeps very-short slots as melismas — too fast
            # for a fresh syllable to articulate cleanly.
            melisma_mode == "off" and last_idx is not None
            and durations[i] < OFF_MELISMA_MIN_DUR
        ) or (
            # 'default' mode auto-melismas slots below the higher threshold,
            # protecting against the smearing on fine-grained metadata while
            # still honouring metadata note_type=3 above.
            melisma_mode == "default" and last_idx is not None
            and durations[i] < DEFAULT_AUTO_MELISMA_DUR
        )

        if force_melisma:
            new_text.append(syllables[last_idx] + "_")
            new_phon.append(held[last_idx])
            new_ntype[i] = "3"
            continue

        idx = cycle_idx % len(syllables)
        new_text.append(syllables[idx])
        if not first_fresh_done:
            candidate = double_plosive_if_needed(phonemes[idx])
            new_phon.append(candidate)
            # Only consume the "first plosive" slot if doubling actually
            # fired. When slot-1's syllable has no plosive lead (e.g. 'hap'
            # in happy-birthday), the recipe should still apply to the next
            # fresh plosive-leading slot — that's where the short+high
            # plosive risk actually lands.
            if candidate != phonemes[idx]:
                first_fresh_done = True
        else:
            new_phon.append(phonemes[idx])
        last_idx = idx
        cycle_idx += 1
        new_ntype[i] = "2"

    item["text"]      = " ".join(new_text)
    item["phoneme"]   = " ".join(new_phon)
    item["note_type"] = " ".join(new_ntype)

    out_path.write_text(json.dumps(template, indent=2))
    return out_path

# ---------------- SoulX inference ------------------------------------------

def _soulx_python() -> str:
    """Pick the Python interpreter used to invoke SoulX cli.inference.

    If the SoulX install has its own venv (local dev), use that.
    Otherwise (HF Spaces, Docker), fall back to the current interpreter.
    """
    import sys
    venv_py = SOULX_ROOT / "venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else sys.executable


def _detect_device() -> str:
    """Pick CUDA when available (HF ZeroGPU), else CPU.

    SINGER_DEVICE env var overrides ('cuda' | 'cpu' | 'mps').
    """
    forced = os.environ.get("SINGER_DEVICE")
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def soulx_render(target_meta: Path, save_dir: Path,
                 n_steps: int | None = None, seed: int | None = None) -> Path:
    """Invoke SoulX cli.inference; returns the produced generated.wav path.

    n_steps: number of CFM diffusion steps. None = use config default (32).
    seed:    pins PyTorch RNG for reproducible renders. None = stochastic.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    device = _detect_device()
    cmd = [
        _soulx_python(), "-m", "cli.inference",
        "--device", device,
        "--model_path", str(SOULX_ROOT / "pretrained_models/SoulX-Singer/model.pt"),
        "--config",     str(SOULX_ROOT / "soulxsinger/config/soulxsinger.yaml"),
        "--prompt_wav_path",      str(PROMPT_WAV),
        "--prompt_metadata_path", str(PROMPT_META),
        "--target_metadata_path", str(target_meta),
        "--phoneset_path", str(SOULX_ROOT / "soulxsinger/utils/phoneme/phone_set.json"),
        "--save_dir", str(save_dir),
        "--pitch_shift", "0",
    ]
    if device == "cuda":
        cmd.append("--fp16")
    if n_steps is not None:
        cmd += ["--n_steps", str(n_steps)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    subprocess.run(cmd, cwd=str(SOULX_ROOT), check=True)
    out = save_dir / "generated.wav"
    if not out.exists():
        raise RuntimeError(f"SoulX did not produce {out}")
    return out

# ---------------- ffmpeg mix -----------------------------------------------

def mix_with_accompaniment(vocal: Path, out_path: Path,
                           voc_gain: float = 1.2, acc_gain: float = 0.9) -> Path:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(ACCOMP_WAV),
        "-i", str(vocal),
        "-filter_complex",
        f"[0:a]volume={acc_gain}[acc]; [1:a]volume={voc_gain}[voc]; "
        f"[voc][acc]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[a]",
        "-map", "[a]", "-ar", "44100", "-ac", "2",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path

# ---------------- public entry point ---------------------------------------

def _cache_key(syllables: Sequence[str], n_steps: int | None,
               melisma_mode: str = "default") -> str:
    """Hash the inputs so repeat renders with identical params skip the GPU.

    Lower-cased and stripped so 'BUE' == 'bue ' for cache purposes.
    melisma_mode is only included when not 'default' so existing baked
    caches (which used the implicit default) stay valid.
    """
    import hashlib
    norm = "|".join(s.strip().lower() for s in syllables) + f"::{n_steps}"
    if melisma_mode != "default":
        norm += f"::{melisma_mode}"
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


def render(syllables: Sequence[str], n_steps: int | None = None,
           melisma_mode: str = "default", seed: int | None = None) -> str:
    """Render `syllables` (e.g. ['bue','nos','di','as']) into a mixed cover wav.

    n_steps: CFM diffusion steps. None = SoulX default (32). 8 ≈ 4× faster, 64 ≈ 2× slower.
    melisma_mode: 'off' | 'default' | 'long' — see build_target_metadata.
    Returns absolute path. Caller is responsible for serving / cleaning up.

    Cached: same (syllables, n_steps, melisma_mode) returns the previously
    rendered wav, skipping SoulX entirely. Important on HF ZeroGPU where each
    call reserves duration against the user's daily quota.
    """
    syllables = [s.strip() for s in syllables if s and s.strip()]
    if not syllables:
        raise ValueError("provide at least one non-empty syllable")

    key = _cache_key(syllables, n_steps, melisma_mode)
    # Layer 1: assets/cache — baked into the repo so popular presets are
    # served instantly without burning ZeroGPU quota on the first visitor.
    baked = ASSETS / "cache" / f"{key}_cover.wav"
    if baked.exists():
        print(f"[render] baked cache hit: {baked}")
        return str(baked)
    # Layer 2: per-container working dir — populated on the fly during the
    # container's lifetime; resets on rebuild.
    cached = WORK / f"{key}_cover.wav"
    if cached.exists():
        print(f"[render] runtime cache hit: {cached}")
        return str(cached)

    job_dir = WORK / key
    job_dir.mkdir(parents=True, exist_ok=True)
    target_meta = build_target_metadata(syllables, job_dir / "target.json",
                                        melisma_mode=melisma_mode)
    vocal       = soulx_render(target_meta, job_dir / "vocal",
                                n_steps=n_steps, seed=seed)
    mixed       = mix_with_accompaniment(vocal, cached)
    return str(mixed)


if __name__ == "__main__":
    import sys
    syl = sys.argv[1:] or ["bue", "nos", "di", "as"]
    print(render(syl))
