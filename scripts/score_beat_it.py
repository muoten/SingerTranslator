"""Autoeval for a Beat It render — no human listening needed.

Two independent signals on the VOCAL-only render (generated.wav, not the mix):

  MELODY  F0-CORR: torchcrepe F0 of the take vs the target grid's per-frame MIDI
          (Pearson corr + semitone RMSE over voiced overlap). High corr / low
          RMSE = the synth followed the intended melody.

  WORDS   per sung slot, transcribe its window with the wav2vec2-espeak phoneme
          CTC and check the slot's expected stressed vowel shows up. matched/total
          = understandability (reuses scripts/score_line_words.py logic).

Usage:
  python scripts/score_beat_it.py <vocal.wav> <target.json> [label]

Importable: score(voc, tgt) -> dict, loading models once (for sweep loops).
"""
from __future__ import annotations
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import soundfile as sf

import score_melody_timbre as smt
import score_line_words as slw
import eval_rerank_lang as erl   # language-aware IPA matching (validated)
import w2v_phoneme as wp         # phoneme model with the fixed-decode loader

_M = {}


def _model():
    if not _M:
        _M["proc"], _M["mdl"] = wp.load()
    return _M["proc"], _M["mdl"]


def _dedup(seq):
    out = []
    for p in seq:
        if not out or out[-1] != p:
            out.append(p)
    return out


def words_score(voc, tgt, lang="en"):
    """Per sung slot: transcribe its window, then check the slot's expected IPA
    sequence appears via eval_rerank_lang's skip-tolerant + equivalence matching
    (robust to the multilingual model's phoneme drift). matched/total."""
    y, sr = sf.read(voc)
    if y.ndim > 1:
        y = y.mean(1)
    if sr != 16000:
        import scipy.signal as sps
        y = sps.resample_poly(y, 16000, sr); sr = 16000
    proc, mdl = _model()
    cfg = erl.LANG_CONFIGS[lang]
    slots = slw.load_slots(tgt)
    matched, rows = 0, []
    for s in slots:
        a, b = int(s["start"] * sr), int(min(s["end"] + 0.06, len(y) / sr) * sr)
        hyp = erl.normalize_hyp_tokens(wp.transcribe(proc, mdl, y[a:b], sr).split())
        target = _dedup(erl.phon2ipa(s["phon"]))
        r, _ = erl.find_syllable(hyp, s["text"], target, 0, cfg)
        ok = r != -1
        matched += int(ok)
        rows.append((s["text"], "".join(target), " ".join(hyp), ok))
    return matched, len(slots), rows


def melody_score(voc, tgt):
    take = smt.take_f0_midi(voc)
    target = smt.grid_target_midi(tgt)
    return smt.f0_corr(take, target)


# ---- the validated signals -------------------------------------------------
# Naturalness (catches the chipmunk): WavLM speaker-embedding cosine-sim of the
# render vs MJ's prompt. Validated to rank flat (chipmunk) lowest, ±1.5 highest.
# Intelligibility floor: Whisper word-recall of the intended (target) words.
import torch

_WHISPER = {}


def timbre_sim(voc, prompt):
    ref = smt.xvector(prompt)
    emb = smt.xvector(voc)
    return float(torch.nn.functional.cosine_similarity(ref, emb, dim=-1).mean())


@torch.no_grad()
def _embed_array(wav16k):
    fe, model = smt._load_sv()
    inputs = fe([wav16k], sampling_rate=16000, return_tensors="pt", padding=True)
    return torch.nn.functional.normalize(model(**inputs).embeddings, dim=-1)


def timbre_per_window(voc, prompt, win=1.5, hop=0.5):
    """Per-window timbre-sim to the prompt. The WORST window (min) catches a
    localized hard artifact / chipmunk peak that the global mean would hide.
    Returns (windows[(t, sim)], min_sim, mean_sim)."""
    ref = smt.xvector(prompt)
    y, sr = sf.read(voc)
    if y.ndim > 1:
        y = y.mean(1)
    if sr != 16000:
        import scipy.signal as sps
        y = sps.resample_poly(y, 16000, sr); sr = 16000
    n, h = int(win * sr), int(hop * sr)
    wins = []
    for a in range(0, max(1, len(y) - n + 1), h):
        seg = y[a:a + n]
        if len(seg) < int(0.4 * sr):
            continue
        s = float((_embed_array(seg) * ref).sum())
        wins.append((round(a / sr, 2), s))
    vals = [s for _, s in wins] or [0.0]
    return wins, min(vals), sum(vals) / len(vals)


def _target_words(tgt):
    d = json.loads(Path(tgt).read_text())[0]
    return [w.rstrip("_") for w in d["text"].split() if w != "<SP>"]


def whisper_recall(voc, tgt):
    import re
    if not _WHISPER:
        import whisper
        _WHISPER["m"] = whisper.load_model("large")
    words = _target_words(tgt)
    txt = _WHISPER["m"].transcribe(voc, language="en")["text"].lower()
    hyp = re.findall(r"[a-z']+", txt)
    matched = sum(1 for w in words if w in hyp)
    return matched, len(words)


def score(voc, tgt, song="beat_it"):
    prompt = str(Path(__file__).resolve().parent.parent / "assets" / song / "prompt.wav")
    timb = timbre_sim(voc, prompt)
    wrec, wtot = whisper_recall(voc, tgt)
    return dict(timbre=timb, whisper_recall=wrec, whisper_total=wtot)


def main():
    voc, tgt = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else Path(voc).stem
    r = score(voc, tgt)
    print(f"\n=== {label} ===")
    print(f"TIMBRE (naturalness, vs MJ prompt): {r['timbre']:.4f}   higher=more MJ-like, less chipmunk")
    print(f"WHISPER recall (intelligibility floor): {r['whisper_recall']}/{r['whisper_total']}")


if __name__ == "__main__":
    main()
