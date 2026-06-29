"""verify_neutral.py — STANDARD synthesis/leakage verification step (song-agnostic).

Renders the song's chorus grid on a neutral 'la' (en_L-AA1), keeping the MEASURED
f0/timing/pitch untouched, using whatever prompt the song currently ships
(assets/<song>/prompt.{wav,json}) — it does NOT overwrite the prompt.

WHY 'la' and not the real lyrics (the point of this step):
  - Copyright-safe: it never sings the original lyrics.
  - Leakage detector: you ASKED for 'la la la', so any original words/vowels you
    hear are unambiguously prompt audio leaking. With the real lyrics you can't
    tell correct synthesis apart from leakage (both produce the original words).

EAR VERDICT:
  PASS  -> clean 'la la la' that follows the chorus melody, in the song's timbre.
  FAIL  -> original words/vowels bleed through  => prompt leaks; build/swap to an
           anti-leakage (verse) prompt and re-run.

Usage:
  SINGER_DEVICE=cpu python scripts/verify_neutral.py --song smooth_criminal [--play]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import singer  # noqa: E402


def build_neutral_target(song: str, out_path: Path,
                         melisma_mode: str = "default") -> tuple[Path, int]:
    """Map a neutral 'la' onto the grid via the SAME path the real cover uses.

    Using build_target_metadata (not a 1:1 text rewrite) means rests (pitch 0 /
    note_type 1) stay <SP>, melismas (note_type 3) are held, and ultra-short
    slots auto-melisma — so the neutral render reflects the real render's
    articulation/speed instead of firing a fresh 'la' on every text slot.
    """
    singer.build_target_metadata(["la"], out_path, melisma_mode=melisma_mode, song=song)
    item = json.loads(out_path.read_text())[0]
    n_la = sum(1 for t in item["text"].split() if t != "<SP>")
    return out_path, n_la


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", required=True)
    ap.add_argument("--n_steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--melisma_mode", default="default", choices=["default", "off"],
                    help="'default' holds melismas/short slots (real-render articulation); "
                         "'off' fires a fresh 'la' on every slot >=0.20s")
    ap.add_argument("--play", action="store_true", help="afplay the mix when done (macOS)")
    args = ap.parse_args()
    song = args.song

    pw, pj = singer.prompt_wav(song), singer.prompt_meta(song)
    if not pw.exists() or not pj.exists():
        raise SystemExit(f"no shipping prompt for {song}: expected {pw} and {pj}")

    out = ROOT / "_verify_neutral" / song
    out.mkdir(parents=True, exist_ok=True)
    target, n_voiced = build_neutral_target(song, out / "neutral_target.json",
                                            melisma_mode=args.melisma_mode)
    print(f"[neutral] {n_voiced} 'la' onsets (melisma_mode={args.melisma_mode})  ({target})")
    print(f"[prompt ] using shipping prompt (NOT overwritten): {pw.name}")

    vocal = singer.soulx_render(target.resolve(), out.resolve(),
                               n_steps=args.n_steps, seed=args.seed, song=song)
    mix = singer.mix_with_accompaniment(Path(vocal), out / f"{song}_neutral_mix.wav", song=song)
    print(f"NEUTRAL_VOCAL: {vocal}")
    print(f"NEUTRAL_MIX:   {mix}")
    print("\nEAR VERDICT — PASS: clean on-melody 'la la la' in the song's voice. "
          "FAIL: original words bleed through (prompt leaks).")

    if args.play:
        import subprocess
        subprocess.run(["afplay", str(mix)])


if __name__ == "__main__":
    main()
