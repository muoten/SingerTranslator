"""Unattended autoeval sweep for the Beat It render.

Varies the f0-clamp width (the knob we've been guessing at by ear), and for each:
  reconstruct grid (BEATIT_CLAMP_SEMI) -> build target -> SoulX render -> score.
Scores via scripts/score_beat_it.py:
  WORDS   per-slot phoneme recognition (language-aware matching)  -> understandability
  MELODY  F0-CORR (torchcrepe vs grid)                            -> melody fidelity
Ranks by (words_matched, corr) and prints the winner. Resumable: skips a variant
whose generated.wav already exists. Crash-safe; logs each step.

Run (venv python so transformers/torchcrepe are present):
  SINGER_DEVICE=cpu vendor/SoulX-Singer/venv/bin/python scripts/sweep_beat_it.py \
      > sources/beat_it/sweep.log 2>&1 &
  tail -f sources/beat_it/sweep.log
"""
from __future__ import annotations
import os, sys, json, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import soulx_freelyrics as fl
import singer
import score_beat_it as sb

SONG = "beat_it"
OUT = ROOT / "_tmp_beat_it_sweep"
OUT.mkdir(exist_ok=True)

# f0-clamp variants to try (semitone width; "raw"=untouched, "0"=flatten-to-median)
VARIANTS = [("raw", "raw"), ("c2.0", "2.0"), ("c1.5", "1.5"),
            ("c1.0", "1.0"), ("c0.5", "0.5"), ("flat", "0")]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ok, words = fl.check(fl.DEMOS[SONG], SONG)
    if not ok:
        raise SystemExit("demo lyric doesn't fit template")
    recon = str(ROOT / "scripts" / "reconstruct_beat_it_grid.py")

    results = []
    for name, semi in VARIANTS:
        sdir = OUT / name
        sdir.mkdir(exist_ok=True)
        voc = sdir / "generated.wav"
        tgt = sdir / "target.json"
        if not voc.exists():
            log(f"{name}: reconstruct (clamp={semi}) + build + render ...")
            env = dict(os.environ, BEATIT_CLAMP_SEMI=semi)
            subprocess.run([sys.executable, recon], env=env, check=True,
                           capture_output=True)
            fl.build_target(words, tgt, song=SONG)             # reads the reconstructed grid
            singer.soulx_render(tgt.resolve(), sdir.resolve(), n_steps=32, song=SONG)
        else:
            log(f"{name}: cached render, scoring only")
        log(f"{name}: scoring ...")
        r = sb.score(str(voc), str(tgt))
        results.append((name, semi, r))
        log(f"{name}: WORDS {r['words_matched']}/{r['words_total']}  "
            f"MELODY corr={r['corr']:.3f} semi_rmse={r['semi_rmse']:.2f}")
        (sdir / "score.json").write_text(json.dumps(
            {k: v for k, v in r.items() if k != "rows"}, indent=2))

    results.sort(key=lambda x: (x[2]["words_matched"],
                                x[2]["corr"] if x[2]["corr"] == x[2]["corr"] else -9), reverse=True)
    log("==== RANKED ====")
    for name, semi, r in results:
        log(f"  {name:5} clamp={semi:4}  words={r['words_matched']:>2}/{r['words_total']}  "
            f"corr={r['corr']:.3f}")
    best = results[0]
    log(f"WINNER: {best[0]} (clamp={best[1]}) -> {OUT / best[0] / 'generated.wav'}")


if __name__ == "__main__":
    main()
