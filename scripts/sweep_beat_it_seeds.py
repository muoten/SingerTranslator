"""Best-of-N seed search for higher timbre-naturalness on the Beat It c1.5 grid.

The grid/clamp is fixed (the validated ±1.5 winner). SoulX is stochastic, so we
render N seeds, score each by TIMBRE-sim to MJ's prompt (the validated chipmunk
metric, fast), rank, and run the Whisper intelligibility floor on the top few.
Goal: beat the current c1.5 timbre (0.9009). Resumable (skips rendered seeds).

  SINGER_DEVICE=cpu vendor/SoulX-Singer/venv/bin/python scripts/sweep_beat_it_seeds.py \
      > sources/beat_it/seedsweep.log 2>&1 &
  tail -f sources/beat_it/seedsweep.log
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import soulx_freelyrics as fl
import singer
import score_beat_it as sb

SONG = "beat_it"
SEEDS = list(range(30))      # 0-9 already rendered; 10-29 new
BASELINE_MIN = 0.502         # seed 5 min-window (the bar to beat)
OUT = ROOT / "_tmp_beat_it_seeds"
OUT.mkdir(exist_ok=True)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ok, words = fl.check(fl.DEMOS[SONG], SONG)
    if not ok:
        raise SystemExit("demo doesn't fit template")
    tgt = OUT / "target.json"
    fl.build_target(words, tgt, song=SONG)          # the locked c1.5 grid
    prompt = str(ROOT / "assets" / SONG / "prompt.wav")

    scored = []
    for s in SEEDS:
        sdir = OUT / f"seed{s}"; sdir.mkdir(exist_ok=True)
        voc = sdir / "generated.wav"
        if not voc.exists():
            log(f"seed {s}: render ...")
            singer.soulx_render(tgt.resolve(), sdir.resolve(), n_steps=32, seed=s, song=SONG)
        wins, mn, mean = sb.timbre_per_window(str(voc), prompt)   # objective = min window
        glob = sb.timbre_sim(str(voc), prompt)
        scored.append((s, mn, mean, glob, voc))
        log(f"seed {s}: MIN={mn:.3f} mean={mean:.3f} global={glob:.3f}"
            + ("  ** beats seed5 min **" if mn > BASELINE_MIN else ""))

    scored.sort(key=lambda x: -x[1])   # rank by worst-window (min)
    log("==== ranked by MIN-window timbre (worst spot) ====")
    for s, mn, mean, glob, _ in scored[:10]:
        log(f"  seed {s:>2}: min={mn:.3f}  mean={mean:.3f}  global={glob:.3f}")
    # whisper floor on top 3 by min
    log("---- whisper recall on top 3 ----")
    best = None
    for s, mn, mean, glob, voc in scored[:3]:
        wrec, wtot = sb.whisper_recall(str(voc), str(tgt))
        log(f"  seed {s}: min={mn:.3f} global={glob:.3f} whisper={wrec}/{wtot}")
        if best is None and wrec >= 14:
            best = (s, mn, glob, voc, wrec, wtot)
    top = scored[0]
    log(f"BEST min-window: seed {top[0]} min={top[1]:.3f} (vs seed5 {BASELINE_MIN}) "
        f"global={top[3]:.3f} -> {top[4]}")
    if best:
        log(f"BEST passing whisper>=14: seed {best[0]} min={best[1]:.3f} "
            f"whisper={best[4]}/{best[5]} -> {best[3]}")


if __name__ == "__main__":
    main()
