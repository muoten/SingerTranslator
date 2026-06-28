"""Render + bake a quality cached demo for the current beat_it DEMOS lyric.

Pipeline (all the post-processing we settled on, in one place):
  build target (locked c1.5 grid) -> render N seeds -> score each by TIMBRE-sim
  (validated naturalness metric) -> pick the best -> trim the trailing synth tail
  to the last sung note (+0.2s +fade) -> mix (vocal under accompaniment, -1.5 dBFS
  ceiling) -> bake mix + trimmed vocal into assets/beat_it/cache/.

  SINGER_DEVICE=cpu vendor/SoulX-Singer/venv/bin/python scripts/bake_beat_it_demo.py \
      > sources/beat_it/bake.log 2>&1 &
"""
from __future__ import annotations
import sys, time, json, hashlib, shutil
from pathlib import Path
import numpy as np, soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import soulx_freelyrics as fl
import singer
import score_beat_it as sb

SONG = "beat_it"
SEEDS = list(range(6))
VOC_GAIN, ACC_GAIN, CEILING_DB = 0.6, 0.8, -1.5
OUT = ROOT / "_tmp_beat_it_bake"; OUT.mkdir(exist_ok=True)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    lines = fl.DEMOS[SONG]
    ok, words = fl.check(lines, SONG)
    if not ok:
        raise SystemExit("demo lyric doesn't fit template")
    tgt = OUT / "target.json"
    fl.build_target(words, tgt, song=SONG)
    prompt = str(ROOT / "assets" / SONG / "prompt.wav")

    # 1) seed search by timbre
    best = None
    for s in SEEDS:
        sdir = OUT / f"seed{s}"; sdir.mkdir(exist_ok=True)
        voc = sdir / "generated.wav"
        if not voc.exists():
            log(f"seed {s}: render ...")
            singer.soulx_render(tgt.resolve(), sdir.resolve(), n_steps=32, seed=s, song=SONG)
        t = sb.timbre_sim(str(voc), prompt)
        log(f"seed {s}: timbre={t:.4f}")
        if best is None or t > best[1]:
            best = (s, t, voc)
    log(f"best seed {best[0]} timbre={best[1]:.4f}")

    # 2) trim trailing synth tail to the last sung note
    d = json.loads(tgt.read_text())[0]
    dur = [float(x) for x in d["duration"].split()]
    typ = [int(x) for x in d["note_type"].split()]
    on = np.concatenate([[0], np.cumsum(dur)])
    last_sung = max(i for i, t in enumerate(typ) if t != 1)
    end = on[last_sung + 1] + 0.20
    y, sr = sf.read(best[2])
    y = y[:int(end * sr)].copy()
    f = int(0.12 * sr); y[-f:] *= np.linspace(1, 0, f)
    voc_trim = OUT / "vocal_final.wav"; sf.write(voc_trim, y, sr)
    log(f"trimmed vocal to {end:.2f}s")

    # 3) mix (vocal under accompaniment) + ceiling
    mix = OUT / "mix_final.wav"
    singer.mix_with_accompaniment(voc_trim, mix, voc_gain=VOC_GAIN, acc_gain=ACC_GAIN, song=SONG)
    m, sr2 = sf.read(mix)
    m = m * (10 ** (CEILING_DB / 20) / np.max(np.abs(m)))
    sf.write(mix, m, sr2)

    # 4) bake
    lyric = "/".join(" ".join(l.lower().split()) for l in lines)
    key = hashlib.sha256(f"{SONG}|{lyric}::32::r0".encode()).hexdigest()[:16]
    cdir = ROOT / "assets" / SONG / "cache"
    for old in cdir.glob("fl_*"):
        old.unlink()
    shutil.copy(mix, cdir / f"fl_{key}_mix.wav")
    shutil.copy(voc_trim, cdir / f"fl_{key}_vocal.wav")
    log(f"BAKED demo key={key} (seed {best[0]}, timbre {best[1]:.4f}) -> {cdir}")


if __name__ == "__main__":
    main()
