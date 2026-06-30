"""crappy_fragments.py — find the longest "crappy fragment" per song.

A render is crappy where it does NOT match the grid. Per ~50ms bin, on the baked
vocal vs the built target:
  PHANTOM   bin is a grid REST but the render is loud   (sound where silence is intended)
  OFFPITCH  bin is a grid NOTE, render is loud, but rendered pitch is >TOL semis off
  DEAD      bin is a grid NOTE (interior) but the render is silent (note not sung)
Contiguous crappy bins (small gaps bridged) form a crappy fragment. We report the
LONGEST per song + the list — the localized, ear-matching defect everything else missed.

  python backstage/crappy_fragments.py            # all baked songs
  python backstage/crappy_fragments.py <song> ...
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf, librosa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import soulx_freelyrics as fl  # noqa: E402

BIN = 0.05
LOUD = 0.40        # x global voiced rms = "sound present"
DEADQ = 0.12       # x global voiced rms = "silent"
TOL = 3            # semitones off intended = wrong pitch
BRIDGE = 0.12      # s; bridge crappy bins separated by <= this
MINFRAG = 0.15     # s; ignore fragments shorter than this


def grid_arrays(song, dur, n):
    ok, words = fl.check(fl.DEMOS[song], song)
    p = ROOT / "_crappy" / f"{song}.json"; p.parent.mkdir(exist_ok=True)
    fl.build_target(words, p, song=song)
    d = json.loads(p.read_text())[0]
    ds = [float(x) for x in d["duration"].split()]
    ty = d["note_type"].split(); pit = [int(x) for x in d["note_pitch"].split()]
    voiced = np.zeros(n, bool); ipitch = np.zeros(n); interior = np.zeros(n, bool)
    t = 0.0
    for dd, nt, pp in zip(ds, ty, pit):
        a, b = int(t / BIN), int((t + dd) / BIN)
        if nt != "1":
            voiced[a:b] = True; ipitch[a:b] = pp
            interior[a + 1:max(a + 1, b - 1)] = True   # drop note edges for DEAD test
        t += dd
    return voiced[:n], ipitch[:n], interior[:n]


# Demo-eligibility gate: total grid-mismatched ("crappy") audio. Calibrated on the
# first 7 songs — every accepted song < 5.0s, both hidden/rejected >= 5.0s (gap
# 4.4 -> 5.6). Revisit the threshold as the catalog grows.
GATE_S = 5.0


def evaluate(song, vocal_path=None):
    """Return {longest, total, n_frags, frags, verdict} for a song's baked vocal."""
    if vocal_path is None:
        voc = sorted((ROOT / "assets" / song / "cache").glob("fl_*vocal*.wav"))
        if not voc:
            return None
        vocal_path = voc[0]
    y, sr = sf.read(vocal_path); y = y.mean(1) if y.ndim > 1 else y
    dur = len(y) / sr
    hop = int(BIN * sr)
    rms = np.array([np.sqrt(np.mean(y[i:i + hop] ** 2) + 1e-12) for i in range(0, len(y) - hop, hop)])
    f0, _, _ = librosa.pyin(y.astype("float32"), fmin=80, fmax=600, sr=sr,
                            frame_length=1024, hop_length=hop)
    midi = librosa.hz_to_midi(f0)
    n = min(len(rms), len(midi))
    rms = rms[:n]; midi = midi[:n]
    voiced, ipitch, interior = grid_arrays(song, dur, n)
    gv = np.mean(rms[voiced & (rms > 0)]) if voiced.any() else rms.mean()
    crappy = np.zeros(n, bool); reason = [""] * n
    for i in range(n):
        loud = rms[i] > LOUD * gv
        if not voiced[i]:
            # PHANTOM = loud during a grid rest. NOTE: this over-flags songs that fill a
            # sparse grid with acceptable VOICED sustain (Rock With You, 9.45s yet sounds
            # fine). A voiced-fraction refinement to exclude sustains was tried (PH_VFRAC
            # 0.80/0.90) and REJECTED: BoW/SC's bad bleed is voiced at a similar level, so
            # any threshold that clears RWY also clears them. Treat RWY as a manual override.
            if loud:
                crappy[i] = True; reason[i] = "phantom"
        else:
            m = midi[i]
            if loud and ipitch[i] > 0 and m == m and abs(m - ipitch[i]) > TOL:
                crappy[i] = True; reason[i] = "offpitch"
            elif interior[i] and rms[i] < DEADQ * gv:
                crappy[i] = True; reason[i] = "dead"
    gb = int(BRIDGE / BIN)
    idx = np.where(crappy)[0]
    filled = crappy.copy()
    for a, b in zip(idx, idx[1:]):
        if 0 < b - a <= gb:
            filled[a:b] = True
    frags = []
    i = 0
    while i < n:
        if filled[i]:
            j = i
            while j < n and filled[j]:
                j += 1
            if (j - i) * BIN >= MINFRAG:
                rs = max(set(reason[i:j]) - {""}, key=lambda r: reason[i:j].count(r)) if any(reason[i:j]) else "?"
                frags.append((round(i * BIN, 2), round(j * BIN, 2), round((j - i) * BIN, 2), rs))
            i = j
        else:
            i += 1
    total = round(sum(f[2] for f in frags), 2)
    longest = max((f[2] for f in frags), default=0.0)
    return {"longest": longest, "total": total, "n_frags": len(frags),
            "frags": frags, "verdict": "ENABLE" if total < GATE_S else "HIDE"}


def main():
    songs = sys.argv[1:] or [s for s in fl.ORDERS
                             if list((ROOT / "assets" / s / "cache").glob("fl_*vocal*.wav"))]
    print(f"{'song':26s} {'longest':>8} {'#frags':>6} {'total':>6} {'gate':>7}  longest-fragment")
    out = []
    for song in songs:
        r = evaluate(song)
        if r is None:
            continue
        lf = max(r["frags"], key=lambda f: f[2]) if r["frags"] else None
        out.append((song, r))
        print(f"{song:26s} {r['longest']:>7.2f}s {r['n_frags']:>6} {r['total']:>5.2f}s "
              f"{r['verdict']:>7}  {lf}")
    print(f"\nranked by total crappy (gate {GATE_S}s):")
    for s, r in sorted(out, key=lambda kv: -kv[1]["total"]):
        print(f"  {s:26s} total={r['total']:.2f}s  {r['verdict']}")


if __name__ == "__main__":
    main()
