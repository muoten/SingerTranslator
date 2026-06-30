"""find_window.py — propose a clean chorus WINDOW whose start/end don't cut a word
or a beat.

A good cut point is where the LEAD is silent (a gap between sung phrases) AND it
falls on a beat (so the sliced accompaniment loops cleanly). We:
  1. beat-track the full mix (librosa) -> beat times,
  2. measure the demucs VOCAL envelope -> sung phrases (voiced runs),
  3. take whole phrases starting at/after --near up to ~--dur, snapping the first
     onset / last offset to the nearest beat.

CLI:   python backstage/find_window.py --song rock_with_you --near 108 --dur 16
import: propose(song, near, dur=16.0) -> {"start","end","tempo","n_phrases", ...}
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, soundfile as sf, librosa

ROOT = Path(__file__).resolve().parents[1]


def _load(p):
    y, sr = sf.read(p)
    if y.ndim > 1:
        y = y.mean(1)
    return y.astype("float32"), sr


def propose(song, near, dur=16.0, gap_thr=0.20):
    sep = ROOT / "sources" / song / "_sep" / "htdemucs" / f"{song}_full"
    voc, sr = _load(sep / "vocals.wav")
    mix, srm = _load(next((ROOT / "sources" / song).glob("*_full.wav")))

    tempo, beats = librosa.beat.beat_track(y=mix, sr=srm, units="time")
    tempo = float(np.atleast_1d(tempo)[0])

    rms = np.array([np.sqrt(np.mean(voc[i:i + int(0.02 * sr)] ** 2) + 1e-12)
                    for i in range(0, len(voc) - int(0.02 * sr), int(0.02 * sr))])
    cregion = rms[int(near / 0.02):int((near + dur) / 0.02)]
    med = float(np.median(cregion[cregion > 0])) if (cregion > 0).any() else rms.mean()
    thr = gap_thr * med

    voiced = rms > thr
    runs, i, n = [], 0, len(voiced)
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            runs.append([i * 0.02, j * 0.02]); i = j
        else:
            i += 1
    merged = []
    for a, b in runs:                                   # merge within-phrase dips
        if merged and a - merged[-1][1] < 0.20:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    phrases = [p for p in merged if p[1] - p[0] > 0.15]

    def snap(t):
        return float(min(beats, key=lambda x: abs(x - t)))

    chosen = [p for p in phrases if p[1] > near]
    if not chosen:
        return None
    first = chosen[0]
    lastoff = first[0]
    n_ph = 0
    for p in chosen:
        if p[1] - first[0] <= dur + 1.0:
            lastoff = p[1]; n_ph += 1
        else:
            break
    start, end = snap(first[0]), snap(lastoff)
    return {"start": round(start, 2), "end": round(end, 2), "tempo": tempo,
            "n_phrases": n_ph, "onset": first[0], "offset": lastoff,
            "beats_in": round((end - start) / (60.0 / tempo), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", required=True)
    ap.add_argument("--near", type=float, required=True)
    ap.add_argument("--dur", type=float, default=16.0)
    a = ap.parse_args()
    r = propose(a.song, a.near, a.dur)
    if not r:
        print("  no sung phrases after --near"); return
    print(f"  tempo ~{r['tempo']:.0f} BPM  whole phrases covered: {r['n_phrases']}")
    print(f"  onset {r['onset']:.2f}s -> START {r['start']:.2f}s   "
          f"offset {r['offset']:.2f}s -> END {r['end']:.2f}s")
    print(f"  PROPOSED WINDOW  {r['start']:.2f}:{r['end']:.2f}   "
          f"({r['end']-r['start']:.2f}s, {r['beats_in']} beats)")


if __name__ == "__main__":
    main()
