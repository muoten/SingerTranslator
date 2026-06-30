"""validate_grid.py — READ-ONLY linter for chorus_target grids. Flags note
segmentation / voice-demux artifacts so we know which grids are dirty BEFORE any
cleaning. Mutates nothing — pair with a separate refinement step.

Artifact categories (a real sung note has a pitch, lasts long enough to sing, and
is mostly voiced):
  IMPOSSIBLE  type in {2,3} (sung) but pitch == 0           -> definitionally bogus
  MICRO       sung note shorter than MIN_NOTE                -> over-segmentation fragment
  NOISE       sung note with f0-voiced fraction < F0_FRAC    -> breath/bleed, not sung

Usage:
  python scripts/validate_grid.py                # all songs (assets/*/chorus_target.json)
  python scripts/validate_grid.py <song> ...     # specific songs
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_NOTE = 0.12     # s; below this a "note" is a fragment
F0_FRAC = 0.50      # min fraction of a sung note's frames that must be voiced


def analyze(tgt: Path):
    d = json.loads(tgt.read_text())[0]
    ds = [float(x) for x in d["duration"].split()]
    ty = d["note_type"].split()
    pit = [int(x) for x in d["note_pitch"].split()]
    f0 = [float(x) for x in d["f0"].split()]
    fps = len(f0) / sum(ds) if sum(ds) else 0
    imp, micro, noise, sung = [], [], [], 0
    t = 0.0
    for i, (dd, nt, p) in enumerate(zip(ds, ty, pit)):
        if nt in ("2", "3"):
            sung += 1
            a, b = int(t * fps), int((t + dd) * fps)
            seg = f0[a:b]
            vf = (sum(1 for x in seg if x > 0) / len(seg)) if seg else 0.0
            if p == 0:
                imp.append((i, round(t, 2), dd))
            elif dd < MIN_NOTE:
                micro.append((i, round(t, 2), dd))
            elif vf < F0_FRAC:
                noise.append((i, round(t, 2), round(vf, 2)))
        t += dd
    return sung, imp, micro, noise


def main():
    args = sys.argv[1:]
    songs = args or sorted(p.parent.name for p in (ROOT / "assets").glob("*/chorus_target.json"))
    print(f"{'song':26s} {'sung':>4} {'IMPOSSIBLE':>10} {'MICRO':>6} {'NOISE':>6}  verdict")
    for s in songs:
        tgt = ROOT / "assets" / s / "chorus_target.json"
        if not tgt.exists():
            print(f"{s:26s}  (no chorus_target.json)"); continue
        sung, imp, micro, noise = analyze(tgt)
        dirty = len(imp) + len(micro) + len(noise)
        verdict = "CLEAN" if dirty == 0 else (f"DIRTY ({dirty})" if (len(imp) or dirty > sung * 0.25) else "minor")
        print(f"{s:26s} {sung:>4} {len(imp):>10} {len(micro):>6} {len(noise):>6}  {verdict}")
        if imp:
            print(f"    IMPOSSIBLE (sung but pitch 0): {imp}")
    print(f"\nthresholds: MIN_NOTE={MIN_NOTE}s  F0_FRAC={F0_FRAC}")


if __name__ == "__main__":
    main()
