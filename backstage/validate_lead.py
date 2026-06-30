"""validate_lead.py — solo-vs-chorus flag for the karaoke-isolated lead.

Some choruses are sung by the lead IN UNISON WITH A CHOIR (e.g. Heal the World's
"heal the world" hook). No separator can pull a clean solo lead out of unison choir,
so the isolated lead is a smeared blend -> garbled grid -> bad render. This catches
that RIGHT AFTER preproc, before wasting a render/bake.

Signal (direct, not a pitch proxy): the karaoke step emits the LEAD (vocal.wav) and
the REMOVED backing (acc.wav). backing/(lead+backing) energy = how much other-vocal
surrounded the lead. Solo lead -> low; unison choir -> high. Activity-gated; also
localizes the worst window.

Requires preproc run with --vocal_sep True (so acc.wav exists). Threshold is
PROVISIONAL (calibrated n=3: BoW solo-lead 28%, Rock With You 63% acceptable-by-ear
borderline, Heal-the-World 73% genuinely choral); ADVISORY only — it warns at preproc
and (combined with the scat gate) gates demo at bake. Refine as more songs build.

  python backstage/validate_lead.py <song> ...     # or a vocal.wav + acc.wav pair
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
CHORAL = 0.68      # backing/total >= this = likely choral (n=3: 28 solo, 63 ok, 73 choral)
WIN = 2.0
HOP = 0.05


def _env(y, sr):
    h = int(HOP * sr)
    return np.array([np.sqrt(np.mean(y[i:i + h] ** 2) + 1e-12) for i in range(0, len(y) - h, h)])


def evaluate(song=None, lead=None, acc=None):
    if lead is None:
        lead = ROOT / "sources" / song / "preproc" / "vocal.wav"
        acc = ROOT / "sources" / song / "preproc" / "acc.wav"
    lead, acc = Path(lead), Path(acc)
    if not (lead.exists() and acc.exists()):
        return None
    l, sr = sf.read(lead); l = l.mean(1) if l.ndim > 1 else l
    a, _ = sf.read(acc); a = a.mean(1) if a.ndim > 1 else a
    n = min(len(l), len(a))
    le, ae = _env(l[:n], sr), _env(a[:n], sr)
    m = min(len(le), len(ae)); le, ae = le[:m], ae[:m]
    ratio = ae / (le + ae + 1e-9)
    act = (le + ae) > 0.3 * np.mean(le + ae)
    overall = float(np.mean(ratio[act])) if act.any() else 0.0
    # worst WIN-second window + its location
    w = int(WIN / HOP); worst, at = 0.0, 0.0
    for i in range(0, max(1, m - w)):
        sl = act[i:i + w]
        if sl.sum() > w * 0.5:
            v = float(np.mean(ratio[i:i + w][sl]))
            if v > worst:
                worst, at = v, i * HOP
    return {"backing_ratio": overall, "worst_window": worst, "worst_at": round(at, 1),
            "verdict": "CHORAL" if overall >= CHORAL else "SOLO"}


def main():
    songs = sys.argv[1:] or [p.parent.parent.name for p in (ROOT / "sources").glob("*/preproc/acc.wav")]
    print(f"{'song':26s} {'backing/total':>13} {'worst 2s @':>14}  verdict")
    for s in songs:
        r = evaluate(s)
        if r is None:
            print(f"{s:26s}  (no acc.wav — run preproc with --vocal_sep True)"); continue
        print(f"{s:26s} {r['backing_ratio']*100:11.0f}% {r['worst_window']*100:8.0f}% @{r['worst_at']:>4.1f}s  {r['verdict']}")


if __name__ == "__main__":
    main()
