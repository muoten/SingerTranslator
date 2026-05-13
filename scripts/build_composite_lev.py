"""Build a per-window-best composite for a phrase.

Reads /tmp/aichael_<phrase>_results.jsonl (written by eval_rerank_lang.py)
and stitches the best-F1 source for each window with 50 ms equal-power
crossfades at 4 s boundaries. Output:
    ~/Downloads/<prefix>_composite_lev_xf.wav

Usage:
    python scripts/build_composite_lev.py <phrase_name>
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import PHRASES registry to resolve prefix per phrase.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "eval_rerank_lang", str(Path(__file__).parent / "eval_rerank_lang.py"))
erl = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(erl)  # type: ignore[union-attr]
PHRASES = erl.PHRASES

DL = Path("/Users/milhouse/Downloads")
N_WINDOWS = 4
FADE_MS = 50


def eqp(n):
    t = np.linspace(0, np.pi / 2, n)
    return np.cos(t) ** 2, np.sin(t) ** 2


def main(phrase_name: str):
    cfg = PHRASES[phrase_name]
    prefix = cfg["prefix"]
    jsonl_path = Path(f"/tmp/aichael_{phrase_name}_results.jsonl")
    if not jsonl_path.exists():
        sys.exit(f"missing {jsonl_path} — run eval_rerank_lang.py {phrase_name} first")

    recs = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    # filter to SINGLE renders only (don't compose composites)
    pattern = re.compile(rf"{re.escape(prefix)}_thr(\d+)_seed(\d+)\.wav$")
    singles = []
    for r in recs:
        m = pattern.match(r["fname"])
        if m:
            singles.append({**r, "thr": int(m.group(1)) / 100, "seed": int(m.group(2))})
    if not singles:
        sys.exit("no single renders found matching prefix")

    # Pick per-window best by F1
    picks = []
    print(f"Best per-window picks for {phrase_name}:")
    for w in range(N_WINDOWS):
        best = max(singles, key=lambda r: r["win_f1"][w])
        picks.append(best)
        print(f"  w{w+1}: thr={best['thr']:.2f} seed={best['seed']:>2}  F1={best['win_f1'][w]:.2f}  fname={best['fname']}")

    # Load reference for shape
    ref_path = DL / picks[0]["fname"]
    a0, sr = sf.read(str(ref_path), dtype="float32")
    total = a0.shape[0]
    win_len = total // N_WINDOWS
    fade_n = int(FADE_MS * 1e-3 * sr)
    if fade_n % 2: fade_n -= 1

    sources = []
    for p in picks:
        path = DL / p["fname"]
        a, s = sf.read(str(path), dtype="float32")
        assert s == sr, f"sr mismatch in {path.name}"
        if a.shape != a0.shape:
            if a.shape[0] < a0.shape[0]:
                pad = np.zeros((a0.shape[0] - a.shape[0],) + a0.shape[1:], dtype=a.dtype)
                a = np.concatenate([a, pad])
            else:
                a = a[: a0.shape[0]]
        sources.append(a)

    out = np.zeros_like(a0)
    boundaries = [w * win_len for w in range(N_WINDOWS + 1)]
    fo, fi = eqp(fade_n)
    if a0.ndim == 2:
        fo = fo[:, None]; fi = fi[:, None]

    for w in range(N_WINDOWS):
        b_left, b_right = boundaries[w], boundaries[w + 1]
        body_start = b_left + (fade_n // 2 if w > 0 else 0)
        body_end = b_right - (fade_n // 2 if w < N_WINDOWS - 1 else 0)
        out[body_start:body_end] = sources[w][body_start:body_end]
        if w < N_WINDOWS - 1:
            xs, xe = b_right - fade_n // 2, b_right + fade_n // 2
            out[xs:xe] = sources[w][xs:xe] * fo + sources[w + 1][xs:xe] * fi

    dst = DL / f"{prefix}_composite_lev_xf.wav"
    sf.write(str(dst), out, sr)
    print(f"\nComposite (LEV-aware): {dst}   {out.shape[0]/sr:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1])
