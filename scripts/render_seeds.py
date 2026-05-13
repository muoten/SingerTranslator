"""Render a phrase across multiple seeds at a chosen melisma threshold,
bypassing singer.render's cache.

singer.render keys its cache only on (syllables, n_steps, melisma_mode) —
seed AND melisma threshold are excluded. So back-to-back calls would hit
the cache after the first render. This script clears the matching cache
entry between renders to force fresh runs, and patches
singer.DEFAULT_AUTO_MELISMA_DUR so the chosen threshold actually applies.

Output filenames encode (syllables, threshold, seed) so different threshold
sweeps don't overwrite each other:
    aichael_<syllables>_HYPOTHESIS_thr<NNN>_seed<NNN>.wav

Usage:
    python scripts/render_seeds.py THR SYL1 SYL2 SYL3 SYL4 SEED1 SEED2 ...
Example:
    python scripts/render_seeds.py 0.40 llue ve mu cho 0 1 2 3
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import singer

CACHE_DIR = Path("/var/folders/kp/b53jtvb57nv_hkqnj76mdd2w0000gn/T/singer_renders")
N_STEPS = 16


def clear_cache(key: str):
    for p in CACHE_DIR.glob(f"{key}*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def main():
    args = sys.argv[1:]
    if len(args) < 6:
        print(__doc__); sys.exit(1)
    thr = float(args[0])
    syllables = args[1:5]
    seeds = [int(s) for s in args[5:]]

    singer.DEFAULT_AUTO_MELISMA_DUR = thr  # patch so build_target_metadata picks it up
    phrase_tag = "_".join(syllables)
    thr_tag = f"thr{int(thr * 100):03d}"
    key = singer._cache_key(syllables, N_STEPS, "default")  # type: ignore[attr-defined]
    print(f"Cache key: {key}  syllables={syllables}  thr={thr}")

    for seed in seeds:
        clear_cache(key)
        print(f"\n=== {thr_tag} seed={seed} ===", flush=True)
        wav = singer.render(syllables, n_steps=N_STEPS, seed=seed)
        dst = Path(f"/Users/milhouse/Downloads/aichael_{phrase_tag}_HYPOTHESIS_{thr_tag}_seed{seed}.wav")
        shutil.copy(wav, dst)
        print(f"  → {dst}")


if __name__ == "__main__":
    main()
