"""Render a phrase across multiple seeds at a chosen melisma threshold,
bypassing BOTH singer.render's runtime cache AND the baked preset cache.

singer.render has two cache layers, both keyed on
(syllables, n_steps, melisma_mode) only — seed AND melisma threshold are
excluded:
  Layer 1 (baked): assets/cache/<key>_cover.wav — checked FIRST, short-
                   circuits with no SoulX invocation. If the phrase has a
                   baked preset, EVERY seed/threshold returns the same file.
  Layer 2 (runtime): /var/folders/.../singer_renders/<key>* — populated
                     per container lifetime.

This script clears Layer 2 between renders and temporarily moves the
Layer 1 file aside (then restores it) so each render is genuinely fresh
at the requested seed and threshold.

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
import contextlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import singer

CACHE_DIR = Path("/var/folders/kp/b53jtvb57nv_hkqnj76mdd2w0000gn/T/singer_renders")
BAKED_DIR = Path(singer.__file__).parent / "assets" / "cache"
N_STEPS = 16


def clear_runtime_cache(key: str):
    for p in CACHE_DIR.glob(f"{key}*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


@contextlib.contextmanager
def baked_aside(key: str):
    """Temporarily rename baked preset out of the way; restore on exit.

    Without this, singer.render returns the baked file immediately and
    SoulX is never invoked — every seed/threshold produces identical
    output. The .bak suffix is restored even if rendering fails.
    """
    baked = BAKED_DIR / f"{key}_cover.wav"
    parked = BAKED_DIR / f"{key}_cover.wav.SWEEP_PARKED"
    moved = False
    if baked.exists():
        baked.rename(parked)
        moved = True
    try:
        yield
    finally:
        if moved and parked.exists():
            parked.rename(baked)


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

    with baked_aside(key):
        for seed in seeds:
            clear_runtime_cache(key)
            print(f"\n=== {thr_tag} seed={seed} ===", flush=True)
            wav = singer.render(syllables, n_steps=N_STEPS, seed=seed)
            dst = Path(f"/Users/milhouse/Downloads/aichael_{phrase_tag}_HYPOTHESIS_{thr_tag}_seed{seed}.wav")
            shutil.copy(wav, dst)
            print(f"  → {dst}")


if __name__ == "__main__":
    main()
