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
    python backstage/render_seeds.py THR SYL1 SYL2 SYL3 SYL4 SEED1 SEED2 ...
Example:
    python backstage/render_seeds.py 0.40 llue ve mu cho 0 1 2 3
"""
from __future__ import annotations
import os
import shutil
import sys
import contextlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import singer

SONG = "thriller"  # multi-song support: parameterize when sweeping other songs
CACHE_DIR = singer.WORK / SONG
BAKED_DIR = singer.cache_dir(SONG)
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


def score_and_report(vocals, song):
    """Rank the swept seeds by melody-following (F0-CORR) + timbre (SIM).

    This is an ADDITIONAL axis, not a replacement for the perceived
    (intelligibility) metric — a seed can follow the melody yet be mumbled.
    Cross-check against perceived before baking. Disable with RENDER_SEEDS_SCORE=0;
    SCORE_CREPE=full for a slower/accurate pass (default 'tiny').
    """
    if os.environ.get("RENDER_SEEDS_SCORE", "1") == "0" or not vocals:
        return
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import score_melody_timbre as smt
        grid = str(singer.template_json(song))
        ref = str(singer.prompt_wav(song))
        model = os.environ.get("SCORE_CREPE", "tiny")
        print(f"\n=== melody+timbre scoring (F0-CORR / SIM, crepe={model}) ===", flush=True)
        target = smt.grid_target_midi(grid)
        ref_emb = smt.xvector(ref)
        rows = []
        for seed, vp in vocals:
            fc = smt.f0_corr(smt.take_f0_midi(str(vp), model=model), target)
            s = smt.sim(str(vp), ref_emb)
            comb = 0.6 * max(fc["corr"], 0.0) + 0.4 * max(s, 0.0)
            rows.append((comb, seed, fc["corr"], s, fc["semi_rmse"], fc["voiced_overlap"]))
        rows.sort(reverse=True)
        print(f"{'seed':>5} {'COMB':>6} {'F0-CORR':>8} {'SIM':>6} {'semiRMSE':>9} {'cover':>6}")
        print("-" * 50)
        for comb, seed, corr, s, rmse, cov in rows:
            flag = "  <- low SIM (timbre drift)" if s < 0.75 else ""
            print(f"{seed:>5} {comb:6.3f} {corr:8.3f} {s:6.3f} {rmse:9.2f} {cov * 100:5.0f}%{flag}")
        b = rows[0]
        print(f"\nBEST melody+timbre: seed {b[1]}  (F0-CORR={b[2]:.3f}, SIM={b[3]:.3f})")
        print("NOTE: melody+timbre only — cross-check perceived (intelligibility) before baking.")
    except Exception as e:  # noqa: BLE001 — never let scoring lose the renders
        print(f"[score] skipped (scoring failed: {e})")


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

    vocals = []
    with baked_aside(key):
        for seed in seeds:
            clear_runtime_cache(key)
            print(f"\n=== {thr_tag} seed={seed} ===", flush=True)
            wav = singer.render(syllables, n_steps=N_STEPS, seed=seed)
            dst = Path(f"/Users/milhouse/Downloads/aichael_{phrase_tag}_HYPOTHESIS_{thr_tag}_seed{seed}.wav")
            shutil.copy(wav, dst)
            print(f"  → {dst}")
            # Also save the dry vocal stem (pre-mix) for melody/timbre scoring.
            voc_src = singer.WORK / SONG / key / "vocal" / "generated.wav"
            if voc_src.exists():
                voc_dst = dst.with_name(dst.stem + "_VOCAL.wav")
                shutil.copy(voc_src, voc_dst)
                vocals.append((seed, voc_dst))
                print(f"  → {voc_dst}")

    score_and_report(vocals, SONG)


if __name__ == "__main__":
    main()
