"""Score every currently-baked preset against the current metric and the
top of its pool. Flags any bake that has slipped >15% behind the new top.

Run this after EVERY metric change to catch silent regressions:

    python scripts/validate_bakes.py

Exit code 0 if all bakes pass. Non-zero if any bake fails the 0.85 ratio.
Output is a table — keep the bake or re-evaluate.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import singer
import eval_rerank_lang as erl

# Phrase → (cache key, source threshold, preset label).
# Cache keys must match singer._cache_key(syllables, 16, "default").
# Threshold is the melisma threshold the bake source was rendered at —
# used to score the bake against its OWN slot timings (otherwise we'd
# evaluate a thr=0.20 render against thr=0.30 slots and get spurious
# regressions, as observed 2026-05-14).
BAKES = {
    "buenos_dias":     ("64f4f7bb8b14f1aa", 0.30, "bue-nos di-as"),
    "happy_birthday":  ("932f839b4a224606", 0.30, "hap-pee birth-day"),
    "buenas_tardes":   ("a92d6bb1a4a3a99c", 0.20, "bue-nas tar-des"),
    "llueve_mucho":    ("a0a62483c200280f", 0.20, "llue-ve mu-cho"),
    # buenas_noches and mola_mazo are not currently baked — skipped.
}

# 0.85 = "still within 15% of the new top". Below this, the metric has
# moved away from this bake and we should re-evaluate.
RATIO_THRESHOLD = 0.85


STANDARD_THRESHOLDS = {0.20, 0.30, 0.40}


def score_top_of_pool(phrase_name: str) -> tuple[float, str]:
    """Return (top_new, top_fname) under the current metric, restricted to
    SINGLE renders at STANDARD thresholds (0.20, 0.30, 0.40). Composites
    are excluded (often ear-rejected even when metric-#1). In-between
    thresholds (0.25, 0.35) are also excluded — they were empirically
    shown to add cost without surfacing better candidates (2026-05-14).
    """
    import json
    import re
    erl.main(phrase_name)
    jp = Path(f"/tmp/aichael_{phrase_name}_results.jsonl")
    thr_pat = re.compile(r"_thr(\d+)_seed")
    for line in jp.read_text().splitlines():
        rec = json.loads(line)
        if not rec["label"].startswith("SINGLE"):
            continue
        m = thr_pat.search(rec["fname"])
        if not m:
            continue
        thr = int(m.group(1)) / 100
        if thr in STANDARD_THRESHOLDS:
            return rec["new"], rec["fname"]
    raise RuntimeError(f"no SINGLE rows at standard thresholds in {jp}")


def score_one_file(phrase_name: str, audio_path: Path, source_thr: float) -> float:
    """Score a specific file as if it were in the pool, using the slot
    timings of the threshold the bake was rendered at."""
    cfg = erl.PHRASES[phrase_name]
    lang, syllables = cfg["lang"], cfg["syllables"]
    from transformers import AutoProcessor, AutoModelForCTC
    proc = AutoProcessor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    mdl = AutoModelForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    mdl.eval()
    slots = erl.build_slots(syllables, source_thr)
    ws, _ = erl.score(proc, mdl, audio_path, slots, lang, len(set(syllables)))
    # w1-weighted-2x formula — must match eval_rerank_lang's `new`.
    return (2 * ws[0] + ws[1] + ws[2] + ws[3]) / 5


def main():
    print(f"{'phrase':<18}  {'bake_score':>10}  {'top_score':>10}  "
          f"{'ratio':>6}  status")
    print("-" * 60)
    failures = []
    for phrase, (key, source_thr, label) in BAKES.items():
        baked = Path(singer.__file__).parent / "assets" / "cache" / f"{key}_cover.wav"
        if not baked.exists():
            print(f"{phrase:<18}  (no bake at {baked.name})")
            continue
        top_new, top_fname = score_top_of_pool(phrase)
        bake_new = score_one_file(phrase, baked, source_thr)
        ratio = bake_new / top_new if top_new > 0 else 0.0
        status = "OK" if ratio >= RATIO_THRESHOLD else "REGRESSION"
        if status == "REGRESSION":
            failures.append(phrase)
        print(f"{phrase:<18}  {bake_new:>10.3f}  {top_new:>10.3f}  "
              f"{ratio:>6.2%}  {status}")
    if failures:
        print(f"\nFAIL: {len(failures)} bake(s) below {RATIO_THRESHOLD:.0%} of pool top: "
              f"{', '.join(failures)}")
        sys.exit(1)
    print(f"\nPASS: all bakes within {1-RATIO_THRESHOLD:.0%} of pool top.")


if __name__ == "__main__":
    main()
