"""Search for a Beat It demo lyric that the synth renders INTELLIGIBLY.

Generates green, sustainable-ending candidate lyrics (template: "<verb> the <adj>
[<adj>] <end>"), renders each with a fixed seed + the melisma-hold, transcribes the
vocal with Whisper, and scores by content-word RECALL (how many of the intended
words come back). Ranks; the best-transcribed lyric wins. Resumable.

  SINGER_DEVICE=cpu vendor/SoulX-Singer/venv/bin/python scripts/search_beat_it_lyric.py \
      > sources/beat_it/lyricsearch.log 2>&1 &
  tail -f sources/beat_it/lyricsearch.log
"""
from __future__ import annotations
import sys, time, json, re, random, hashlib
from pathlib import Path
import numpy as np, soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import soulx_freelyrics as fl
import singer

SONG, SEED, HOLD = "beat_it", 1, 0.4
N_CANDIDATES = 8
OUT = ROOT / "_tmp_beat_it_lyric"; OUT.mkdir(exist_ok=True)

# green, 1-syllable pools (clean onsets, simple vowels)
VERBS = ["feel", "see", "keep", "move", "take", "tap", "beat", "dig", "make"]
ADJS  = ["deep", "big", "cool", "calm", "neat", "fat", "good", "dark"]
# sustainable endings only (open vowel / nasal / liquid — never a final stop)
ENDS  = ["sea", "moon", "beam", "tune", "sun", "fun", "team", "mood", "food",
         "cool", "deal", "feel", "dawn", "noon", "dome", "dream"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def all_green(lines):
    for ph in fl.slots_by_phrase(SONG).values():
        pass
    for line in lines:
        for w in line.split():
            score, _ = singer.word_difficulty(w)
            if score > 1:
                return False
    return True


def make_candidate(rng):
    def line(n):
        body = [rng.choice(VERBS), "the"] + [rng.choice(ADJS) for _ in range(n - 3)]
        return " ".join(body + [rng.choice(ENDS)])
    return [line(4), line(4), line(5), line(5)]


def recall(voc, intended):
    import whisper
    if "m" not in _W:
        _W["m"] = whisper.load_model("large")
    txt = _W["m"].transcribe(str(voc), language="en")["text"].lower()
    hyp = set(re.findall(r"[a-z']+", txt))
    content = [w for w in intended if w != "the"]
    return sum(1 for w in content if w in hyp), len(content), txt.strip()


_W = {}


def main():
    rng = random.Random(0)
    cands, seen = [], set()
    while len(cands) < N_CANDIDATES and len(seen) < 400:
        c = make_candidate(rng); key = tuple(c)
        if key in seen:
            continue
        seen.add(key)
        ok, words = fl.check(c, SONG)
        if ok and all_green(c):
            cands.append((words, c))

    results = []
    for i, (words, lines) in enumerate(cands):
        sdir = OUT / f"cand{i}"; sdir.mkdir(exist_ok=True)
        voc = sdir / "generated.wav"
        intended = [w for line in lines for w in line.split()]
        if not voc.exists():
            tgt = sdir / "target.json"
            fl.build_target(words, tgt, song=SONG, hold_dur=HOLD)
            log(f"cand{i} {' / '.join(lines)} : render ...")
            singer.soulx_render(tgt.resolve(), sdir.resolve(), n_steps=32, seed=SEED, song=SONG)
        m, t, txt = recall(voc, intended)
        results.append((m, t, lines, txt, voc))
        log(f"cand{i}: recall {m}/{t}  | heard: {txt[:90]!r}")

    results.sort(key=lambda x: -x[0] / max(1, x[1]))
    log("==== ranked by content-word recall ====")
    for m, t, lines, txt, _ in results:
        log(f"  {m}/{t}  {' / '.join(lines)}")
    best = results[0]
    log(f"BEST: {best[0]}/{best[1]}  {' / '.join(best[2])}  -> {best[4]}")


if __name__ == "__main__":
    main()
