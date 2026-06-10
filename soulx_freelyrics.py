"""
soulx_freelyrics.py -- render FREE lyrics on the Thriller chorus melody with SoulX.

Past the old 4-syllable cyclic demo: write a full 4-line lyric whose per-slot
syllable counts match the Thriller template; a checker validates the fit; the words
are then mapped 1:1 onto the grid notes and SoulX renders them with the melody +
timing LOCKED (robotic but controlled -- pair with Vevo2/VC for a natural voice).

  python3 soulx_freelyrics.py                 # demo: the 'summer' parody
  python3 soulx_freelyrics.py --lyric my.txt  # 4 lines, words = 5 / 11 / 6 / 9

The template below is the de-inflated, off-by-one-corrected Thriller grid we built
this session: each slot = (original_word, note_indices, phrase). RESTS are kept.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import singer  # noqa: E402

# ordered note structure: ("w", reference_word, [note indices], phrase) or ("R", note_idx)
# reference_word only fixes the per-slot syllable count shown in the template.
THRILLER_ORDER = [
    ("w", "is", [0, 1, 2], 1), ("w", "thriller", [3], 1),
    ("R", 4), ("w", "thriller", [5, 6], 1), ("w", "night", [7], 1), ("R", 8),
    ("w", "and", [9], 2), ("w", "no", [10], 2), ("w", "one's", [11], 2),
    ("w", "gonna", [12], 2), ("w", "save", [13], 2), ("w", "you", [14], 2),
    ("w", "from", [15], 2), ("w", "the", [16], 2), ("w", "beast", [17], 2),
    ("w", "about", [18], 2), ("w", "strike", [19], 2),
    ("w", "you", [20], 3), ("w", "know", [21], 3), ("w", "it's", [22], 3),
    ("w", "thriller", [23], 3), ("w", "thriller", [24, 25], 3), ("w", "night", [26], 3),
    ("w", "you're", [27], 4), ("w", "fighting", [28], 4), ("w", "for", [29], 4),
    ("w", "your", [30], 4), ("w", "life", [31], 4), ("w", "inside", [32], 4),
    ("w", "a", [33], 4), ("w", "killer", [34], 4), ("w", "thriller", [35], 4),
]

# Billie Jean: 7 / 11 / 7 slots matching the real chorus. We specify only the note
# STRUCTURE here (indices / fraction-splits — no lyrics); the per-slot reference
# WORDS are pulled at import from the mapping in _bj_truelyrics.py (the repo file we
# built), never re-typed. note-spec = [note indices] or ("frac", note, start, length).
# P1: uh Billie Jean is | not/my/lover share the long note-5 sustain via fractions.
_BJ_P1_SPEC = [[0, 1], [2], [3], [4],
               ("frac", 5, 0.117, 0.164), ("frac", 5, 0.281, 0.328), ("frac", 5, 0.609, 0.391)]
_BJ_P2_SPEC = [[7], [8], [9], [10, 11], [12], [13, 14], [15], [16], [17], [18], [19, 20]]
_BJ_P3_SPEC = [[21, 22], [23], [24], [25], [26], [27, 28], [29]]


_BJ_REF_JSON = ROOT / "assets/billie_jean/freelyric_reference.json"


def _bj_ref_words():
    """Per-phrase reference words for the BJ template. Loaded from the committed
    data file (deploy path); falls back to deriving them from the _bj_truelyrics
    mapping for local dev / regenerating the data file."""
    if _BJ_REF_JSON.exists():
        return json.loads(_BJ_REF_JSON.read_text())["phrases"]
    import _bj_truelyrics as bt  # dev fallback only — not needed at deploy time
    p1, p2, p3 = [], [], []
    for it in bt.ORDER_ALIGN:
        if it[0] == "w":
            n0 = it[2][0]
            (p1 if n0 < 6 else p2 if n0 <= 20 else p3).append(it[1])
        elif it[0] == "split":
            onsets = [p[0] for p in it[2]
                      if p[0] != "<SP>" and (len(p) < 5 or p[4] != "3")]
            if it[1] == 4:
                p1 += onsets
            elif it[1] == 5:
                p1 += onsets[:2] + ["".join(onsets[2:])]
    return [p1, p2, p3]


def _build_bj_order():
    p1w, p2w, p3w = _bj_ref_words()
    o = [("w", w, spec, 1) for spec, w in zip(_BJ_P1_SPEC[:4], p1w[:4])]   # uh Billie Jean is
    o.append(("R", ("frac", 5, 0.0, 0.117)))                              # pickup gap in note 5
    o += [("w", w, spec, 1) for spec, w in zip(_BJ_P1_SPEC[4:], p1w[4:])] # the note-5 trio
    o.append(("R", 6))
    o += [("w", w, spec, 2) for spec, w in zip(_BJ_P2_SPEC, p2w)]
    o += [("w", w, spec, 3) for spec, w in zip(_BJ_P3_SPEC, p3w)]
    o += [("R", 30), ("R", 31)]
    return o


BILLIE_JEAN_ORDER = _build_bj_order()

ORDERS = {"thriller": THRILLER_ORDER, "billie_jean": BILLIE_JEAN_ORDER}

DEMOS = {
    "thriller": [
        "is summer summer day",
        "and no one gonna take you from the heat about rise",
        "you know it's summer summer day",
        "you're living for your dreams inside a sunny summer",
    ],
    # Default = the real working chorus, pulled from _bj_truelyrics (not re-typed).
    "billie_jean": [" ".join(ws) for ws in _bj_ref_words()],
}


def grid_path(song="thriller"):
    return singer.template_json(song)


# Back-compat module defaults (used by main()'s Thriller path).
GRID = grid_path("thriller")
ORDER = THRILLER_ORDER
DEMO = DEMOS["thriller"]

_VOW = ("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW")


def syl(word):
    """Syllable count = number of vowel phonemes in the SoulX G2P of the word."""
    ph = singer.syllable_to_phoneme(word).replace("en_", "")
    n = sum(1 for t in ph.replace("-", " ").split() if any(t.startswith(v) for v in _VOW))
    return max(n, 1)


def slots_by_phrase(song="thriller"):
    out = {}
    for item in ORDERS[song]:
        if item[0] == "w":
            out.setdefault(item[3], []).append(item)
    return dict(sorted(out.items()))


def _dot(score):
    """Difficulty bucket: 0-1 easy, 2-3 risky, 4+ hard."""
    return "🟢" if score <= 1 else ("🟡" if score <= 3 else "🔴")


def word_counts(song="thriller"):
    """Per-phrase word count, e.g. [4, 11, 6, 9]."""
    return [len(v) for v in slots_by_phrase(song).values()]


_SUP = {2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶"}


def template_md(song="thriller"):
    """Concise, mobile-friendly guide: just the example slots, with multi-syllable
    words flagged by a superscript. The example doubles as the per-slot syllable map.
    Reference words come from the ORDER (Thriller original / BJ from _bj_truelyrics)."""
    blocks = []
    for p, slots in slots_by_phrase(song).items():
        marked = []
        for s in slots:
            w, n = s[1], syl(s[1])
            marked.append(w if n == 1 else f"{w}{_SUP.get(n, f'({n})')}")
        blocks.append(f"**{p}.** " + " · ".join(marked))
    return ("Swap each word for your own — keep the syllable count. A small ² marks a "
            "slot that needs a **2-syllable** word:\n\n" + "\n\n".join(blocks))


def check(lines, song="thriller"):
    """Validate the lyric against the template; return (ok, words_per_phrase).

    Prints, per word: syllable match AND an articulation-difficulty estimate
    (singer.word_difficulty) so you can edit hard words (🔴/🟡) to greener ones
    BEFORE rendering. Aim for all 🟢 — that's the cheapest path to clean articulation.
    """
    sp = slots_by_phrase(song)
    ok = True
    words = {}
    tally = {"🟢": 0, "🟡": 0, "🔴": 0}
    print(f"=== syllable + articulation-difficulty check (free lyric vs {song} template) ===")
    for p in sp:
        tw = lines[p - 1].split() if p - 1 < len(lines) else []
        words[p] = tw
        need = len(sp[p])
        tag = "OK" if len(tw) == need else f"WRONG WORD COUNT ({len(tw)} vs {need})"
        nsyl = sum(syl(x[1]) for x in sp[p])
        print(f"P{p}: needs {nsyl} syllables / {need} words  [{tag}]")
        if len(tw) != need:
            ok = False
            continue
        for (_, ow, _, _), w in zip(sp[p], tw):
            want, got = syl(ow), syl(w)
            mark = "  ok" if want == got else "  XX"
            if want != got:
                ok = False
            score, reasons = singer.word_difficulty(w)
            dot = _dot(score)
            tally[dot] += 1
            rstr = f"   [{', '.join(reasons)}]" if reasons else ""
            print(f"   {mark} {ow}({want}) -> {w}({got})  {dot}{rstr}")
    n = sum(tally.values())
    hard = tally["🔴"] + tally["🟡"]
    print(f"\ndifficulty: 🟢 {tally['🟢']}  🟡 {tally['🟡']}  🔴 {tally['🔴']}"
          + (f"   ({hard} word(s) likely to garble — edit toward 🟢)" if hard else "   (all green — good to render)"))
    return ok, words


def build_target(words, out_path, song="thriller", recipes=False, reinforce=False):
    """Swap each slot's phoneme for the free word; keep grid timing/pitch/f0.

    recipes=True applies the two GENERAL articulation rules from build_target_metadata
    (no per-word hacks): double leading plosives, and on multi-note (held) slots put
    the consonants on the first note + a held VOWEL on the continuation notes.

    reinforce=True (without recipes) applies ONLY the onset-reinforcement recipe
    (singer.reinforce_onset: double weak HH/R/L onsets, cluster + post-schwa plosives)
    on the whole word, with NO held-vowel split — the clean, isolated articulation fix.
    """
    u = json.loads(grid_path(song).read_text())[0]
    order = ORDERS[song]
    dur = [float(x) for x in u["duration"].split()]
    pit = [int(x) for x in u["note_pitch"].split()]
    f0 = [float(x) for x in u["f0"].split()]
    onsets = np.concatenate([[0], np.cumsum(dur)])[:-1]
    fps = len(f0) / sum(dur)

    def pitch_of(g):
        cand = [(dur[i], pit[i]) for i in g if pit[i] > 0]
        if cand:
            return max(cand)[1]
        a, b = int(onsets[g[0]] * fps), int((onsets[g[-1]] + dur[g[-1]]) * fps)
        seg = [x for x in f0[a:b] if x > 0]
        return int(round(69 + 12 * np.log2(np.median(seg) / 440.0))) if seg else 0

    def is_frac(spec):
        return isinstance(spec, tuple) and spec and spec[0] == "frac"

    cursor = {p: 0 for p in slots_by_phrase(song)}
    phon, dura, npit, ntyp, text = [], [], [], [], []
    for item in order:
        if item[0] == "R":
            spec = item[1]
            d = dur[spec[1]] * spec[3] if is_frac(spec) else dur[spec]
            phon.append("<SP>"); dura.append(round(d, 4)); npit.append(0); ntyp.append(1)
            continue
        _, _, g, phrase = item
        w = words[phrase][cursor[phrase]]; cursor[phrase] += 1
        base = singer.syllable_to_phoneme(w)
        if is_frac(g):
            # one word on a time-fraction of a shared note (no melisma split)
            n = g[1]
            ptch = pit[n] if pit[n] > 0 else pitch_of([n])
            phon.append(singer.reinforce_onset(base) if reinforce else base)
            dura.append(round(dur[n] * g[3], 4)); npit.append(ptch); ntyp.append(2); text.append(w)
        elif recipes:
            # reinforce the onset consonant (weak HH/R/L + plosives) on note 1, held vowel after
            phon.append(singer.reinforce_onset(base))
            dura.append(round(dur[g[0]], 4)); npit.append(pitch_of([g[0]])); ntyp.append(2); text.append(w)
            held = singer.held_form(base)
            for k in g[1:]:
                phon.append(held); dura.append(round(dur[k], 4)); npit.append(pitch_of([k])); ntyp.append(3); text.append(w + "_")
        else:
            phon.append(singer.reinforce_onset(base) if reinforce else base)
            dura.append(round(sum(dur[i] for i in g), 4)); npit.append(pitch_of(g)); ntyp.append(2); text.append(w)

    meta = dict(u)
    meta["phoneme"] = " ".join(phon)
    meta["duration"] = " ".join(f"{x:.4f}" for x in dura)
    meta["note_pitch"] = " ".join(str(x) for x in npit)
    meta["note_type"] = " ".join(str(x) for x in ntyp)
    meta["text"] = " ".join(text)
    out_path.write_text(json.dumps([meta], indent=2))
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lyric", help="text file with 4 lines (else the summer demo)")
    ap.add_argument("--out_dir", default="_tmp_freelyrics")
    ap.add_argument("--name", default="summer")
    ap.add_argument("--recipes", action="store_true",
                    help="full recipe: onset reinforce + held-vowel split on multi-note slots")
    ap.add_argument("--reinforce", action="store_true",
                    help="onset reinforcement ONLY (no held split) — the clean isolated fix")
    ap.add_argument("--n_steps", type=int, default=32, help="CFM diffusion steps (default 32)")
    args = ap.parse_args()
    lines = (Path(args.lyric).read_text().splitlines() if args.lyric else DEMO)
    lines = [l.strip() for l in lines if l.strip()]

    ok, words = check(lines)
    if not ok:
        print("\n[abort] lyric does not fit the template -- adjust the flagged words.")
        sys.exit(1)
    print("\n[ok] lyric fits -> building SoulX target + rendering ...")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(exist_ok=True)
    tgt = build_target(words, out_dir / f"{args.name}_target.json",
                       recipes=args.recipes, reinforce=args.reinforce)
    vocal = singer.soulx_render(tgt.resolve(), out_dir.resolve(), n_steps=args.n_steps, seed=0)
    vocal_named = out_dir / f"{args.name}_vocal.wav"
    vocal_named.write_bytes(Path(vocal).read_bytes())
    mix = singer.mix_with_accompaniment(vocal_named, out_dir / f"{args.name}_mix.wav", song="thriller")
    print(f"\nDONE\n  vocal: {vocal_named}\n  mix:   {mix}")


if __name__ == "__main__":
    main()
