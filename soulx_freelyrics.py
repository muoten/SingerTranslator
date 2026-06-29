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

# Beat It: chorus grid from 114.1–130.1s (assets/beat_it/chorus_target.json). Window
# starts cleanly on the hook. Vocals isolated with SoulX's mel-band-roformer karaoke +
# dereverb, Whisper aligned with the true text as initial_prompt for per-slot timing.
#
# RELIABLE mapping — one clean syllable per note that can actually voice it. Two synth
# limits force this: (a) a single note can't articulate a 2-syllable word (the 2nd
# syllable crams/drops: power->"pow", stronger->"strong"), so every slot is 1-syllable;
# (b) notes under ~0.3s drop the syllable entirely (too fast to articulate), so they're
# rested. Also rested: the silent reconstructed slots (unvoiced f0 -> no sound), the
# >1s held note's overflow, and the next-repeat tail. What remains all voices cleanly:
# 4 / 4 / 5 / 5 = 18 syllables. (The original packs ~22-31; the synth can't articulate
# that density, so reliable-clarity caps lower.) Measured grid: chorus_target.roformer_raw.json.
BEAT_IT_ORDER = [
    ("R", 0),
    ("w", "la", [1], 1), ("w", "la", [2], 1), ("w", "la", [3], 1), ("w", "la", [4], 1),
    ("R", 5),
    ("w", "la", [6], 2), ("w", "la", [7], 2), ("w", "la", [8], 2), ("w", "la", [9], 2),
    ("R", 10), ("R", 11), ("R", 12), ("R", 13), ("R", 14),
    ("w", "la", [15], 3), ("w", "la", [16], 3), ("w", "la", [17], 3),
    ("R", 18),
    ("w", "la", [19, 20], 3),
    ("R", 21), ("R", 22),
    ("w", "la", [23], 3),
    ("R", 24),
    ("w", "la", [25], 4), ("w", "la", [26], 4), ("w", "la", [27], 4),
    ("R", 28),
    ("w", "la", [29], 4), ("w", "la", [30], 4),
    ("R", 31), ("R", 32),
]

# Bad: stitched full-chorus grid (roformer-isolated head + demucs tail — roformer strips
# the layered final phrase, so the climactic "who's bad?" tail is stitched from demucs).
# Parody default: the "I'm bad" hook -> "I'm back", "who's bad?" -> "who's back?" on the
# closing beat. Slots group each word with its trailing melisma notes; 49-slot grid.
BAD_ORDER = [
    ("R", 0),
    ("w", "I'm", [1], 1),
    ("w", "back", [2, 3, 4], 1),
    ("w", "I'm", [5], 1),
    ("w", "back", [6, 7], 1),
    ("w", "no", [8], 1),
    ("w", "more", [9, 10, 11], 1),
    ("w", "I'm", [12], 2),
    ("w", "back", [13], 2),
    ("w", "I'm", [14], 2),
    ("w", "back", [15], 2),
    ("w", "you", [16, 17], 2),
    ("w", "know", [18], 2),
    ("R", 19),
    ("R", 20),
    ("w", "no", [21, 22], 3),
    ("w", "more", [23, 24], 3),
    ("w", "I'm", [25], 3),
    ("w", "back", [26], 3),
    ("w", "I'm", [27], 3),
    ("w", "back", [28, 29], 3),
    ("w", "you", [30, 31], 3),
    ("w", "know", [32], 3),
    ("R", 33),
    ("R", 34),
    ("R", 35),
    ("w", "the", [36], 4),
    ("R", 37),
    ("w", "whole", [38], 4),
    ("w", "world", [39], 4),
    ("w", "knows", [40], 4),
    ("w", "now", [41], 4),
    ("w", "I'll", [42], 4),
    ("w", "tell", [43], 4),
    ("w", "you", [44], 4),
    ("w", "once", [45], 4),
    ("w", "more", [46], 4),
    ("w", "who's", [47], 4),
    ("w", "back", [48], 4),
]

# Smooth Criminal — auto-proposed de-inflation ORDER (11 audible slots), with the
# phrase numbers renumbered contiguous (1..4) so DEMOS is a clean 4-line lyric.
# Reference words are 'la' (1 syllable) -> any 1-syllable word fits. NOT ear-refined:
# the chorus grid is sparse and quality is "not good" -> kept demo=false (hidden).
SMOOTH_CRIMINAL_ORDER = [
    ("R", 0), ("R", 1), ("R", 2), ("w", "la", [3], 1),
    ("R", 4), ("R", 5), ("R", 6), ("w", "la", [7], 2),
    ("R", 8), ("R", 9), ("w", "la", [10], 2), ("R", 11),
    ("w", "la", [12], 2), ("w", "la", [13], 2), ("R", 14), ("w", "la", [15], 2),
    ("R", 16), ("R", 17), ("R", 18), ("R", 19), ("w", "la", [20], 3),
    ("R", 21), ("R", 22), ("w", "la", [23], 3), ("w", "la", [24], 3),
    ("R", 25), ("R", 26), ("R", 27), ("R", 28), ("R", 29), ("R", 30), ("R", 31), ("R", 32),
    ("w", "la", [33], 4), ("R", 34), ("R", 35), ("w", "la", [36], 4), ("R", 37),
]

# The Way You Make Me Feel — auto de-inflation ORDER, 20 audible slots across 3
# contiguous phrases (3/7/10). Reference words 'la' (1 syllable). Not ear-refined.
THE_WAY_YOU_MAKE_ME_FEEL_ORDER = [
    ("w", "la", [0], 1), ("R", 1), ("w", "la", [2], 1), ("w", "la", [3], 1),
    ("R", 4), ("R", 5), ("R", 6), ("w", "la", [7], 2), ("R", 8), ("R", 9),
    ("w", "la", [10], 2), ("w", "la", [11], 2), ("w", "la", [12], 2), ("R", 13), ("R", 14),
    ("w", "la", [15], 2), ("R", 16), ("w", "la", [17], 2), ("R", 18), ("w", "la", [19], 2),
    ("R", 20), ("R", 21), ("R", 22), ("R", 23),
    ("w", "la", [24], 3), ("w", "la", [25], 3), ("w", "la", [26], 3), ("R", 27),
    ("w", "la", [28], 3), ("w", "la", [29], 3), ("w", "la", [30], 3), ("w", "la", [31], 3),
    ("R", 32), ("R", 33), ("w", "la", [34], 3), ("w", "la", [35], 3), ("w", "la", [36], 3),
    ("R", 37), ("R", 38),
]

ORDERS = {
    "thriller": THRILLER_ORDER,
    "billie_jean": BILLIE_JEAN_ORDER,
    "beat_it": BEAT_IT_ORDER,
    "bad": BAD_ORDER,
    "smooth_criminal": SMOOTH_CRIMINAL_ORDER,
    "the_way_you_make_me_feel": THE_WAY_YOU_MAKE_ME_FEEL_ORDER,
}

DEMOS = {
    "thriller": [
        "is summer summer day",
        "and no one gonna take you from the heat about rise",
        "you know it's summer summer day",
        "you're living for your dreams inside a sunny summer",
    ],
    # Default = the real working chorus, pulled from _bj_truelyrics (not re-typed).
    "billie_jean": [" ".join(ws) for ws in _bj_ref_words()],
    # Original 4-line demo on the Beat It chorus (4 / 4 / 5 / 5, one syllable per note).
    # Chosen for clean articulation: ALL 18 words score green in the checker (strong
    # consonant onsets, no vowel-leads/glides/weak-onsets), and pure-vowel line endings
    # (beat/moon/deep/back) so the held end-notes don't double a diphthong glide.
    "beat_it": [
        "stand up and run",
        "the work is done",
        "lift your hands have fun",
        "now we all are one",
    ],
    # Bad: "I'm back" comeback parody, ending on "who's back?" (mirrors "who's bad?").
    "bad": [
        "I'm back I'm back no more",
        "I'm back I'm back you know",
        "no more I'm back I'm back you know",
        "the whole world knows now I'll tell you once more who's back",
    ],
    # Placeholder monosyllabic lyric matching the 1/5/3/2 slot counts. Hidden song
    # (demo=false); replace + ear-refine the ORDER before ever promoting it.
    "smooth_criminal": [
        "now",
        "take me to the top",
        "day and night",
        "we go",
    ],
    # Placeholder monosyllabic lyric matching the 3/7/10 slot counts. Swap for a
    # real lyric + ear-refine before promoting (demo flag in config.json).
    "the_way_you_make_me_feel": [
        "day by day",
        "take me to the top with you",
        "we dance and dance through the dark all night long",
    ],
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


def build_target(words, out_path, song="thriller", recipes=False, reinforce=False,
                 hold_dur=None):
    """Swap each slot's phoneme for the free word; keep grid timing/pitch/f0.

    recipes=True applies the two GENERAL articulation rules from build_target_metadata
    (no per-word hacks): double leading plosives, and on multi-note (held) slots put
    the consonants on the first note + a held VOWEL on the continuation notes.

    reinforce=True (without recipes) applies ONLY the onset-reinforcement recipe
    (singer.reinforce_onset: double weak HH/R/L onsets, cluster + post-schwa plosives)
    on the whole word, with NO held-vowel split — the clean, isolated articulation fix.

    hold_dur=None (default) pulls the per-song cap from assets/<song>/config.json;
    pass an explicit number to override, or 0 to force it off.
    """
    if hold_dur is None:
        hold_dur = singer.song_config(song).get("hold_dur")
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
            text.append("<SP>")  # keep text aligned with the other arrays (rests included)
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
            d = round(sum(dur[i] for i in g), 4); ptc = pitch_of(g)
            ph = singer.reinforce_onset(base) if reinforce else base
            if hold_dur and d > hold_dur:
                # Long note: articulate the syllable on a short onset, then HOLD the
                # vowel (note_type 3) for the remainder — stops SoulX from re-firing
                # the whole syllable on the held note ("beat beat beat").
                on_d = min(0.30, round(d * 0.45, 4))
                phon.append(ph); dura.append(on_d); npit.append(ptc); ntyp.append(2); text.append(w)
                phon.append(singer.held_form(base)); dura.append(round(d - on_d, 4))
                npit.append(ptc); ntyp.append(3); text.append(w + "_")
            else:
                phon.append(ph); dura.append(d); npit.append(ptc); ntyp.append(2); text.append(w)

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
    ap.add_argument("--song", default="thriller", choices=list(ORDERS),
                    help="which chorus to sing on (default thriller)")
    args = ap.parse_args()
    song = args.song
    lines = (Path(args.lyric).read_text().splitlines() if args.lyric else DEMOS[song])
    lines = [l.strip() for l in lines if l.strip()]

    ok, words = check(lines, song)
    if not ok:
        print("\n[abort] lyric does not fit the template -- adjust the flagged words.")
        sys.exit(1)
    print("\n[ok] lyric fits -> building SoulX target + rendering ...")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(exist_ok=True)
    tgt = build_target(words, out_dir / f"{args.name}_target.json", song=song,
                       recipes=args.recipes, reinforce=args.reinforce)
    vocal = singer.soulx_render(tgt.resolve(), out_dir.resolve(), n_steps=args.n_steps, seed=0, song=song)
    vocal_named = out_dir / f"{args.name}_vocal.wav"
    vocal_named.write_bytes(Path(vocal).read_bytes())
    mix = singer.mix_with_accompaniment(vocal_named, out_dir / f"{args.name}_mix.wav", song=song)
    print(f"\nDONE\n  vocal: {vocal_named}\n  mix:   {mix}")


if __name__ == "__main__":
    main()
