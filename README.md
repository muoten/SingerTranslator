---
title: AIchael Jackson
emoji: 🎤
colorFrom: pink
colorTo: indigo
sdk: gradio
python_version: "3.11"
app_file: app.py
pinned: false
short_description: AIchael sings 4 syllables on Thriller's chorus
suggested_hardware: zero-a10g
---

# SingerTranslator

Translate a *score* (lyrics + melody + voice prompt) into a *sung performance*.

Music generators invent the song. **SingerTranslator renders one you specify.**
You pick the lyrics, you pick the melody (MIDI/F0), you pick the voice; the
model produces the singing audio.

This is the user-controlled-composition workflow on top of
[SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) running locally.
It includes the helpers and recipes we found necessary to make English work.

## Why this exists

ACE-Step v1.5 cover-gen, Voicify (Demucs+RVC+ACE-Step), and SoulX SVC mode
all failed to deliver "custom lyrics on a custom melody in a chosen voice".
The first three either had no F0 channel or locked you into the target's
lyrics. SoulX **SVS** mode does have F0/MIDI input — but only when used
locally with hand-built metadata. That's what this repo wraps.

Validated 2026-05-05: produced clean English singing of the lyric "Who says
you're not broken" on a chosen melody with hard-K plosive. First time the
pipeline clicked end-to-end.

## Layout

```
.
├── README.md           — this
├── swap_word.py        — replace word X with word Y in metadata
│                         (auto regenerates phoneme via g2p_en)
│                         supports --phoneme override + --duration_boost
├── split_word.py       — replace one word slot with N consecutive slots
│                         (used to test mid-word splits; PROVED WORSE)
├── scripts/
│   └── sing.sh         — wrapper around SoulX-Singer SVS inference
└── examples/           — outputs we want to keep around
```

## Prerequisites

You need SoulX-Singer locally with weights downloaded. See
`project_english_singing_synthesis.md` in the auto-memory for the full install
path. Briefly:

```bash
cd ~/claude-code
git clone https://github.com/Soul-AILab/SoulX-Singer.git
cd SoulX-Singer
/Users/milhouse/.pyenv/versions/3.10.16/bin/python3.10 -m venv venv
venv/bin/pip install -r requirements.txt
mkdir pretrained_models && venv/bin/hf download Soul-AILab/SoulX-Singer \
    --local-dir pretrained_models/SoulX-Singer
venv/bin/python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')"
```

CPU-only on Mac (MPS broken in the vocoder, see memory notes). ~5x realtime.

## Workflow

```
[ source metadata.json ]
        |
        | swap_word.py / split_word.py     (edit lyrics, phonemes, durations)
        |
        v
[ edited metadata.json ]
        |
        | scripts/sing.sh                  (run SoulX SVS inference)
        |
        v
[ generated.wav ]                          a sung performance
```

## Quick start: word swap example

Take SoulX's shipped `en_target.json` (the "Who says you're not pretty" song),
swap "pretty" → "broken" with the hard-K recipe, and synthesize:

```bash
SOULX=~/claude-code/SoulX-Singer
PY=$SOULX/venv/bin/python

# 1. Edit the metadata (apply the triple-K plosive recipe + duration boost)
$PY swap_word.py \
    --in $SOULX/example/audio/en_target.json \
    --out /tmp/en_broken.json \
    --old pretty --new broken \
    --phoneme 'en_B-R-OW1-K-K-K-AH0-N' \
    --duration_boost 0.20

# 2. Synthesize
scripts/sing.sh \
    $SOULX/example/audio/en_prompt.mp3 \
    $SOULX/example/audio/en_prompt.json \
    /tmp/en_broken.json \
    /tmp/sung_broken

# 3. Listen
afplay /tmp/sung_broken/generated.wav
```

## Recipes

### English plosives are weak — use the triple-phone trick

The model has only 70 English phonemes vs ~2700 Chinese. English K/P/T
articulation is poor by default. Workaround: triple the plosive in the
phoneme override, with a moderate duration boost.

| Word | Phoneme override |
|---|---|
| broken | `en_B-R-OW1-K-K-K-AH0-N` |
| pretty | `en_P-P-P-R-IH1-T-T-T-IY0` (untested but follows the pattern) |
| broken (verified)  | tested 2026-05-05, produces hard K |

Validated trade-off (2026-05-05):
- Less than 3 K-phones: K is dropped or sounds soft
- More than 3 K-phones (4K, 5K): per-phone time falls below ~50ms, K
  collapses to a vowel transition
- Boost much beyond +0.20s: K → G voicing leak (closure gets filled with
  vocal-fold vibration from neighboring vowels — Chinese-prior unaspirated
  stops dominate)

The sweet spot is **3 K-phones at ~56ms each** in a slot ~0.45s long.

### Sonorants are fine

Words like "lovely", "morning", "shining" come out clean without any tricks.
Lyric-engineer toward sonorants when you can.

### Slot-splitting is worse

`split_word.py` tested mid-word splitting like "broken" → "brok" + "ken"
(2 slots with K at the slot boundary). The hypothesis was that `<EOW>`/`<BOW>`
markers would force harder articulation. **It didn't work** — each piece
got too little time, model isn't trained on mid-word slot splits. Kept the
script around for future experimentation but the single-slot triple-phone
recipe is what's been validated.

## Status (2026-05-05)

- ✅ Local install verified
- ✅ Lyric swap + phoneme override + duration boost working (`swap_word.py`)
- ✅ "broken benchmark" achieved with 3K + boost20 recipe
- ⏳ User's voice prompt (Shana clip) — needs prompt_metadata generation
  via SoulX preprocess pipeline (extra model downloads)
- ⏳ User's actual Thriller MIDI integration — currently using
  en_target's melody as the surrogate
- ⏳ Generalize plosive recipe to P, T (untested)

## What it isn't

- **Not a music generator** — doesn't invent songs from prompts
- **Not a karaoke maker** — doesn't separate voices from existing recordings
- **Not a voice cloner alone** — that's what RVC and SoulX SVC do; this
  controls more (lyrics + melody + voice, not just voice)

It's the rendering step of a composition pipeline. You bring the score,
SingerTranslator brings the singer.
