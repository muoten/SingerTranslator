---
title: AIchael Jackson
emoji: 🎤
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 6.3.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Put your lyrics into the chorus of MJ's songs
suggested_hardware: zero-a10g
---

# SingerTranslator

Custom lyrics on a fixed melody sung by a target voice — a control surface
most music AIs don't expose. This repo is the rendering pipeline behind the
**AIchael Jackson** demo at [aichaeljackson.com](https://aichaeljackson.com)
(HF Space: [muoten/aichael-jackson](https://huggingface.co/spaces/muoten/aichael-jackson)).

You pick a song (**Thriller** or **Billie Jean**) and write your own lyric — one
line per chorus phrase, each word matching the syllable count of the slot it
replaces. The pipeline:

1. Loads a frozen *score* (notes + timing) extracted from MJ's actual chorus
   via [Demucs](https://github.com/facebookresearch/demucs) +
   [Whisper](https://github.com/openai/whisper) +
   [ROSVOT](https://github.com/RickyL-2000/ROSVOT) preprocessing.
2. Maps your words onto the chorus's sung slots — one word per slot — preserving
   the notes, durations, and held-vowel melismas, so the melody, pitch and
   timing stay locked.
3. Asks [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) to
   synthesize the voice using a clip of MJ's actual chorus as the timbre
   prompt (so the resulting voice sounds like MJ).
4. Mixes the synthesized vocal with the original accompaniment.

Result: ~16 seconds of synthetic cover with *your* lyric on MJ's chorus.

## Why this exists

Most music AIs operate on raw audio and give you no symbolic handle on the
melody:

- **Suno / Udio / ACE-Step** — let you provide lyrics, but the model invents
  the melody and arrangement.
- **RVC / So-VITS / voice cloning** — let you swap timbre on existing audio,
  but lyrics and melody stay the originals.

Only two tools accept symbolic input (notes + lyrics) and synthesize singing
on top:

- **[ACE-Studio](https://acestudio.ai)** — commercial DAW-style editor;
  closed, no custom voice references.
- **[SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer)** (ACL'24) —
  open source, accepts any audio as a timbre reference. That's the unlock
  that makes voice cloning + lyric/melody control composable in a pipeline.
  This repo wraps SoulX as that pipeline.

## Demo (no install needed)

Visit [aichaeljackson.com](https://aichaeljackson.com) → pick a song → write your
lyric to match the per-line syllable guide → hit "Make AIchael sing it". The two
default lyrics are pre-cached for instant playback; a fresh lyric renders in ~15s
on the Space's GPU (~2 min on CPU). Identical (song + lyric + settings) requests
are cached, so repeats are free.

## Local install

```bash
git clone --recursive https://github.com/muoten/SingerTranslator.git
cd SingerTranslator
pip install -r requirements.txt
python bootstrap_soulx.py    # downloads SoulX weights + NLTK corpora (~5GB)
python app.py                # local Gradio at http://127.0.0.1:7860
```

CPU works (~2 min per render). For GPU pass `SINGER_DEVICE=cuda` (auto-detected
on HF Spaces).

## How it works internally

### Pipeline at runtime

```
[user picks a song + writes a lyric]
        |
        | soulx_freelyrics.build_target(words, song)
        |   - g2p per word (singer.syllable_to_phoneme)
        |   - map one word per sung slot (syllable-checked)
        |   - keep grid notes / durations / held-vowel melismas
        |   - optional onset-reinforce recipe (off by default)
        v
[target_metadata.json]
        |
        | singer.soulx_render()  →  SoulX CLI inference
        |   - prompt_wav: assets/<song>/prompt.wav (MJ chorus, timbre prompt)
        |   - prompt_metadata: assets/<song>/prompt.json
        |   - target_metadata: above
        |   - --n_steps, --seed 0, --pitch_shift 0
        v
[generated.wav]  (~16s sung vocal)
        |
        | ffmpeg mix with assets/<song>/accompaniment.wav
        v
[cover.wav]  (cached under assets/<song>/cache/)
```

### Score data (frozen)

Each song's chorus score in `assets/<song>/chorus_target.json` was built once
and ships with the repo. For Thriller it came from:

1. `vocals.wav` of Thriller, isolated by Demucs.
2. Slice 120-136s (the chorus instance we picked).
3. Whisper-large transcribed the words; ROSVOT transcribed the notes.
4. SoulX preprocess merged them into ~34 slots.
5. Manual surgical fix to split over-merged slots and demote spurious melismas.

Billie Jean's chorus score was built the same way (its multitracked chorus made
F0 noisier — see the build history). The timbre prompt `assets/<song>/prompt.wav`
is the same chorus offset by ~1.5s to avoid an audio-leakage failure mode in
SoulX where the
output mimics the prompt audio when prompt and target derive from the same
clip.

### Tunable parameters

The demo (`soulx_freelyrics_demo.py`) exposes:

- **song** — which chorus to sing on (`thriller` / `billie_jean`).
- **lyric** — one line per chorus phrase; each word's syllable count must match
  its slot (the UI shows a live syllable check).
- **n_steps** — CFM diffusion steps (default 32). Lower = faster, rougher.
- **reinforce weak onsets** — doubles weak HH/R/L onsets and clusters. Off by
  default (it tested net-negative on most words).

Renders use `seed=0`. To add a song, drop its four assets under
`assets/<song>/` and register a note-mapping `ORDER` in `soulx_freelyrics.py`.

### Phonetic learnings (English g2p approximating Spanish)

- **Plosives in short slots get swallowed** — double the leading consonant
  via the override (e.g. `bue` → `en_B-B-W-EH1`). Recipe is auto-applied to
  slot 1 only.
- **L coda on high notes drops** — `sole` becomes "soy". Fix: split the word
  across two slots so L moves to onset position (`soh / lee`).
- **`SYLLABLE_OVERRIDES`** in `singer.py` pins g2p_en's output for known
  Spanish syllables it would otherwise mispronounce (e.g. `nos` → `N-OW1-S`
  instead of g2p_en's `N-AA1-S`).

### Intelligibility metric

Whisper-prompted transcription is biased and can produce confident text on
audio that's actually noise. Honest signal comes from **unprompted Whisper +
phoneme-level Levenshtein** against the target. See the variant-sweep
discussion in `feedback_preferences.md` for the methodology.

## Repo layout

```
.
├── app.py                       — HF Space entry: bootstrap + launch the demo
├── soulx_freelyrics_demo.py     — Gradio UI (song picker, lyric box, render+cache)
├── soulx_freelyrics.py          — per-song engine: word→slot mapping, syllable check
├── singer.py                    — soulx_render(), g2p, mixing, helpers
├── bootstrap_soulx.py           — downloads weights + NLTK on first run
├── assets/
│   ├── thriller/  &  billie_jean/
│   │   ├── chorus_target.json       — frozen chorus score (notes + timing)
│   │   ├── prompt.wav / prompt.json — MJ chorus timbre prompt + metadata
│   │   ├── accompaniment.wav        — chorus instrumental (~16s)
│   │   └── cache/                   — rendered covers (incl. pre-baked defaults)
│   └── billie_jean/freelyric_reference.json  — per-phrase reference words
├── scripts/                     — eval / scoring tooling (not needed to run the app)
├── vendor/SoulX-Singer/         — submodule: muoten fork with --n_steps + --seed
└── examples/                    — milestone outputs from the build history
```

## Submodule patches

`vendor/SoulX-Singer` is pinned to the `patch-n-steps-cli` branch of
[muoten/SoulX-Singer](https://github.com/muoten/SoulX-Singer), which adds two
flags to `cli/inference.py`:

- `--n_steps` — overrides `config.infer.n_steps` for the CFM solver.
- `--seed` — calls `torch.manual_seed` (+ numpy + random + cuda) for
  reproducible renders.

Otherwise it tracks upstream `Soul-AILab/SoulX-Singer`.

## License

Code under MIT. The per-song `prompt.wav`, `accompaniment.wav`, and cached cover
wavs under `assets/<song>/` are derivatives of Michael Jackson recordings
("Thriller", "Billie Jean") and are included for demonstration purposes only.
