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

You pick one of the demo songs — **Thriller**, **Billie Jean**, **Beat It**,
**Bad**, or **The Way You Make Me Feel** — and write your own lyric, one line per
chorus phrase, each word matching the syllable count of the slot it replaces.
The pipeline:

1. Loads a frozen *score* (notes + timing) extracted from MJ's actual chorus
   via [Demucs](https://github.com/facebookresearch/demucs) + a mel-band-roformer
   karaoke step (to isolate MJ's *lead* from any backing choir) +
   [Whisper](https://github.com/openai/whisper) +
   [ROSVOT](https://github.com/RickyL-2000/ROSVOT) preprocessing.
2. Maps your words onto the chorus's sung slots — one word per slot — preserving
   the notes, durations, and held-vowel melismas, so the melody, pitch and
   timing stay locked.
3. Asks [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) to
   synthesize the voice using a clip of MJ's voice as the timbre prompt — a
   single shared MJ verse clip reused across all songs (see *Score data* below).
4. Mixes the synthesized vocal with the original accompaniment.

More songs are built and shipped but kept out of the public demo when a render
isn't clean enough (automatic quality gates, below); the demo only lists the
songs that pass.

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
lyric to match the per-line syllable guide → hit "Make AIchael sing it". Each
song's default lyric is pre-cached for instant playback; a fresh lyric renders in
~15s on the Space's GPU (~2 min on CPU). Identical (song + lyric + settings) requests
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
        |   - prompt_wav: assets/<song>/prompt.wav if present, else the shared
        |       assets/_shared/mj_prompt.wav (MJ verse clip, timbre prompt)
        |   - prompt_metadata: matching prompt.json
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

Each song's chorus score in `assets/<song>/chorus_target.json` is built once and
ships with the repo. Songs are added through one standardized, resumable pipeline,
`build_song.py`, whose stages are:

```
fetch     yt-dlp by title  → sources/<song>/<song>_full.wav  (+ validates length)
separate  Demucs → vocals / accompaniment
slice     cut the chosen chorus WINDOW (an ear-judgment)
preproc   mel-band-roformer KARAOKE → isolate MJ lead (drop backing choir),
          then Whisper align + f0 + ROSVOT notes  → grid
grid      preproc output → chorus_target.json
prompt    resolve the timbre prompt (shared MJ clip; per-song override optional)
accomp    vocal-stripped chorus instrumental
verify    neutral-'la' render — copyright-safe leakage check (an ear-judgment)
order     propose note→word ORDER by de-inflation (an ear-judgment)
register  print the ORDERS/DEMOS snippet for soulx_freelyrics.py
```

The karaoke step matters: some choruses are sung by MJ *in unison with a choir*,
which no separator can cleanly split — the isolated lead comes out smeared and the
render is poor. That case is flagged automatically (see *Quality gates*).

**Timbre prompt.** Rather than a per-song clip, all songs share one canonical MJ
voice prompt, `assets/_shared/mj_prompt.wav` — a clean Thriller *verse* clip.
Because it's verse audio it differs from every song's *chorus* target, which
sidesteps a SoulX audio-leakage failure mode (the output mimics the prompt when
prompt and target derive from the same clip). `singer.prompt_wav()` falls back to
the shared prompt unless a song ships its own `assets/<song>/prompt.wav` override.

### Tunable parameters

The demo (`soulx_freelyrics_demo.py`) exposes:

- **song** — which chorus to sing on (any demo song: `thriller`, `billie_jean`,
  `beat_it`, `bad`, `the_way_you_make_me_feel`).
- **lyric** — one line per chorus phrase; each word's syllable count must match
  its slot (the UI shows a live syllable check).
- **n_steps** — CFM diffusion steps (default 32). Lower = faster, rougher.
- **reinforce weak onsets** — doubles weak HH/R/L onsets and clusters. Off by
  default (it tested net-negative on most words).

Renders use `seed=0`. To add a song, run `build_song.py` end to end (it stops at
the few ear-judgment stages — window, verify, order), register the printed
`ORDERS`/`DEMOS` snippet in `soulx_freelyrics.py`, then `bake_song.py` renders a
seed-best cover and the quality gates decide demo-eligibility. A song appears in
the demo only when it is registered **and** `assets/<song>/config.json` has
`"demo": true` (new songs default to hidden).

### Phonetic learnings (English g2p approximating Spanish)

- **Plosives in short slots get swallowed** — double the leading consonant
  via the override (e.g. `bue` → `en_B-B-W-EH1`). Recipe is auto-applied to
  slot 1 only.
- **L coda on high notes drops** — `sole` becomes "soy". Fix: split the word
  across two slots so L moves to onset position (`soh / lee`).
- **`SYLLABLE_OVERRIDES`** in `singer.py` pins g2p_en's output for known
  Spanish syllables it would otherwise mispronounce (e.g. `nos` → `N-OW1-S`
  instead of g2p_en's `N-AA1-S`).

### Quality gates (demo-eligibility)

Whether a baked cover is good enough for the public demo is decided by two
automatic, complementary checks — a render is demo-eligible only if **both** pass:

- **scat gate** (`scripts/crappy_fragments.py`) — measures the total duration of
  *grid-mismatched* audio: sound during a rest (PHANTOM), wrong pitch on a note
  (OFFPITCH), or silence on a note (DEAD). `total_crappy < 5.0s` = pass. This
  catches a render that fails to track an otherwise-good grid.
- **solo-vs-chorus gate** (`scripts/validate_lead.py`) — measures how much
  *other-vocal* energy surrounded the isolated lead (from the karaoke step's
  removed-backing output). A high ratio means the chorus is choral (lead in
  unison with a choir) → the grid itself is corrupt → the scat gate can't see it.

The bake (`scripts/bake_song.py`) runs both, auto-sets `demo: false` on failure,
and only *recommends* `demo: true` (publishing is a human confirm).
Phoneme-recognition metrics (per-slot, recall/precision/F1, Whisper, timbre-sim)
were all tried and **rejected** — they don't track perceived quality.

## Repo layout

```
.
├── app.py                       — HF Space entry: bootstrap + launch the demo
├── soulx_freelyrics_demo.py     — Gradio UI (song picker, lyric box, render+cache)
├── soulx_freelyrics.py          — per-song engine: word→slot mapping (ORDERS/DEMOS)
├── singer.py                    — soulx_render(), g2p, mixing, prompt fallback, config
├── build_song.py                — add a song: fetch→…→register (resumable stages)
├── bootstrap_soulx.py           — downloads weights + NLTK on first run
├── assets/
│   ├── _shared/mj_prompt.{wav,json}   — canonical MJ timbre prompt (shared by all)
│   └── <song>/                        — thriller, billie_jean, beat_it, bad, …
│       ├── chorus_target.json         — frozen chorus score (notes + timing)
│       ├── accompaniment.wav          — chorus instrumental (~16s)
│       ├── config.json                — per-song knobs incl. the "demo" flag
│       ├── prompt.{wav,json}          — OPTIONAL per-song timbre-prompt override
│       └── cache/                     — rendered covers (incl. pre-baked defaults)
├── scripts/                     — build/bake/gate + eval tooling (not needed to run app)
│   ├── bake_song.py                   — seed-best render + auto quality gates
│   ├── crappy_fragments.py            — scat gate (grid-mismatch duration)
│   ├── validate_lead.py               — solo-vs-chorus gate
│   └── verify_neutral.py, validate_grid.py, …  — other checks & eval
└── vendor/SoulX-Singer/         — submodule: muoten fork with --n_steps + --seed
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

Code under MIT. The audio assets — the shared `assets/_shared/mj_prompt.wav`, and
each song's `accompaniment.wav`, any `prompt.wav`, and cached cover wavs under
`assets/<song>/` — are derivatives of Michael Jackson recordings (Thriller,
Billie Jean, Beat It, Bad, The Way You Make Me Feel, and others) and are included
for demonstration purposes only.
