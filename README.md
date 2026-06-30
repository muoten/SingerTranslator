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
   synthesize the voice using a clip of MJ's voice as the timbre prompt.
4. Mixes the synthesized vocal with the original accompaniment.

Result: ~16 seconds of synthetic cover with *your* lyric on MJ's chorus.

> **Building the songs** (adding new ones, the score pipeline, quality gates,
> repo internals)? That's all in **[BACKSTAGE.md](BACKSTAGE.md)**. This page is
> about *using* the demo.

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

## Try it (no install)

Visit [aichaeljackson.com](https://aichaeljackson.com) → pick a song → write your
lyric to match the per-line syllable guide → hit "Make AIchael sing it". Each
song's default lyric is pre-cached for instant playback; a fresh lyric renders in
~15s on the Space's GPU (~2 min on CPU). Identical (song + lyric + settings)
requests are cached, so repeats are free.

## Run it locally

```bash
git clone --recursive https://github.com/muoten/SingerTranslator.git
cd SingerTranslator
pip install -r requirements.txt
python bootstrap_soulx.py    # downloads SoulX weights + NLTK corpora (~5GB)
python app.py                # local Gradio at http://127.0.0.1:7860
```

CPU works (~2 min per render). For GPU pass `SINGER_DEVICE=cuda` (auto-detected
on HF Spaces).

## Inputs

The demo (`soulx_freelyrics_demo.py`) exposes:

- **song** — which chorus to sing on (any demo song: `thriller`, `billie_jean`,
  `beat_it`, `bad`, `the_way_you_make_me_feel`).
- **lyric** — one line per chorus phrase; each word's syllable count must match
  its slot (the UI shows a live syllable check).
- **n_steps** — CFM diffusion steps (default 32). Lower = faster, rougher.
- **reinforce weak onsets** — doubles weak HH/R/L onsets and clusters. Off by
  default (it tested net-negative on most words).

Renders use `seed=0`, so the same inputs always give the same audio.

## How a render works

```
[you pick a song + write a lyric]
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

The frozen score and timbre prompt this consumes are built offline — see
[BACKSTAGE.md](BACKSTAGE.md).

## License

Code under MIT. The audio assets — the shared `assets/_shared/mj_prompt.wav`, and
each song's `accompaniment.wav`, any `prompt.wav`, and cached cover wavs under
`assets/<song>/` — are derivatives of Michael Jackson recordings (Thriller,
Billie Jean, Beat It, Bad, The Way You Make Me Feel, and others) and are included
for demonstration purposes only.
