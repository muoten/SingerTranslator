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

Custom lyrics on a fixed melody sung by a target voice — a control surface
most music AIs don't expose. This repo is the rendering pipeline behind the
**AIchael Jackson** demo at [aichaeljackson.com](https://aichaeljackson.com)
(HF Space: [muoten/aichael-jackson](https://huggingface.co/spaces/muoten/aichael-jackson)).

You type 4 syllables. The pipeline:

1. Loads a frozen *score* (notes + timing) extracted from MJ's actual Thriller
   chorus via [Demucs](https://github.com/facebookresearch/demucs) +
   [Whisper](https://github.com/openai/whisper) +
   [ROSVOT](https://github.com/RickyL-2000/ROSVOT) preprocessing.
2. Cycles your 4 syllables across the 36 sung slots of the chorus, preserving
   notes, durations, and held-vowel melismas.
3. Asks [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) to
   synthesize the voice using a clip of MJ's actual chorus as the timbre
   prompt (so the resulting voice sounds like MJ).
4. Mixes the synthesized vocal with the original Thriller accompaniment.

Result: 16 seconds of synthetic cover with your syllables on MJ's chorus.

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

Visit [aichaeljackson.com](https://aichaeljackson.com) → type 4 syllables →
hit "Make AIchael sing it". Presets: `bue-nos di-as`, `hap-pee birth-day`,
`syn-thet tic-voice`. ~15s per render on the Space's GPU.

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
[user types 4 syllables]
        |
        | singer.build_target_metadata()
        |   - g2p_en → phonemes per syllable
        |   - SYLLABLE_OVERRIDES for known Spanish syllables
        |   - apply double-plosive recipe to slot 1
        |   - assign syllables cyclically to 36 slots
        |   - auto-melisma slots shorter than 0.30s
        v
[target_metadata.json]
        |
        | singer.soulx_render()  →  SoulX CLI inference
        |   - prompt_wav: assets/prompt.wav (MJ chorus 121.5-135.5s)
        |   - prompt_metadata: assets/prompt.json
        |   - target_metadata: above
        |   - --n_steps 16, --seed 100, --pitch_shift 0
        v
[generated.wav]  (16s sung vocal)
        |
        | ffmpeg mix with assets/accompaniment.wav
        v
[cover.wav]
```

### Score data (frozen)

The chorus score in `assets/chorus_target.json` was built once and ships with
the repo. It came from:

1. `vocals.wav` of Thriller, isolated by Demucs.
2. Slice 120-136s (the chorus instance we picked).
3. Whisper-large transcribed the words; ROSVOT transcribed the notes.
4. SoulX preprocess merged them into 34 slots.
5. Manual surgical fix: split one over-merged 1.66s "thriller" slot into 3
   (B4 fresh, rest, B4 fresh) and demoted two spurious melismas. Final: 36
   slots in `chorus_target.json`.

The timbre prompt `assets/prompt.wav` is the same chorus offset by 1.5s
(121.5-135.5s) to avoid an audio-leakage failure mode in SoulX where the
output mimics the prompt audio when prompt and target derive from the same
clip.

### Tunable parameters

`singer.render()` exposes:

- `syllables: list[str]` — the 4 syllables.
- `n_steps: int` — CFM diffusion steps (default 16). Lower = faster, rougher.
- `melisma_mode: 'off' | 'default'` — whether to honour the metadata's tied
  notes as held vowels (default) or force every slot to fresh syllable (off).
- `seed: int | None` — pinned `torch.manual_seed` for reproducible audio.

The shipped buenos dias preset uses `seed=100`, picked from a 5-seed sweep
scored on coverage × phonetic similarity against a Whisper unprompted
transcription.

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
├── app.py                       — Gradio UI (the demo)
├── singer.py                    — render() + build_target_metadata()
├── bootstrap_soulx.py           — downloads weights + NLTK on first run
├── build_buenos_dias.py         — script that built the buenos dias metadata
├── run_preproc_with_whisper.py  — full preproc (Demucs+Whisper+ROSVOT)
├── swap_word.py                 — legacy: replace a word in metadata
├── split_word.py                — legacy: split a word across slots
├── scripts/sing.sh              — SoulX CLI wrapper
├── assets/
│   ├── chorus_target.json       — frozen 36-slot Thriller chorus score
│   ├── prompt.wav               — MJ chorus 121.5-135.5s (timbre prompt)
│   ├── prompt.json              — matching metadata for prompt.wav
│   ├── accompaniment.wav        — Thriller chorus instrumental (16s)
│   └── cache/                   — baked preset outputs
├── data/                        — backup copies of frozen metadata
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

Code under MIT. The `assets/prompt.wav`, `assets/accompaniment.wav`, and
cached preset wavs are derivatives of Michael Jackson's "Thriller" and are
included for demonstration purposes only.
