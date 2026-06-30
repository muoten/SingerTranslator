# Backstage — adding songs & internals

How songs get built, gated, and shipped behind the **AIchael Jackson** demo.
For *using* the demo (the webapp, inputs, running locally), see the
[README](README.md).

## Build pipeline (`build_song.py`)

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

Stages skip if their output exists and are resumable (`--from`, `--only`,
`--force`). Most run automatically; the three ear-judgments — **window**,
**verify**, **order** — stop and ask you to confirm.

The karaoke step matters: some choruses are sung by MJ *in unison with a choir*,
which no separator can cleanly split — the isolated lead comes out smeared and the
render is poor. That case is flagged automatically (see *Quality gates*).

### Adding a song, end to end

1. `python backstage/build_song.py --song <key>` — runs `fetch → … → register`, pausing at
   the ear-judgments (pick the chorus `--window`, confirm the neutral-`la`
   verify, confirm the `order`).
2. Paste the printed `ORDERS` / `DEMOS` snippet into `soulx_freelyrics.py`.
3. `python backstage/bake_song.py --song <key>` — renders several seeds, keeps the
   best by timbre, mixes, caches the cover, and runs the **quality gates**.
4. A song appears in the demo only when it is registered **and**
   `assets/<song>/config.json` has `"demo": true`. New songs default to hidden,
   so a song can be fully built and baked yet kept out of the demo.

## Timbre prompt

Rather than a per-song clip, all songs share one canonical MJ voice prompt,
`assets/_shared/mj_prompt.wav` — a clean Thriller *verse* clip. Because it's verse
audio it differs from every song's *chorus* target, which sidesteps a SoulX
audio-leakage failure mode (the output mimics the prompt when prompt and target
derive from the same clip). `singer.prompt_wav()` falls back to the shared prompt
unless a song ships its own `assets/<song>/prompt.wav` override.

## Quality gates (demo-eligibility)

Whether a baked cover is good enough for the public demo is decided by two
automatic, complementary checks — a render is demo-eligible only if **both** pass:

- **scat gate** (`backstage/crappy_fragments.py`) — measures the total duration of
  *grid-mismatched* audio: sound during a rest (PHANTOM), wrong pitch on a note
  (OFFPITCH), or silence on a note (DEAD). `total_crappy < 5.0s` = pass. This
  catches a render that fails to track an otherwise-good grid.
- **solo-vs-chorus gate** (`backstage/validate_lead.py`) — measures how much
  *other-vocal* energy surrounded the isolated lead (from the karaoke step's
  removed-backing output). A high ratio means the chorus is choral (lead in
  unison with a choir) → the grid itself is corrupt → the scat gate can't see it.

The bake (`backstage/bake_song.py`) runs both, auto-sets `demo: false` on failure,
and only *recommends* `demo: true` (publishing is a human confirm).
Phoneme-recognition metrics (per-slot, recall/precision/F1, Whisper, timbre-sim)
were all tried and **rejected** — they don't track perceived quality.

## Phonetic learnings (English g2p approximating Spanish)

- **Plosives in short slots get swallowed** — double the leading consonant
  via the override (e.g. `bue` → `en_B-B-W-EH1`). Recipe is auto-applied to
  slot 1 only.
- **L coda on high notes drops** — `sole` becomes "soy". Fix: split the word
  across two slots so L moves to onset position (`soh / lee`).
- **`SYLLABLE_OVERRIDES`** in `singer.py` pins g2p_en's output for known
  Spanish syllables it would otherwise mispronounce (e.g. `nos` → `N-OW1-S`
  instead of g2p_en's `N-AA1-S`).

## Repo layout

```
.
├── app.py                       — HF Space entry: bootstrap + launch the demo
├── soulx_freelyrics_demo.py     — Gradio UI (song picker, lyric box, render+cache)
├── soulx_freelyrics.py          — per-song engine: word→slot mapping (ORDERS/DEMOS)
├── singer.py                    — soulx_render(), g2p, mixing, prompt fallback, config
├── bootstrap_soulx.py           — downloads weights + NLTK on first run
├── assets/
│   ├── _shared/mj_prompt.{wav,json}   — canonical MJ timbre prompt (shared by all)
│   └── <song>/                        — thriller, billie_jean, beat_it, bad, …
│       ├── chorus_target.json         — frozen chorus score (notes + timing)
│       ├── accompaniment.wav          — chorus instrumental (~16s)
│       ├── config.json                — per-song knobs incl. the "demo" flag
│       ├── prompt.{wav,json}          — OPTIONAL per-song timbre-prompt override
│       └── cache/                     — rendered covers (incl. pre-baked defaults)
├── backstage/                   — all offline song-building code (not loaded by the app)
│   ├── build_song.py                  — add a song: fetch→…→register (resumable stages)
│   ├── run_preproc_with_whisper.py    — SoulX preprocess wrapper (karaoke+Whisper+ROSVOT)
│   ├── bake_song.py                   — seed-best render + auto quality gates
│   ├── crappy_fragments.py            — scat gate (grid-mismatch duration)
│   ├── validate_lead.py               — solo-vs-chorus gate
│   ├── split_word.py, swap_word.py    — per-slot word editing helpers
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
