"""SingerTranslator Gradio UI.

User picks 4 syllables; the app renders MJ singing them on the Thriller
chorus melody, mixes with the accompaniment, and returns the wav.
"""
from __future__ import annotations

import os
import traceback

import gradio as gr

# Run bootstrap first (idempotent): downloads NLTK data + SoulX weights if
# missing. On HF Spaces this is the first-boot ~5GB download; on local dev
# with weights already present it's a no-op.
from bootstrap_soulx import main as bootstrap
bootstrap()

from singer import render

# Optional: HF Spaces ZeroGPU decorator. No-op when not on Spaces.
# duration kept low because ZeroGPU free-tier quota is ~5 min/day and the
# `duration` value is what gets reserved against the quota — not what's
# actually used. Renders at n_steps=32 take ~10-20s on A10G.
try:
    import spaces  # type: ignore
    GPU = spaces.GPU(duration=30)
except ImportError:
    def GPU(fn):  # type: ignore
        return fn


# Songs registry. label → song_dir key. Order = display order.
# Add new songs here once their assets/{key}/ directory exists.
SONGS = {
    "Thriller":               "thriller",
    "Billie Jean (WIP)":      "billie_jean",
}


def _song_ready(song_key: str) -> bool:
    """A song is renderable iff all 4 asset files exist."""
    from singer import prompt_wav, prompt_meta, template_json, accomp_wav
    return all(getter(song_key).exists()
               for getter in (prompt_wav, prompt_meta, template_json, accomp_wav))


@GPU
def _gpu_render(syllables, n_steps, melisma_mode, song):
    """The GPU-bound part. Wrapped separately so the cache-hit path bypasses
    @spaces.GPU entirely (otherwise quota is burned even on cache hits)."""
    return render(syllables, n_steps=n_steps, melisma_mode=melisma_mode, song=song)


def go(s1: str, s2: str, s3: str, s4: str, n_steps: int, melisma_mode: str,
       song_label: str, progress: gr.Progress = gr.Progress()):
    progress(0, desc="checking cache")
    syllables = [s.strip() for s in (s1, s2, s3, s4) if s and s.strip()]
    if not syllables:
        raise gr.Error("Provide at least one syllable.")
    song = SONGS.get(song_label)
    if song is None:
        raise gr.Error(f"Unknown song: {song_label}")
    if not _song_ready(song):
        raise gr.Error(f"'{song_label}' isn't ready yet — assets/{song}/ is missing one or more files (prompt.wav, prompt.json, chorus_target.json, accompaniment.wav).")
    n_steps_i = int(n_steps)
    # Fast path: serve straight from cache (no GPU allocation, no quota).
    from singer import _cache_key, cache_dir, WORK
    key = _cache_key(syllables, n_steps_i, melisma_mode)
    for candidate in (cache_dir(song) / f"{key}_cover.wav",
                      WORK / song / f"{key}_cover.wav"):
        if candidate.exists():
            progress(1.0, desc="cache hit")
            return str(candidate)
    # Slow path: invoke SoulX (will allocate GPU on Spaces).
    try:
        progress(0.1, desc=f"invoking SoulX (n_steps={n_steps_i}, melisma={melisma_mode})")
        wav = _gpu_render(syllables, n_steps_i, melisma_mode, song)
    except Exception as exc:  # surface SoulX/ffmpeg errors clearly
        traceback.print_exc()
        raise gr.Error(f"Render failed: {exc}") from exc
    progress(1.0)
    return wav


PRESETS = {
    "bue-nos di-as":      ("bue", "nos", "di", "as"),
    "mu-chos di-as":      ("mu", "chos", "di", "as"),
    "hap-pee birth-day":  ("hap", "pee", "birth", "day"),
    "mu-chas tar-des":    ("mu", "chas", "tar", "des"),
    "llue-ve mu-cho":     ("llue", "ve", "mu", "cho"),
    "bue-nas tar-des":    ("bue", "nas", "tar", "des"),
    "hoy-no llue-ve":     ("hoy", "no", "llue", "ve"),
}

CSS = """
#title { text-align: center; }
#footer { text-align: center; color: #888; font-size: 0.85em; margin-top: 1em; }
"""

with gr.Blocks(title="AIchael Jackson") as demo:
    gr.Markdown(
        "# 🎤 AIchael Jackson\n"
        "**🤖 AIchael Jackson sings four syllables of your choice on a chosen MJ chorus melody 🧟**",
        elem_id="title",
    )
    ready_songs = [label for label, key in SONGS.items() if _song_ready(key)]
    wip_songs   = [label for label, key in SONGS.items() if not _song_ready(key)]
    song = gr.Dropdown(
        choices=ready_songs,
        value=ready_songs[0],
        label="Song",
        info="Which MJ chorus to sing your syllables on."
             + (f"  (Coming soon: {', '.join(wip_songs)})" if wip_songs else ""),
    )
    with gr.Row():
        s1 = gr.Textbox(value="bue", label="syllable 1", max_lines=1, scale=1, interactive=True)
        s2 = gr.Textbox(value="nos", label="syllable 2", max_lines=1, scale=1, interactive=True)
        s3 = gr.Textbox(value="di",  label="syllable 3", max_lines=1, scale=1, interactive=True)
        s4 = gr.Textbox(value="as",  label="syllable 4", max_lines=1, scale=1, interactive=True)
    gr.Markdown("Or pick a preset (you can still edit any syllable after):")
    with gr.Row():
        preset_buttons = [gr.Button(label, size="sm") for label in PRESETS]
    with gr.Accordion("🔧 Advanced Settings", open=False):
        n_steps = gr.Slider(
            minimum=8, maximum=64, value=16, step=1,
            label="Diffusion steps (n_steps)",
            info=(
                "Number of CFM denoising iterations during synthesis. "
                "32 is the SoulX default and what produced our reference results. "
                "Lower = faster but the voice can sound rougher/less natural; "
                "higher = slower with diminishing returns. On CPU, render time "
                "scales roughly linearly with n_steps. Try 16 for a quick preview."
            ),
        )
        melisma_mode = gr.Radio(
            choices=["off", "default"],
            value="default",
            label="Melisma control",
            info=(
                "How to handle tied notes (one syllable held across multiple notes). "
                "'off' = every slot articulates a fresh syllable (more lyric clarity, "
                "less smooth). 'default' = use the metadata's tied notes as held vowels."
            ),
        )
    btn = gr.Button("Make AIchael sing it 🎶", variant="primary", size="lg")
    out = gr.Audio(label="Result", autoplay=False, type="filepath")
    gr.Markdown(
        "Renders take ~2 min on CPU (≈15 s on GPU). Click once and wait — "
        "the four syllables are cycled across 26 sung slots of the chorus, "
        "preserving AIchael's note timings.",
        elem_id="footer",
    )

    btn.click(go, inputs=[s1, s2, s3, s4, n_steps, melisma_mode, song], outputs=out, show_progress="full")

    # Preset buttons → fill the 4 syllable boxes (boxes remain editable after).
    for button, (_, syllables) in zip(preset_buttons, PRESETS.items()):
        button.click(lambda s=syllables: list(s), outputs=[s1, s2, s3, s4])


if __name__ == "__main__":
    demo.queue(max_size=int(os.environ.get("SINGER_QUEUE_SIZE", 20))).launch(
        # HF Spaces requires 0.0.0.0 so the proxy can reach the container.
        # Override with SINGER_HOST=127.0.0.1 for local-only dev if desired.
        server_name=os.environ.get("SINGER_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("SINGER_PORT", 7860)),
        share=os.environ.get("SINGER_SHARE", "0") == "1",
        # SSR is experimental in gradio 6.x; HF's healthcheck doesn't always
        # play nice with it. Off by default; flip with SINGER_SSR=1.
        ssr_mode=os.environ.get("SINGER_SSR", "0") == "1",
        theme=gr.themes.Soft(),
        css=CSS,
    )
