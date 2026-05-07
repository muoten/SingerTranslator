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
try:
    import spaces  # type: ignore
    GPU = spaces.GPU(duration=180)
except ImportError:
    def GPU(fn):  # type: ignore
        return fn


@GPU
def go(s1: str, s2: str, s3: str, s4: str, n_steps: int,
       progress: gr.Progress = gr.Progress()):
    progress(0, desc="building metadata")
    syllables = [s.strip() for s in (s1, s2, s3, s4) if s and s.strip()]
    if not syllables:
        raise gr.Error("Provide at least one syllable.")
    try:
        progress(0.1, desc=f"invoking SoulX (n_steps={n_steps})")
        wav = render(syllables, n_steps=int(n_steps))
    except Exception as exc:  # surface SoulX/ffmpeg errors clearly
        traceback.print_exc()
        raise gr.Error(f"Render failed: {exc}") from exc
    progress(1.0)
    return wav


PRESETS = {
    "bue-nos di-as":     ("bue", "nos", "di", "as"),
    "mi-chael jack-son": ("mi", "chael", "jack", "son"),
    "hap-py birth-day":  ("hap", "py", "birth", "day"),
}

CSS = """
#title { text-align: center; }
#footer { text-align: center; color: #888; font-size: 0.85em; margin-top: 1em; }
"""

with gr.Blocks(title="AIchael Jackson") as demo:
    gr.Markdown(
        "# 🎤 AIchael Jackson\n"
        "**🤖 AIchael Jackson sings four syllables of your choice on the Thriller chorus melody 🧟**",
        elem_id="title",
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
            minimum=8, maximum=64, value=32, step=1,
            label="Diffusion steps (n_steps)",
            info=(
                "Number of CFM denoising iterations during synthesis. "
                "32 is the SoulX default and what produced our reference results. "
                "Lower = faster but the voice can sound rougher/less natural; "
                "higher = slower with diminishing returns. On CPU, render time "
                "scales roughly linearly with n_steps. Try 16 for a quick preview."
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

    btn.click(go, inputs=[s1, s2, s3, s4, n_steps], outputs=out, show_progress="full")

    # Preset buttons → fill the 4 syllable boxes (boxes remain editable after).
    for button, (_, syllables) in zip(preset_buttons, PRESETS.items()):
        button.click(lambda s=syllables: list(s), outputs=[s1, s2, s3, s4])


if __name__ == "__main__":
    demo.queue(max_size=int(os.environ.get("SINGER_QUEUE_SIZE", 20))).launch(
        server_name=os.environ.get("SINGER_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("SINGER_PORT", 7860)),
        share=os.environ.get("SINGER_SHARE", "0") == "1",
        theme=gr.themes.Soft(),
        css=CSS,
    )
