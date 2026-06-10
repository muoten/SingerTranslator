"""SingerTranslator / AIchael Jackson — Hugging Face Space entry point.

Serves the free-lyric cover demo: pick a song (Thriller or Billie Jean), write
your own words on the locked MJ chorus melody, and SoulX sings them.

The UI + render logic live in soulx_freelyrics_demo.py (which also runs standalone
on :7863 for local dev). This module only adds the Spaces plumbing: first-boot
weight/NLTK download, then launch.
"""
from __future__ import annotations

import os

import gradio as gr  # noqa: F401  (ensures gradio is importable before launch)

# First-boot bootstrap (idempotent): downloads NLTK data + SoulX weights if
# missing. On HF Spaces this is the ~5GB first-boot download; a no-op locally.
from bootstrap_soulx import main as bootstrap
bootstrap()

from soulx_freelyrics_demo import build  # noqa: E402


if __name__ == "__main__":
    build().queue(max_size=int(os.environ.get("SINGER_QUEUE_SIZE", 20))).launch(
        # HF Spaces requires 0.0.0.0 so the proxy can reach the container.
        server_name=os.environ.get("SINGER_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("SINGER_PORT", 7860)),
        share=os.environ.get("SINGER_SHARE", "0") == "1",
    )
