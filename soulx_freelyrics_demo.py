"""
Gradio UI for soulx_freelyrics — pick a song, type a lyric, see the live syllable
check, and render it on that song's LOCKED chorus melody/timing with SoulX.

  SINGER_DEVICE=cpu vendor/SoulX-Singer/venv/bin/python soulx_freelyrics_demo.py
  -> http://localhost:7863   (7862 is the Vevo2 demo)
"""
import os
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import singer  # noqa: E402
import soulx_freelyrics as fl  # noqa: E402

OUT = ROOT / "_tmp_freelyrics_demo"
OUT.mkdir(exist_ok=True)

# Optional HF Spaces ZeroGPU decorator. No-op when `spaces` isn't installed (local).
try:
    import spaces  # type: ignore
    GPU = spaces.GPU(duration=int(os.environ.get("SINGER_GPU_DURATION", 120)))
except ImportError:
    def GPU(fn):  # type: ignore
        return fn

# Display label -> song key. Order = dropdown order.
SONGS = {"Thriller": "thriller", "Billie Jean": "billie_jean"}


@GPU
def _gpu_render(tgt, out_dir, n_steps):
    """GPU-bound SoulX inference, isolated so @spaces.GPU only wraps the heavy part."""
    return singer.soulx_render(tgt, out_dir, n_steps=int(n_steps), seed=0)


def _cache_key(song, lines, n_steps, reinforce):
    """Stable hash of everything that affects the audio. Lyric is normalized on
    words (case- and whitespace-insensitive) so trivial spacing doesn't miss."""
    import hashlib
    lyric = "/".join(" ".join(l.lower().split()) for l in lines)
    norm = f"{song}|{lyric}::{int(n_steps)}::r{int(bool(reinforce))}"
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


def header_md(song_key):
    return fl.template_md(song_key)


def check_md(text, song_label):
    song = SONGS[song_label]
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    sp = fl.slots_by_phrase(song)
    out, all_ok = ["### Syllable + articulation check"], True
    tally = {"🟢": 0, "🟡": 0, "🔴": 0}
    notes = []
    for p in sp:
        tw = lines[p - 1].split() if p - 1 < len(lines) else []
        need = len(sp[p])
        nsyl = sum(fl.syl(x[1]) for x in sp[p])
        if len(tw) != need:
            all_ok = False
            out.append(f"**P{p}** *(needs {nsyl} syl / {need} words)* — ⚠️ you have **{len(tw)}** words")
            continue
        parts = []
        for slot, w in zip(sp[p], tw):
            ow = slot[1]
            a, b = fl.syl(ow), fl.syl(w)
            score, reasons = singer.word_difficulty(w)
            dot = fl._dot(score)
            tally[dot] += 1
            if a == b:
                parts.append(f"{dot} {w}")
            else:
                all_ok = False
                parts.append(f"❌ {w}\\({b}≠{a}\\)")
            if score >= 2 and a == b:
                notes.append(f"- {dot} **{w}** — {', '.join(reasons)}")
        out.append(f"**P{p}** *(needs {nsyl} syl)* — " + " · ".join(parts))
    hard = tally["🟡"] + tally["🔴"]
    out.append(f"**Difficulty:** 🟢 {tally['🟢']} · 🟡 {tally['🟡']} · 🔴 {tally['🔴']}")
    if notes:
        out.append("\n".join(notes))
    if not all_ok:
        out.append("❌ **fix the ⚠️/❌ words first** — syllable / word count must match")
    elif hard:
        out.append(f"⚠️ **fits, but {hard} word(s) may garble** — edit 🔴/🟡 toward 🟢 "
                   "(keep the same syllable count) for cleaner articulation before rendering")
    else:
        out.append("✅ **all green — ready to render**")
    return "\n\n".join(out)


def on_song_change(song_label):
    """Swap header, default lyric, and the check panel when the song changes."""
    song = SONGS[song_label]
    default = "\n".join(fl.DEMOS[song])
    return header_md(song), default, check_md(default, song_label)


def render(text, name, reinforce, n_steps, song_label):
    song = SONGS[song_label]
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    ok, words = fl.check(lines, song)
    if not ok:
        raise gr.Error("Lyric doesn't fit the template — see the check panel.")

    # Cache by (song + lyric + n_steps + reinforce). A hit returns the saved wavs
    # WITHOUT calling _gpu_render, so it never allocates the GPU (saves quota).
    key = _cache_key(song, lines, n_steps, reinforce)
    cdir = singer.cache_dir(song); cdir.mkdir(parents=True, exist_ok=True)
    cmix, cvoc = cdir / f"fl_{key}_mix.wav", cdir / f"fl_{key}_vocal.wav"
    if cmix.exists() and cvoc.exists():
        return str(cmix), str(cvoc)

    tgt = fl.build_target(words, OUT / f"fl_{key}_target.json", song=song, reinforce=bool(reinforce))
    vocal = _gpu_render(tgt.resolve(), OUT.resolve(), int(n_steps))
    cvoc.write_bytes(Path(vocal).read_bytes())
    mix = singer.mix_with_accompaniment(cvoc, cmix, song=song)
    return str(mix), str(cvoc)


def build():
    first_label = next(iter(SONGS))
    first_key = SONGS[first_label]
    first_default = "\n".join(fl.DEMOS[first_key])
    with gr.Blocks(title="AIchael Jackson", css="#aj-title{text-align:center}") as demo:
        gr.Markdown(
            "# 🎤 AIchael Jackson\n"
            "**🤖 AIchael Jackson sings the words *you* write on an MJ chorus melody 🧟**",
            elem_id="aj-title",
        )
        song = gr.Dropdown(choices=list(SONGS), value=first_label, label="🎵 Song",
                           info="Which MJ chorus to sing your words on.")
        with gr.Row():
            with gr.Column(scale=3):
                target = gr.Textbox(label="✍️ Your lyric — one line per phrase",
                                    value=first_default, lines=4)
                run = gr.Button("🎤 Make AIchael sing it (~2 min)", variant="primary", size="lg")
                chk = gr.Markdown(check_md(first_default, first_label))
                with gr.Accordion("📐 How many words / syllables per line?", open=True):
                    head = gr.Markdown(header_md(first_key))
            with gr.Column(scale=2):
                mix_audio = gr.Audio(label="Cover (vocal + accompaniment)", type="filepath")
                dry_audio = gr.Audio(label="Dry vocal", type="filepath")
                with gr.Accordion("⚙️ Settings", open=False):
                    name = gr.Textbox(label="Name (output filename)", value="mytest")
                    reinforce = gr.Checkbox(
                        label="Reinforce weak onsets (double HH/R/L + clusters)",
                        value=False)
                    n_steps = gr.Slider(20, 64, value=32, step=2,
                                        label="Diffusion steps (higher = cleaner, slower)")
        song.change(fn=on_song_change, inputs=song, outputs=[head, target, chk])
        target.change(fn=check_md, inputs=[target, song], outputs=chk)
        run.click(fn=render, inputs=[target, name, reinforce, n_steps, song],
                  outputs=[mix_audio, dry_audio])
    return demo


if __name__ == "__main__":
    build().launch(server_name="0.0.0.0",
                   server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7863)))
