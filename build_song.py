"""build_song.py — add a new song to the SingerTranslator pipeline, end to end.

Phase 2 of song-build standardization. Turns "days of manual slog" into a
resumable stage pipeline: the mechanical steps run automatically; the few
ear-judgment steps stop and ask you to confirm.

PIPELINE (each stage skips if its output already exists; --force re-runs):
  0. fetch      yt-dlp ytsearch by --title -> sources/<song>/<song>_full.wav   [shell-out]
                + validate (duration/sr; flags snippets & compilations)
  1. separate   demucs full track -> vocals.wav + accompaniment (no_vocals)   [shell-out]
  2. slice      cut the chorus WINDOW from vocals/accomp/full                  [auto, ffmpeg]
  3. preproc    run_preproc_with_whisper.py on the chorus vocal               [shell-out]
                (Whisper align w/ true lyrics as initial_prompt + f0 + notes)
  4. grid       preproc output -> assets/<song>/chorus_target.json            [needs-verify]
  5. prompt     prompt.wav (chorus offset ~PROMPT_OFFSET s, anti-leakage)     [auto, ffmpeg]
  6. accomp     accompaniment.wav (vocal-stripped chorus, faded)              [auto, ffmpeg]
  7. verify     neutral-'la' synthesis + leakage gate (copyright-safe)        [EAR: confirm]
                (FAIL = original words bleed -> build a verse anti-leak prompt)
  8. config     assets/<song>/config.json (defaults; tune later)              [auto]
  9. order      PROPOSE the note->word ORDER by de-inflation heuristic        [EAR: confirm]
 10. register   print the ORDERS / DEMOS snippet + how to surface in the demo [auto]

A song is only ever shown in the demo when it is registered (ORDERS/DEMOS) AND its
config.json has "demo": true. New songs default to demo=false, so they can be fully
built/baked yet stay hidden until promoted (or kept hidden permanently).

EAR-JUDGMENT (the irreducible craft — you supply / confirm these):
  - WINDOW    which chorus instance (--window START:END), auto-proposes none yet
  - VERIFY    neutral-'la' render: clean on-melody 'la' = PASS; original words = leak
  - ORDER     which slots voice cleanly vs rest (stage 9 proposes, you confirm)
  - ALIGNMENT spot-check Whisper transcript vs the true lyric (printed in stage 3)
  - SEED      picked later at bake time (scripts/bake_*), not here

PREREQUISITES (see feedback_soulx_preproc_macos_gotchas):
  - demucs installed (its own venv); run setup_rosvot_macos.sh after every reboot
  - SoulX-Singer checkout at SOULX_ROOT with its venv (Whisper, ROSVOT)
  - run long stages directly (not nohup&); WHISPER_INITIAL_PROMPT carries the lyric

Usage:
  python build_song.py --song bad --only fetch       # download+validate the source by title
  python build_song.py --song bad --window 60.0:76.0 \\
      --lyrics "<true chorus lyric, ALIGNMENT ONLY>" --device cpu   # fetch->...->register
  python build_song.py --song bad --title "Michael Jackson Bad official audio" --only fetch
  python build_song.py --song bad --only order       # re-propose the ORDER
  python build_song.py --song bad --from grid        # resume from a stage
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
import singer  # noqa: E402  (asset paths + song_config defaults)

SOULX_ROOT = Path(os.environ.get("SOULX_ROOT", "/Users/milhouse/claude-code/SoulX-Singer"))
PROMPT_OFFSET = 1.5     # seconds; prompt clip offset from target to avoid SoulX audio leakage
MIN_AUDIBLE = 0.30      # seconds; slots shorter than this can't articulate a fresh syllable
FADE = 0.12             # seconds; accompaniment/vocal tail fade


def sources_dir(song: str) -> Path:
    return ROOT / "sources" / song


# ---------------- tiny state machine (resumable) ----------------------------

def _state_path(song: str) -> Path:
    return sources_dir(song) / "build_state.json"


def _load_state(song: str) -> dict:
    p = _state_path(song)
    return json.loads(p.read_text()) if p.exists() else {"done": []}


def _save_state(song: str, st: dict):
    _state_path(song).parent.mkdir(parents=True, exist_ok=True)
    _state_path(song).write_text(json.dumps(st, indent=2))


def _run(cmd: list[str], **kw):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kw)


# ---------------- stages ----------------------------------------------------
# Each stage(song, args) returns None. Convention: read inputs from sources/<song>/,
# write assets/<song>/ outputs. Raise to abort; print a clear next-step on stubs.

def _full_track(song: str) -> Path:
    """The song's full-track wav. Prefers <song>_full.wav, else any *_full.wav in
    sources/<song>/ (back-compat with sc_full.wav etc.), else the canonical path."""
    d = sources_dir(song)
    canonical = d / f"{song}_full.wav"
    if canonical.exists():
        return canonical
    other = next(iter(sorted(d.glob("*_full.wav"))), None)
    return other or canonical


def _validate_source(full: Path):
    """Sanity-check a downloaded source. Catches the failure modes we actually hit:
    a 30s preview/snippet, or a multi-song compilation/mix (e.g. a 13-min upload)."""
    import soundfile as sf
    info = sf.info(str(full))
    dur, sr, ch = info.duration, info.samplerate, info.channels
    print(f"  source: {full.name}  dur={dur:.1f}s ({dur/60:.2f} min)  sr={sr}  ch={ch}")
    problems = []
    if dur < 90:
        problems.append(f"only {dur:.0f}s — likely a preview/snippet, not the full track")
    if dur > 600:
        problems.append(f"{dur/60:.1f} min — likely a compilation/mix, not a single song")
    if sr < 32000:
        problems.append(f"low sample rate {sr} — poor source quality")
    if problems:
        print("  >>> SOURCE WARNING: " + "; ".join(problems))
        print("      Re-run fetch with a better --title query (or --force) if this is wrong.")
    else:
        print("  source looks like a plausible full single track.")
    print("  >>> EAR-CONFIRM it is the right song/version before building on it.")


def stage_fetch(song, args):
    """Download the source full track by TITLE (yt-dlp ytsearch) + validate it.

    Standardizes the acquisition front of the pipeline. Default query is
    'Michael Jackson <Label> official audio'; override with --title. Skips if a
    *_full.wav already exists for the song."""
    existing = _full_track(song)
    if existing.exists() and not args.force:
        print(f"  [skip] source exists: {existing}")
        _validate_source(existing); return
    query = args.title or f"Michael Jackson {singer.song_label(song)} official audio"
    dest = sources_dir(song) / f"{song}_full.wav"
    sources_dir(song).mkdir(parents=True, exist_ok=True)
    print(f"  yt-dlp ytsearch1: {query!r}")
    _run(["yt-dlp", "-x", "--audio-format", "wav",
          "-o", str(sources_dir(song) / f"{song}_full.%(ext)s"),
          f"ytsearch1:{query}"])
    _validate_source(dest)


def stage_separate(song, args):
    """demucs the full track -> sources/<song>/_sep/htdemucs/<stem>/{vocals,no_vocals}.wav"""
    full = Path(args.source) if args.source else _full_track(song)
    if not full.exists():
        raise SystemExit(f"no source track: {full} (run stage 'fetch' or pass --source)")
    out = sources_dir(song) / "_sep"
    voc = next(out.glob("**/vocals.wav"), None)
    if voc and not args.force:
        print(f"  [skip] separated vocals exist: {voc}")
        return
    print("  demucs two-stem separation (needs the demucs venv on PATH)")
    _run(["demucs", "--two-stems", "vocals", "-o", str(out), str(full)])


def stage_slice(song, args):
    """Cut the chorus WINDOW from vocals + accompaniment into sources/<song>/."""
    if not args.window:
        raise SystemExit("stage 'slice' needs --window START:END (the chorus instance)")
    a, b = (float(x) for x in args.window.split(":"))
    sep = sources_dir(song) / "_sep"
    voc_full = next(sep.glob("**/vocals.wav"), None)
    acc_full = next(sep.glob("**/no_vocals.wav"), None)
    if not (voc_full and acc_full):
        raise SystemExit("run stage 'separate' first (no vocals/no_vocals found)")
    for src, name in [(voc_full, "chorus_vocal.wav"), (acc_full, "chorus_accomp.wav")]:
        dst = sources_dir(song) / name
        if dst.exists() and not args.force:
            print(f"  [skip] {dst.name}"); continue
        _run(["ffmpeg", "-y", "-ss", str(a), "-to", str(b), "-i", str(src),
              "-ar", "44100", str(dst)], capture_output=True)
    print(f"  chorus window {a:.2f}-{b:.2f}s ({b-a:.2f}s) -> sources/{song}/chorus_*.wav")


def stage_preproc(song, args):
    """SoulX preprocess (Whisper align + f0 + ROSVOT notes) on the chorus vocal."""
    voc = sources_dir(song) / "chorus_vocal.wav"
    save = sources_dir(song) / "preproc"
    if (save / "metadata.json").exists() and not args.force:
        print(f"  [skip] preproc metadata exists: {save/'metadata.json'}"); return
    env = dict(os.environ)
    if args.lyrics:
        env["WHISPER_INITIAL_PROMPT"] = args.lyrics    # ALIGNMENT ONLY (not stored as-is)
    # run_preproc_with_whisper.py needs the SoulX venv (whisper / ROSVOT / preprocess.*)
    soulx_py = SOULX_ROOT / "venv" / "bin" / "python"
    py = str(soulx_py) if soulx_py.exists() else sys.executable
    # SoulX preproc loads its f0/ROSVOT weights via paths RELATIVE to SOULX_ROOT,
    # so the subprocess must run from there (audio_path/save_dir are absolute).
    _run([py, str(ROOT / "run_preproc_with_whisper.py"),
          "--audio_path", str(voc.resolve()), "--save_dir", str(save.resolve()),
          "--language", args.language, "--device", args.device,
          "--vocal_sep", "False", "--midi_transcribe", "True"],
         env=env, cwd=str(SOULX_ROOT))
    print("  >>> EAR-CHECK: compare the printed Whisper text above against your lyric.")


def stage_grid(song, args):
    """Place the preproc metadata as assets/<song>/chorus_target.json.

    The SoulX preprocess already emits our exact 9-key schema (index, language,
    time, duration, text, phoneme, note_pitch, note_type, f0), so this is just a
    relabel + copy. The result is the RAW measured grid — buried syllables sit on
    pitch-0, spikes are unclamped; clean it later (cf. reconstruct_beat_it_grid.py)
    once you've heard it.
    """
    meta_p = sources_dir(song) / "preproc" / "metadata.json"
    if not meta_p.exists():
        raise SystemExit("run stage 'preproc' first")
    item = dict(json.loads(meta_p.read_text())[0])
    expect = ["duration", "text", "phoneme", "note_pitch", "note_type", "f0"]
    missing = [k for k in expect if k not in item]
    if missing:
        raise SystemExit(f"preproc metadata missing {missing}; keys: {list(item)}")
    item["index"] = f"{song}_chorus"
    item["language"] = args.language
    n = len(item["note_pitch"].split())
    # keep the raw measured grid as a backup (mirrors beat_it's roformer_raw)
    raw = singer.template_json(song).with_suffix(".raw.json")
    dst = singer.template_json(song)
    dst.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps([item], indent=2)
    raw.write_text(blob); dst.write_text(blob)
    voiced = sum(1 for t in item["note_type"].split() if t != "1")
    print(f"  wrote {dst} ({n} slots, {voiced} voiced) + {raw.name} backup")
    print(f"  >>> VERIFY by ear: text='{item['text'][:60]}...'")


def stage_prompt(song, args):
    """Resolve the prompt — default is the shared canonical MJ voice (anti-leak for
    any chorus target). No per-song prompt is cut: singer.prompt_wav/meta fall back
    to assets/_shared/mj_prompt.* when a song has no prompt.* of its own. To override
    for a specific song, drop a verse-derived prompt.{wav,json} in assets/<song>/."""
    pw, pj = singer.prompt_wav(song), singer.prompt_meta(song)
    if not (pw.exists() and pj.exists()):
        raise SystemExit(
            f"no prompt available for {song}: neither assets/{song}/prompt.* nor the "
            f"canonical {singer.MJ_PROMPT_WAV} exist. Install the canonical MJ prompt.")
    own = (singer.song_dir(song) / "prompt.wav").exists()
    print(f"  [ok] prompt = {'per-song override' if own else 'shared canonical MJ voice'}: {pw}")


def stage_accomp(song, args):
    """accompaniment.wav = the chorus instrumental, faded tail."""
    acc = sources_dir(song) / "chorus_accomp.wav"
    dst = singer.accomp_wav(song)
    if dst.exists() and not args.force:
        print(f"  [skip] {dst.name}"); return
    import soundfile as sf
    dur = sf.info(str(acc)).duration
    st = max(0.0, dur - FADE)            # fade out over the LAST FADE seconds (not the first!)
    _run(["ffmpeg", "-y", "-i", str(acc), "-af", f"afade=t=out:st={st:.3f}:d={FADE}:curve=tri",
          "-ar", "44100", "-ac", "2", str(dst)], capture_output=True)
    print(f"  wrote {dst} (vocal-stripped chorus). If vocal bleeds through, redo "
          "separation with mel-band-roformer karaoke + dereverb.")


def stage_config(song, args):
    """Write assets/<song>/config.json seeded with safe defaults (tune later)."""
    dst = singer.config_json(song)
    if dst.exists() and not args.force:
        print(f"  [skip] {dst.name} exists"); return
    cfg = {**singer.SONG_CONFIG_DEFAULTS, "mix": dict(singer.SONG_CONFIG_DEFAULTS["mix"])}
    if args.window:
        a, b = (float(x) for x in args.window.split(":"))
        cfg["accomp_len"] = round(b - a, 2)
    dst.write_text(json.dumps(cfg, indent=2))
    print(f"  wrote {dst} (defaults — set hold_dur / f0_clamp_semi / mix / trim after listening)")
    (singer.cache_dir(song)).mkdir(parents=True, exist_ok=True)


def stage_order(song, args):
    """EAR-CONFIRM: propose the note->word ORDER by de-inflation.

    Heuristic: a slot is AUDIBLE (a sung word) if note_type==2 and duration >=
    MIN_AUDIBLE and pitch>0; everything else is a rest. Phrases split on long
    rests. You then map real cover words onto the audible slots & adjust by ear.
    """
    g = json.loads(singer.template_json(song).read_text())[0]
    pit = [int(x) for x in g["note_pitch"].split()]
    typ = [int(x) for x in g["note_type"].split()]
    dur = [float(x) for x in g["duration"].split()]
    order, phrase, since_word = [], 1, 0.0
    audible = 0
    for i, (p, t, d) in enumerate(zip(pit, typ, dur)):
        is_word = (t == 2 and d >= MIN_AUDIBLE and p > 0)
        if is_word:
            order.append(("w", "la", [i], phrase)); audible += 1; since_word = 0.0
        else:
            order.append(("R", i)); since_word += d
            if since_word > 0.8 and audible:        # long gap -> next phrase
                phrase += 1
    print(f"  proposed {audible} audible slots across {phrase} phrase(s):\n")
    print(f"{song.upper()}_ORDER = [")
    for e in order:
        print(f"    {e!r},")
    print("]")
    print("\n  >>> EAR-CONFIRM: rename 'la' to your per-slot reference words, fix phrase "
          "boundaries, and rest/merge any slot that doesn't voice cleanly (cf. BEAT_IT_ORDER).")


def stage_verify(song, args):
    """Neutral-'la' synthesis + leakage gate (copyright-safe). EAR-judgment.

    Renders the chorus grid on a neutral 'la' through the SAME path the real cover
    uses (so speed/articulation match), with the song's shipping prompt. You can't
    use the real lyrics here: they trip copyright AND can't distinguish correct
    synthesis from prompt leakage (both produce the original words). With 'la',
    any original words you hear are unambiguously leakage -> swap to a verse
    (anti-leakage) prompt and re-run. See scripts/verify_neutral.py.
    """
    mix = ROOT / "_verify_neutral" / song / f"{song}_neutral_mix.wav"
    if mix.exists() and not args.force:
        print(f"  [skip] neutral check exists: {mix}"); return
    env = dict(os.environ); env["SINGER_DEVICE"] = args.device
    _run([sys.executable, str(ROOT / "scripts" / "verify_neutral.py"),
          "--song", song], env=env)
    print(f"  >>> EAR-CHECK: PASS = clean on-melody 'la la la' in the song's voice; "
          f"FAIL = original words bleed (prompt leaks). {mix}")


def stage_register(song, args):
    """Print the registration snippet + how to surface the song in the demo."""
    n = 1
    try:
        g = json.loads(singer.template_json(song).read_text())[0]
        n = max(1, sum(1 for t in g["note_type"].split() if t == "2"))
    except Exception:
        pass
    print("  paste into soulx_freelyrics.py:")
    print(f'    ORDERS["{song}"] = {song.upper()}_ORDER')
    print(f'    DEMOS["{song}"]  = ["<line1>", "<line2>", "<line3>", "<line4>"]  '
          f'# ~{n} syllables total, matched per phrase')
    print(f"  to SURFACE in the demo, set in assets/{song}/config.json:")
    print('    "demo": true            # default false = baked-but-hidden')
    print("  (the demo auto-lists registered songs with demo=true; no SONGS edit needed)")


STAGES = [
    ("fetch", stage_fetch),
    ("separate", stage_separate), ("slice", stage_slice), ("preproc", stage_preproc),
    ("grid", stage_grid), ("prompt", stage_prompt), ("accomp", stage_accomp),
    ("verify", stage_verify),
    ("config", stage_config), ("order", stage_order), ("register", stage_register),
]
STAGE_NAMES = [n for n, _ in STAGES]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", required=True, help="song key, e.g. 'bad'")
    ap.add_argument("--title", help="search query for stage 'fetch' (default: 'Michael Jackson <Label> official audio')")
    ap.add_argument("--source", help="full-track wav override (else sources/<song>/<song>_full.wav)")
    ap.add_argument("--window", help="chorus window START:END in seconds, e.g. 60.0:76.0")
    ap.add_argument("--lyrics", default="", help="true chorus lyric — ALIGNMENT ONLY (Whisper initial_prompt)")
    ap.add_argument("--language", default="English")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--from", dest="from_stage", choices=STAGE_NAMES, help="resume from this stage")
    ap.add_argument("--only", choices=STAGE_NAMES, help="run only this stage")
    ap.add_argument("--force", action="store_true", help="re-run even if outputs exist")
    args = ap.parse_args()

    sources_dir(args.song).mkdir(parents=True, exist_ok=True)
    st = _load_state(args.song)

    if args.only:
        todo = [args.only]
    else:
        start = STAGE_NAMES.index(args.from_stage) if args.from_stage else 0
        todo = STAGE_NAMES[start:]

    for name, fn in STAGES:
        if name not in todo:
            continue
        print(f"\n=== stage: {name} ===")
        fn(args.song, args)
        if name not in st["done"]:
            st["done"].append(name); _save_state(args.song, st)
    print(f"\nstate: {sorted(set(st['done']))}")


if __name__ == "__main__":
    main()
