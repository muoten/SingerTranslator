"""bake_song.py — STANDARD song-agnostic bake step.

Generalizes bake_beat_it_demo.py: build the registered DEMOS target -> render N
seeds -> pick the best by TIMBRE-sim (validated naturalness metric, WavLM-SV vs
prompt.wav) -> trim the trailing synth tail per the song's config trim recipe ->
mix (config gains + optional dBFS ceiling) -> bake mix + trimmed vocal into
assets/<song>/cache/ so the demo serves it instantly.

A baked song is NOT necessarily a demo song: visibility is the config 'demo' flag
([[project_neutral_verify_and_demo_flag]]). Baking a hidden song archives it.

Usage:
  SINGER_DEVICE=cpu vendor/SoulX-Singer/venv/bin/python backstage/bake_song.py \
      --song smooth_criminal > sources/smooth_criminal/bake.log 2>&1 &
"""
from __future__ import annotations
import argparse, sys, time, json, hashlib, shutil
from pathlib import Path
import numpy as np, soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "backstage"))
import soulx_freelyrics as fl
import singer
import score_melody_timbre as sb   # timbre_sim: song-agnostic WavLM-SV cosine vs prompt


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _trim_end(trim_spec, on, typ):
    """Per-song trim recipe (config['trim']) -> end time (s), or None."""
    if not trim_spec:
        return None
    if trim_spec.startswith("last_sung+"):
        off = float(trim_spec.split("+", 1)[1])
        last_sung = max(i for i, t in enumerate(typ) if t != 1)
        return on[last_sung + 1] + off
    raise ValueError(f"unknown trim recipe {trim_spec!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", required=True, choices=list(fl.ORDERS))
    ap.add_argument("--seeds", type=int, default=6, help="how many seeds to search")
    ap.add_argument("--n_steps", type=int, default=32)
    args = ap.parse_args()
    SONG = args.song
    OUT = ROOT / "_tmp_bake" / SONG; OUT.mkdir(parents=True, exist_ok=True)

    lines = fl.DEMOS[SONG]
    ok, words = fl.check(lines, SONG)
    if not ok:
        raise SystemExit("demo lyric doesn't fit template")
    tgt = OUT / "target.json"
    fl.build_target(words, tgt, song=SONG)
    prompt = str(singer.prompt_wav(SONG))   # per-song prompt or canonical MJ fallback

    # 1) seed search by timbre
    best = None
    for s in range(args.seeds):
        sdir = OUT / f"seed{s}"; sdir.mkdir(exist_ok=True)
        voc = sdir / "generated.wav"
        if not voc.exists():
            log(f"seed {s}: render ...")
            singer.soulx_render(tgt.resolve(), sdir.resolve(), n_steps=args.n_steps, seed=s, song=SONG)
        t = sb.timbre_sim(str(voc), prompt)
        log(f"seed {s}: timbre={t:.4f}")
        if best is None or t > best[1]:
            best = (s, t, voc)
    log(f"best seed {best[0]} timbre={best[1]:.4f}")

    cfg = singer.song_config(SONG)

    # 2) trim trailing synth tail per the song's trim recipe (config-driven)
    d = json.loads(tgt.read_text())[0]
    dur = [float(x) for x in d["duration"].split()]
    typ = [int(x) for x in d["note_type"].split()]
    on = np.concatenate([[0], np.cumsum(dur)])
    y, sr = sf.read(best[2])
    end = _trim_end(cfg["trim"], on, typ)
    if end is not None:
        y = y[:int(end * sr)].copy()
        f = int(0.12 * sr); y[-f:] *= np.linspace(1, 0, f)
        log(f"trimmed vocal to {end:.2f}s")
    voc_trim = OUT / "vocal_final.wav"; sf.write(voc_trim, y, sr)

    # 3) mix (gains from config) + optional ceiling (config)
    mix = OUT / "mix_final.wav"
    singer.mix_with_accompaniment(voc_trim, mix, song=SONG)
    ceiling_db = cfg["mix"]["ceiling_db"]
    if ceiling_db is not None:
        m, sr2 = sf.read(mix)
        m = m * (10 ** (ceiling_db / 20) / np.max(np.abs(m)))
        sf.write(mix, m, sr2)

    # 4) bake (cache key MUST match soulx_freelyrics_demo._cache_key)
    lyric = "/".join(" ".join(l.lower().split()) for l in lines)
    key = hashlib.sha256(f"{SONG}|{lyric}::{args.n_steps}::r0".encode()).hexdigest()[:16]
    cdir = ROOT / "assets" / SONG / "cache"; cdir.mkdir(parents=True, exist_ok=True)
    for old in cdir.glob("fl_*"):
        old.unlink()
    shutil.copy(mix, cdir / f"fl_{key}_mix.wav")
    shutil.copy(voc_trim, cdir / f"fl_{key}_vocal.wav")
    log(f"BAKED {SONG} key={key} (seed {best[0]}, timbre {best[1]:.4f}) -> {cdir}")

    # 5) COMBINED demo-eligibility gate. Two complementary failure modes:
    #   scat (crappy_fragments): render doesn't match the grid (assumes grid is correct).
    #   solo-vs-chorus (validate_lead): the grid itself is corrupt because the chorus is
    #     choral (lead in unison with a choir) -> render matches a garbage grid, so scat
    #     can't see it. Demo-eligible only if scat ENABLE *and* lead SOLO.
    # HIDE is auto-applied (safe); ENABLE is only RECOMMENDED (enabling publishes live).
    import crappy_fragments as cf
    import validate_lead as vl
    r = cf.evaluate(SONG, cdir / f"fl_{key}_vocal.wav")
    lead = vl.evaluate(SONG)
    if r:
        log(f"SCAT-GATE {SONG}: total_crappy={r['total']:.2f}s (gate {cf.GATE_S}s) -> {r['verdict']}")
        if lead:
            log(f"LEAD-GATE {SONG}: backing/total={lead['backing_ratio']*100:.0f}% -> {lead['verdict']}")
        else:
            log("LEAD-GATE: skipped (no acc.wav)")
        scat_ok = r["verdict"] == "ENABLE"
        lead_ok = (lead is None) or (lead["verdict"] == "SOLO")
        cfgp = singer.config_json(SONG)
        conf = json.loads(cfgp.read_text()) if cfgp.exists() else {}
        if scat_ok and lead_ok:
            log(f"  -> RECOMMEND demo=true (confirm before publishing): set \"demo\": true in {cfgp}")
        else:
            why = ([] if scat_ok else [f"scat {r['total']:.1f}s"]) + ([] if lead_ok else ["choral lead"])
            if conf.get("demo"):
                conf["demo"] = False; cfgp.write_text(json.dumps(conf, indent=2))
                log(f"  -> set demo=false ({', '.join(why)})")
            else:
                log(f"  -> demo stays false ({', '.join(why)})")


if __name__ == "__main__":
    main()
