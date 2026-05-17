"""Incremental render + rescore loop.

For each (seed, threshold, phrase) tuple, render the audio (skipping if
the output file already exists — resumable), update the wav2vec2 cache,
and rescore that phrase under the current metric. Logs whether the top
file changed.

Order: outer=seed, middle=threshold, inner=phrase. After every 30
renders (one full sweep), all phrases have been touched at all thresholds
once — so you can stop early and still have balanced new seeds.

Usage:
    python scripts/render_and_score_loop.py >> /tmp/render_loop.log 2>&1 &
    tail -f /tmp/render_loop.log
"""
from __future__ import annotations
import sys, time, re, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import eval_rerank_lang as erl
import soundfile as sf
import scipy.signal as sps

DL = Path("/Users/milhouse/Downloads")
N_STEPS = 16

PHRASES_LIST = [
    ("buenos_dias",    ["bue", "nos", "di", "as"]),
    ("happy_birthday", ["hap", "pee", "birth", "day"]),
    ("buenas_tardes",  ["bue", "nas", "tar", "des"]),
    ("llueve_mucho",   ["llue", "ve", "mu", "cho"]),
    ("mola_mucho",     ["mo", "la", "mu", "cho"]),
    ("buenas_noches",  ["bue", "nas", "no", "ches"]),
    ("mola_mazo",      ["mo", "la", "ma", "zo"]),
    ("muchos_dias",    ["mu", "chos", "di", "as"]),
    ("muchas_tardes",  ["mu", "chas", "tar", "des"]),
    ("hoy_no_llueve",  ["hoy", "no", "llue", "ve"]),
]

SEEDS_NEW = [27, 28, 29, 30, 31, 32, 33]   # new seeds per (phrase, thr) per invocation
THRESHOLDS = [0.20, 0.30, 0.40]
STD = {0.20, 0.30, 0.40}


def output_path(sylls, thr, seed):
    phrase_tag = "_".join(sylls)
    return DL / f"aichael_{phrase_tag}_HYPOTHESIS_thr{int(thr*100):03d}_seed{seed}.wav"


def render(sylls, thr, seed):
    cmd = [sys.executable, str(REPO / "scripts" / "render_seeds.py"),
           f"{thr}"] + sylls + [str(seed)]
    subprocess.run(cmd, check=True, cwd=str(REPO))


def score_phrase(phrase, sylls, proc, mdl):
    lang = erl.PHRASES[phrase]["lang"]
    lang_cfg = erl.LANG_CONFIGS[lang]
    phrase_unique = len(set(sylls))
    prefix = erl.PHRASES[phrase]["prefix"]
    hyp_prefix = f"aichael_{'_'.join(sylls)}_HYPOTHESIS"
    pool = set()
    for pre in {prefix, hyp_prefix}:
        for p in DL.glob(f"{pre}_thr*_seed*.wav"):
            m = re.search(r"_thr(\d+)_seed", p.name)
            if m and int(m.group(1))/100 in STD:
                pool.add((p, int(m.group(1))/100))
    best = None
    for fpath, thr in pool:
        slots = erl.build_slots(sylls, thr)
        audio, sr = sf.read(str(fpath), dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if sr != 16000:
            audio = sps.resample_poly(audio, 16000, sr); sr = 16000
        hyp_per_w = erl._hyp_per_window(proc, mdl, audio, sr, fpath)
        win_len = len(audio) / erl.N_W / sr
        ws = []; unique_seen = set()
        for w in range(erl.N_W):
            ws_s, we_s = w*win_len, (w+1)*win_len
            hyp = hyp_per_w[w]
            exp = erl.expected_syllables_for_window(slots, ws_s, we_s, lang)
            found, _, consumed, names = erl.syllable_completion(
                hyp, exp, lang_cfg, phrase_unique)
            cov = found / phrase_unique
            prec = 0.0 if len(hyp) < 6 else consumed / len(hyp)
            ws.append((cov * prec) ** 0.5)
            unique_seen.update(names)
        # W1-doubled geom (matches eval_rerank_lang).
        geom = (ws[0]*ws[0]*ws[1]*ws[2]*ws[3]) ** 0.2
        pcov = len(unique_seen) / phrase_unique
        new = geom * pcov
        if best is None or new > best[0]:
            best = (new, fpath.name)
    return best


def main():
    print("Loading wav2vec2 ...", flush=True)
    from transformers import AutoProcessor, AutoModelForCTC
    proc = AutoProcessor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    mdl = AutoModelForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    mdl.eval()

    print("Initial tops:", flush=True)
    prev_tops = {}
    for phrase, sylls in PHRASES_LIST:
        top = score_phrase(phrase, sylls, proc, mdl)
        if top:
            prev_tops[phrase] = top[1]
            print(f"  {phrase:<16}  {top[0]:.3f}  {top[1]}", flush=True)

    total = len(SEEDS_NEW) * len(THRESHOLDS) * len(PHRASES_LIST)
    idx = 0
    t_start = time.time()
    for seed in SEEDS_NEW:
        for thr in THRESHOLDS:
            for phrase, sylls in PHRASES_LIST:
                idx += 1
                out = output_path(sylls, thr, seed)
                if out.exists():
                    status = "skip(exists)"
                else:
                    t0 = time.time()
                    try:
                        render(sylls, thr, seed)
                    except subprocess.CalledProcessError as e:
                        print(f"[{idx}/{total}] {phrase:<16} thr={thr:.2f} seed={seed} RENDER FAILED ({e})", flush=True)
                        continue
                    status = f"rendered({time.time()-t0:.0f}s)"
                top = score_phrase(phrase, sylls, proc, mdl)
                if top:
                    score, fname = top
                    changed = "  ← TOP CHANGED" if fname != prev_tops.get(phrase) else ""
                    prev_tops[phrase] = fname
                    elapsed = (time.time() - t_start) / 60.0
                    print(f"[{idx}/{total}] {phrase:<16} thr={thr:.2f} seed={seed} | {status} | top={score:.3f} {fname}{changed}  (elapsed {elapsed:.1f}min)", flush=True)
                else:
                    print(f"[{idx}/{total}] {phrase:<16} thr={thr:.2f} seed={seed} | no scoring result", flush=True)


if __name__ == "__main__":
    main()
