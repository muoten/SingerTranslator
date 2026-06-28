"""
Melody-following (F0-CORR) + timbre-similarity (SIM) scorer for SingerTranslator
================================================================================
Plugs the two objective gaps in the perceived-phoneme metric: it does NOT measure
whether a take follows the melody, nor whether it still sounds like the target
singer. This adds both, so seed sweeps can be ranked on melody + timbre, not just
intelligibility.

  F0-CORR  Pearson correlation between the take's F0 contour (torchcrepe) and the
           grid's intended melody (note_pitch expanded by duration). Correlation
           is invariant to a constant octave/key offset, so it scores the SHAPE of
           the melody. Also reports semitone-RMSE (after removing median offset)
           and voiced-overlap (how much of the intended melody the take actually
           sang).
  SIM      Cosine similarity of WavLM-SV speaker x-vectors between the take and the
           song's prompt.wav (the timbre SoulX cloned). ~0.85+ = same singer.

Score one take or a whole seed sweep (glob); ranks by a combined score.

Usage:
    # score a folder of seed renders against the BJ grid + prompt
    python scripts/score_melody_timbre.py --song billie_jean \
        --takes 'assets/billie_jean/cache/*.wav'

    # explicit grid + reference voice
    python scripts/score_melody_timbre.py \
        --grid assets/thriller/chorus_target.json \
        --ref  assets/thriller/prompt.wav \
        --takes /tmp/singer_renders/seed_*.wav --json out.json
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

HOP = 160          # 10 ms @ 16 kHz
SR = 16000
FPS = SR / HOP     # 100 fps
PERIODICITY_THR = 0.21   # torchcrepe voiced/unvoiced cutoff


# --------------------------------------------------------------------------- F0
def load_mono16k(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    return wav  # [1, N]


def take_f0_midi(path, model="full"):
    """Per-frame MIDI (NaN where unvoiced) for a rendered take via torchcrepe."""
    import torchcrepe

    audio = load_mono16k(path)
    f0, per = torchcrepe.predict(
        audio, SR, hop_length=HOP, fmin=50.0, fmax=1100.0,
        model=model, return_periodicity=True, device="cpu", batch_size=512,
    )
    f0 = f0[0].numpy()
    per = per[0].numpy()
    midi = 69.0 + 12.0 * np.log2(np.where(f0 > 0, f0, np.nan) / 440.0)
    midi[per < PERIODICITY_THR] = np.nan
    return midi


def grid_target_midi(grid_path):
    """Per-frame intended MIDI (NaN at rests) from the SoulX note grid."""
    it = json.loads(open(grid_path).read())
    it = it[0] if isinstance(it, list) else it
    pit = [int(x) for x in it["note_pitch"].split()]
    typ = [int(x) for x in it["note_type"].split()]
    dur = [float(x) for x in it["duration"].split()]
    frames = []
    for p, t, d in zip(pit, typ, dur):
        n = max(1, int(round(d * FPS)))
        val = float(p) if (t != 1 and p > 0) else np.nan
        frames.extend([val] * n)
    return np.array(frames)


def f0_corr(take_midi, target_midi):
    n = min(len(take_midi), len(target_midi))
    a, b = take_midi[:n], target_midi[:n]
    voiced = ~np.isnan(a) & ~np.isnan(b)
    target_voiced = ~np.isnan(b)
    overlap = voiced.sum() / max(target_voiced.sum(), 1)   # melody coverage
    if voiced.sum() < 5:
        return dict(corr=float("nan"), semi_rmse=float("nan"), voiced_overlap=float(overlap))
    av, bv = a[voiced], b[voiced]
    corr = float(np.corrcoef(av, bv)[0, 1]) if av.std() > 1e-6 and bv.std() > 1e-6 else float("nan")
    off = np.median(av - bv)                                # octave/key tolerant
    semi_rmse = float(np.sqrt(np.mean(((av - off) - bv) ** 2)))
    return dict(corr=corr, semi_rmse=semi_rmse, voiced_overlap=float(overlap))


# --------------------------------------------------------------------------- SIM
_SV = {"fe": None, "model": None}


def _load_sv():
    if _SV["model"] is None:
        from transformers import AutoFeatureExtractor, WavLMForXVector
        name = "microsoft/wavlm-base-plus-sv"
        _SV["fe"] = AutoFeatureExtractor.from_pretrained(name)
        _SV["model"] = WavLMForXVector.from_pretrained(name).eval()
    return _SV["fe"], _SV["model"]


@torch.no_grad()
def xvector(path):
    fe, model = _load_sv()
    wav = load_mono16k(path)[0].numpy()
    inputs = fe([wav], sampling_rate=SR, return_tensors="pt", padding=True)
    emb = model(**inputs).embeddings        # [1, D]
    return F.normalize(emb, dim=-1)


def sim(take_path, ref_emb):
    return float((xvector(take_path) * ref_emb).sum())


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", help="resolves assets/<song>/chorus_target.json + prompt.wav")
    ap.add_argument("--grid", help="explicit note grid json (overrides --song)")
    ap.add_argument("--ref", help="explicit reference voice wav (overrides --song)")
    ap.add_argument("--takes", required=True, nargs="+", help="wav path(s) or glob(s)")
    ap.add_argument("--w_corr", type=float, default=0.6, help="weight of F0-CORR in combined score")
    ap.add_argument("--crepe_model", default="full", choices=["tiny", "full"],
                    help="torchcrepe model ('tiny' = much faster, fine for a correlation metric)")
    ap.add_argument("--json", help="optional path to dump full results")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    grid = args.grid or os.path.join(root, "assets", args.song, "chorus_target.json")
    ref = args.ref or os.path.join(root, "assets", args.song, "prompt.wav")
    assert os.path.exists(grid), f"grid not found: {grid}"
    assert os.path.exists(ref), f"ref voice not found: {ref}"

    takes = []
    for pat in args.takes:
        takes.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat])
    takes = [t for t in takes if os.path.exists(t)]
    assert takes, "no takes found"

    print(f"[INFO] grid={grid}\n[INFO] ref ={ref}\n[INFO] {len(takes)} take(s) "
          f"| crepe={args.crepe_model}\n", flush=True)
    target = grid_target_midi(grid)
    print("[INFO] loading speaker model + ref embedding ...", flush=True)
    ref_emb = xvector(ref)

    rows = []
    for i, t in enumerate(takes, 1):
        print(f"[{i}/{len(takes)}] scoring {os.path.basename(t)} ...", flush=True)
        fc = f0_corr(take_f0_midi(t, model=args.crepe_model), target)
        s = sim(t, ref_emb)
        # combined: F0-CORR (0..1, clamp neg) weighted with SIM (0..1)
        comb = args.w_corr * max(fc["corr"], 0.0) + (1 - args.w_corr) * max(s, 0.0)
        rows.append(dict(take=t, combined=comb, sim=s, **fc))

    rows.sort(key=lambda r: r["combined"], reverse=True)
    print(f"{'take':40s} {'COMB':>6} {'F0-CORR':>8} {'SIM':>6} {'semiRMSE':>9} {'cover':>6}")
    print("-" * 80)
    for r in rows:
        name = os.path.basename(r["take"])[:40]
        print(f"{name:40s} {r['combined']:6.3f} {r['corr']:8.3f} {r['sim']:6.3f} "
              f"{r['semi_rmse']:9.2f} {r['voiced_overlap']*100:5.0f}%")
    print(f"\nBEST: {os.path.basename(rows[0]['take'])}  "
          f"(F0-CORR={rows[0]['corr']:.3f}, SIM={rows[0]['sim']:.3f})")

    if args.json:
        json.dump(rows, open(args.json, "w"), indent=2)
        print(f"[INFO] wrote {args.json}")


if __name__ == "__main__":
    main()
