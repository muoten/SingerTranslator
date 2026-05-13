"""Dump per-window wav2vec2-phoneme hyps for a given file."""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.dylib")

from pathlib import Path
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForCTC

N_W = 4

proc = AutoProcessor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
mdl = AutoModelForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
mdl.eval()

for fpath in sys.argv[1:]:
    audio, sr = sf.read(fpath, dtype="float32")
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != 16000:
        import scipy.signal as sps
        audio = sps.resample_poly(audio, 16000, sr); sr = 16000
    win_len = len(audio) / N_W / sr
    print(f"\n=== {Path(fpath).name} ===")
    for w in range(N_W):
        ws, we = w * win_len, (w + 1) * win_len
        s_idx, e_idx = int(ws * sr), int(we * sr) if w < N_W - 1 else len(audio)
        chunk = audio[s_idx:e_idx]
        if len(chunk) < 1600:
            print(f"  w{w+1} ({ws:.1f}-{we:.1f}s): <too short>"); continue
        inp = proc(chunk, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = mdl(inp.input_values).logits
        ids = torch.argmax(logits, dim=-1)[0].tolist()
        hyp = proc.batch_decode([ids])[0].strip()
        print(f"  w{w+1} ({ws:.1f}-{we:.1f}s): {hyp}")
