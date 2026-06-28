"""Load the wav2vec2-espeak phoneme model with the positional-conv weights FIXED.

transformers 4.41+ moved weight_norm to a parametrization, so this checkpoint's
`...pos_conv_embed.conv.weight_g` / `weight_v` no longer map to the model's
`parametrizations.weight.original0` / `original1` — they load as RANDOM, and the
model emits gibberish (drifts to Mandarin tone phonemes) even on clear speech.
Shapes match 1:1, so we copy them back after load. Restores the validated metric.
"""
from __future__ import annotations
import json
import torch
from huggingface_hub import hf_hub_download
# NB: use the bare feature extractor + manual vocab decode — the full
# AutoProcessor pulls in the phoneme tokenizer's espeak backend, which isn't
# installed here. The decode below maps CTC ids -> vocab tokens directly.
from transformers import Wav2Vec2FeatureExtractor, AutoModelForCTC

MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
_PREFIX = "wav2vec2.encoder.pos_conv_embed.conv"


def _ckpt_state():
    try:
        import safetensors.torch as st
        return st.load_file(hf_hub_download(MODEL, "model.safetensors"))
    except Exception:
        return torch.load(hf_hub_download(MODEL, "pytorch_model.bin"), map_location="cpu")


def load(model=MODEL):
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model)
    mdl = AutoModelForCTC.from_pretrained(model).eval()
    sd = _ckpt_state()
    g = sd.get(f"{_PREFIX}.weight_g")
    v = sd.get(f"{_PREFIX}.weight_v")
    pw = mdl.wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight
    with torch.no_grad():
        if g is not None and pw.original0.shape == g.shape:
            pw.original0.copy_(g)
        if v is not None and pw.original1.shape == v.shape:
            pw.original1.copy_(v)
    vocab = json.loads(open(hf_hub_download(model, "vocab.json")).read())
    id2tok = {v_: k for k, v_ in vocab.items()}
    return (fe, id2tok), mdl


def _ctc_decode(ids, id2tok):
    out, prev = [], None
    for i in ids:
        if i != prev:
            tok = id2tok.get(i, "")
            if tok and not (tok.startswith("<") and tok.endswith(">")):
                out.append(tok)
        prev = i
    return " ".join(out)


def transcribe(proc, mdl, chunk, sr=16000):
    fe, id2tok = proc
    if len(chunk) < 1600:
        return ""
    inputs = fe(chunk, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        ids = torch.argmax(mdl(inputs.input_values).logits, dim=-1)[0].tolist()
    return _ctc_decode(ids, id2tok)


if __name__ == "__main__":
    import sys, soundfile as sf
    proc, mdl = load()
    y, sr = sf.read(sys.argv[1])
    if y.ndim > 1:
        y = y.mean(1)
    if sr != 16000:
        import scipy.signal as sps
        y = sps.resample_poly(y, 16000, sr); sr = 16000
    for t in [float(x) for x in (sys.argv[2:] or ["1.5", "3.0", "4.5"])]:
        a, b = int(t * sr), int((t + 1.5) * sr)
        print(f"  t={t}s: {transcribe(proc, mdl, y[a:b], sr)}")
