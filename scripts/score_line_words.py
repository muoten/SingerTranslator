"""Word-recognition transcription for a full BJ line render (Thriller-style).

For each sung onset slot in the target JSON, transcribe its audio window with
the espeak wav2vec2 phoneme CTC model and check whether the slot's expected
stressed vowel (and, for the onset consonant, its first phone) shows up. Reports
a per-slot hyp transcript + a matched/total count. Compare across cfg values.

  python scripts/score_line_words.py <vocal.wav> <target.json> [label]
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, soundfile as sf, torch
from transformers import Wav2Vec2FeatureExtractor, AutoModelForCTC
from huggingface_hub import hf_hub_download

MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"

ARPA_VOWEL = {  # ARPABET base -> espeak-IPA vowel glyph(s)
    "AA": "ɑ", "AE": "a", "AH": "ʌ", "AO": "ɔ", "AW": "a", "AY": "a", "EH": "ɛ",
    "ER": "ɚ", "EY": "e", "IH": "ɪ", "IY": "i", "OW": "o", "OY": "ɔ", "UH": "ʊ", "UW": "u",
}


def load_slots(tgt):
    d = json.loads(Path(tgt).read_text())[0]
    durs = [float(x) for x in d["duration"].split()]
    texts, phons, ntypes = d["text"].split(), d["phoneme"].split(), d["note_type"].split()
    slots, t = [], 0.0
    for dur, tx, ph, nt in zip(durs, texts, phons, ntypes):
        if nt != "1" and tx != "<SP>":
            slots.append({"text": tx, "phon": ph, "start": t, "end": t + dur, "mel": nt == "3"})
        t += dur
    return slots


def exp_vowel(phon):
    toks = [t for t in phon.replace("en_", "").split("-") if t]
    vs = [t for t in toks if any(t.startswith(v) for v in ARPA_VOWEL)]
    if not vs:
        return None
    st = [t for t in vs if t.endswith("1")] or vs
    base = "".join(c for c in st[0] if c.isalpha())
    return ARPA_VOWEL.get(base)


def ctc_decode(ids, id2tok):
    out, prev = [], None
    for i in ids:
        if i != prev:
            tok = id2tok.get(i, "")
            if tok and not (tok.startswith("<") and tok.endswith(">")):
                out.append(tok)
        prev = i
    return " ".join(out)


def transcribe(fe, mdl, id2tok, audio, sr):
    if len(audio) < 1600:
        return ""
    inp = fe(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        ids = torch.argmax(mdl(inp.input_values).logits, dim=-1)[0].tolist()
    return ctc_decode(ids, id2tok)


def main():
    voc, tgt = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else Path(voc).stem
    y, sr = sf.read(voc)
    if y.ndim > 1:
        y = y.mean(1)
    if sr != 16000:
        import scipy.signal as sps
        y = sps.resample_poly(y, 16000, sr); sr = 16000
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
    mdl = AutoModelForCTC.from_pretrained(MODEL).eval()
    vocab = json.loads(Path(hf_hub_download(MODEL, "vocab.json")).read_text())
    id2tok = {v: k for k, v in vocab.items()}

    slots = load_slots(tgt)
    matched, rows = 0, []
    for s in slots:
        a, b = int(s["start"] * sr), int(min(s["end"] + 0.06, len(y) / sr) * sr)
        hyp = transcribe(fe, mdl, id2tok, y[a:b], sr)
        ev = exp_vowel(s["phon"])
        ok = ev is not None and ev in hyp.replace(" ", "")
        matched += int(ok)
        rows.append((s["text"], ev or "-", hyp, ok))
    print(f"\n=== {label}  ({matched}/{len(slots)} slots recognized) ===")
    for tx, ev, hyp, ok in rows:
        print(f"  {'OK ' if ok else 'xx '} {tx:<7} vowel[{ev:<2}]  hyp: {hyp}")
    print(f"SCORE {label}: {matched}/{len(slots)}")


if __name__ == "__main__":
    main()
