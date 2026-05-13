"""Language-aware rerank.

Two language configs:
  - "en"     → English (g2p_en + English EQUIV)
  - "es_cas" → Castilian Spanish (manual IPA map + Castilian EQUIV,
               /z/ → /θ/, /v/ → /b/, /h/ silent, /ll/ → /ʎ/)

Per-phrase config selects which language to use. Drop-in compatible
with the singing-aware K=0 adjacency + 0.3·w1 + 0.7·mean F1 framework.

Local-only. Doesn't modify singer.py or app.py — those still default
to English. Wiring `lang` through to render is a follow-up.
"""
from __future__ import annotations
import json, os, re, sys, tempfile, warnings, pathlib
warnings.filterwarnings("ignore")
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/lib/libespeak-ng.dylib")
sys.path.insert(0, "/Users/milhouse/claude-code/SingerTranslator")
import singer

from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForCTC
from g2p_en import G2p

N_W = 4
K_SKIP = 0
DL = Path("/Users/milhouse/Downloads")

# ---------- shared IPA helpers ----------

ARPABET_TO_IPA = {
    "AA":"ɑ","AE":"æ","AH":"ʌ","AO":"ɔ","AW":"aʊ","AY":"aɪ",
    "EH":"ɛ","ER":"ɝ","EY":"eɪ","IH":"ɪ","IY":"i",
    "OW":"oʊ","OY":"ɔɪ","UH":"ʊ","UW":"u",
    "B":"b","CH":"tʃ","D":"d","DH":"ð","F":"f","G":"ɡ",
    "HH":"h","JH":"dʒ","K":"k","L":"l","M":"m","N":"n",
    "NG":"ŋ","P":"p","R":"ɹ","S":"s","SH":"ʃ","T":"t",
    "TH":"θ","V":"v","W":"w","Y":"j","Z":"z","ZH":"ʒ",
}

# ---------- English config ----------

ENGLISH_EQUIV = {
    # voicing fuzz on stops (medial/sung)
    "p": {"p", "b"}, "t": {"t", "d"}, "k": {"k", "g"},
    "b": {"b", "p"}, "d": {"d", "t"}, "g": {"g", "k"},
    # word-initial /h/ → fricative/silence/burst in singing
    "h": {"h", "k", "x", "ʔ", "ʕ"},
    # vowels & diphthongs
    "ɑ":  {"ɑ", "a", "ɑː"}, "a": {"ɑ", "a", "ɑː", "æ"},
    "æ":  {"æ", "a", "ɑ", "ɛ"},
    "i":  {"i", "iː"}, "u":  {"u", "uː"},
    "oʊ": {"oʊ", "o", "ou", "oː"},
    "aʊ": {"aʊ", "au", "ɑʊ"},
    "eɪ": {"eɪ", "e", "ei", "eː"},
    "ɛ":  {"ɛ", "e", "ɪ"}, "ɪ":  {"ɪ", "i"},
    "ɹ":  {"ɹ", "r", "ɾ"}, "ɝ": {"ɝ", "ɚ", "ɹ", "ə", "ʌ"},
    "θ":  {"θ", "s", "f", "ð"},
}

# ---------- Castilian Spanish config ----------

# Castilian features the metric needs to model:
# - /z/ pronounced /θ/ (not /s/, not /z/)
# - /v/ pronounced /b/ (no /v/ phoneme exists)
# - word-initial /h/ silent (orthographic only)
# - /ll/ → /ʎ/ (or /j/ via yeísmo)
# - pure 5-vowel system /a e i o u/, no diphthongs
# - /r/ → /ɾ/ (tap) or /r/ (trill); never /ɹ/
CASTILIAN_EQUIV = {
    "p": {"p", "b"}, "t": {"t", "d"}, "k": {"k", "g"},
    # Castilian /b/ in singing weakens to bilabial fricative /β/;
    # wav2vec2 often emits ʋ, f, m, or elides
    "b": {"b", "β", "ʋ", "p", "m", "f"},
    "d": {"d", "ð", "t"}, "g": {"g", "ɣ", "k"},
    # /θ/ — strict Castilian. No /s/ fallback (would credit any stray
    # /s/ as a `zo` syllable). True seseo renders will fail the metric;
    # that's correct — we're checking Castilian fidelity here.
    "θ": {"θ"},
    # /h/ → silent. NO equivalents — must be matched via OPTIONAL_PREFIX
    "h": set(),
    # /j/ — yeísmo merger is modern Castilian standard. Spans the full
    # rehilamiento continuum (j → ʝ → ʒ → dʒ) since emphatic/sung
    # delivery commonly affricates. Also /i/ as a vocalic fallback and
    # /ʎ/ for any rural speaker preserving the distinction.
    "j": {"j", "ʝ", "ʎ", "i", "iː", "dʒ", "ʒ"},
    "ɲ": {"ɲ", "n"},
    # Pure 5-vowel system
    "a": {"a", "ɑ", "ɑː", "ʌ"},
    "e": {"e", "ɛ", "eɪ"},
    "i": {"i", "iː", "ɪ"},
    "o": {"o", "oʊ", "ou", "oː", "ɔ"},
    "u": {"u", "uː", "ʊ", "w"},
    # /w/ — labiovelar glide. Accept canonical /w/ and its vocalic
    # realizations (u, uː, ʊ) and the labiodental approximant ʋ.
    # Reject ə (schwa is too neutral — would let "lle" pass for "llue"),
    # oʊ/o/ɔ (pure back vowels — collapse /ue/ to /o/ degrading quality),
    # v (labiodental fricative — too rare in Spanish to matter).
    "w": {"w", "u", "uː", "ʊ", "ʋ"},
    # nasals
    "n": {"n", "ŋ", "ɳ", "ɲ", "m"}, "m": {"m", "n", "ŋ", "b"},
    "l": {"l", "ɫ", "ɾ", "ʎ"},
    # tap r
    "ɾ": {"ɾ", "r", "ɹ"},
    # /s/ stays strict-ish. (Note: hyp tokens are normalized to strip
    # trailing dots, so `ts.` from raw output is `ts` by match-time.)
    "s": {"s", "z", "ʃ", "ts"},
    # /tʃ/ — Spanish 'ch'. wav2vec2 commonly emits the fricative /ʃ/
    # alone (t-burst lost in sustained singing). Also tʂ, ʂ. Less
    # phantom-prone than /s/ since /ʃ/ is rare in Spanish flow.
    "tʃ": {"tʃ", "tS", "ʈʂ", "ʃ", "ʂ", "ts"},
}

# Spanish syllable → IPA tokens (curated for current presets).
# Add syllables here as new phrases are introduced.
SPANISH_SYLLABLES = {
    # buenos dias
    "bue": ["b", "w", "e"],
    "nos": ["n", "o", "s"],
    "di":  ["d", "i"],
    "as":  ["a", "s"],
    # mola mazo
    "mo":  ["m", "o"],
    "la":  ["l", "a"],
    "ma":  ["m", "a"],
    "zo":  ["θ", "o"],          # Castilian z = /θ/
    # hola
    "ho":  ["o"],               # h silent
    "ha":  ["a"],
    # hoy no llueve
    "hoy": ["o", "j"],
    "no":  ["n", "o"],
    "llue": ["j", "w", "e"],    # yeísmo — modern Castilian standard
    "ve":  ["b", "e"],          # v → b
    # hola laura
    "lau": ["l", "a", "w"],
    "ra":  ["ɾ", "a"],
    # dias buenos = reuse di, as, bue, nos
    # llueve mucho
    "mu":  ["m", "u"],
    "cho": ["tʃ", "o"],
}

# Per-syllable alternative IPA targets. Tried only when the primary
# target fails. Justified per-syllable rather than via global EQUIV
# relaxation so the spillover stays local.
#
# Example: "cho" — the word-final unstressed /o/ in "mucho" centralizes
# toward /a/ in sung Spanish. We accept /tʃ a/ as an alternative without
# letting /a/ ↔ /o/ everywhere (which would break mola_mazo's "mo", etc.).
SPANISH_SYLLABLE_ALTERNATIVES: dict[str, list[list[str]]] = {
    # cho — word-final unstressed /o/ centralizes in sung Spanish:
    #   [tʃ, a]   /tʃa/   ("mucha" form)
    #   [tʃ, aʊ]  /tʃaʊ/  (ɑu offglide reads as /o/ to a Spanish ear)
    "cho": [["tʃ", "a"], ["tʃ", "aʊ"]],
    # mu — bilabial /m/ in sustained singing often weakens to /p/ or /b/.
    # Scoped to "mu" only: adding /p/ to global m-EQUIV would falsely
    # credit /pa/ as "ma" and /po/ as "mo" in mola_mazo.
    "mu":  [["p", "u"], ["b", "u"]],
}

LANG_CONFIGS = {
    "en": {
        "EQUIV": ENGLISH_EQUIV,
        # No optional-prefix syllables for English. "hap" used to be here
        # (sung /h/ often elides), but it produced phantom matches: bare
        # /a/ vowels would credit as "hap". If the model omits /h/ entirely,
        # the audio isn't singing "hap" — better to require some onset.
        "OPTIONAL_PREFIX": set(),
        "syllable_to_ipa": "english_g2p",
    },
    "es_cas": {
        "EQUIV": CASTILIAN_EQUIV,
        "OPTIONAL_PREFIX": {"bue", "ho", "hoy", "hola", "ha"},  # h-silent / weak-b
        "syllable_to_ipa": "spanish_map",
        "ALTERNATIVES": SPANISH_SYLLABLE_ALTERNATIVES,
    },
}

# ---------- syllable → IPA dispatch ----------

_G2P = G2p()
_IPA_CACHE: dict[tuple[str, str], list[str]] = {}


def phon2ipa(s):
    """ARPABET 'en_X-Y-Z' (singer.SYLLABLE_OVERRIDES style) → list of IPA tokens."""
    if s.startswith("en_"): s = s[3:]
    out = []
    for p in s.split("-"):
        if not p: continue
        if p[-1].isdigit(): p = p[:-1]
        out.append(ARPABET_TO_IPA.get(p, p))
    return out


def english_g2p(syllable):
    k = syllable.lower()
    if k in singer.SYLLABLE_OVERRIDES:
        return phon2ipa(singer.SYLLABLE_OVERRIDES[k])
    raw = []
    for tok in _G2P(syllable):
        tok = tok.strip()
        if not tok or tok in (",", ".", "!", "?", ";", ":", "'"): continue
        if tok[-1].isdigit(): tok = tok[:-1]
        raw.append(ARPABET_TO_IPA.get(tok, tok))
    return raw


def spanish_map(syllable):
    k = syllable.lower()
    if k not in SPANISH_SYLLABLES:
        raise KeyError(f"Spanish syllable {k!r} not in SPANISH_SYLLABLES — add it.")
    return list(SPANISH_SYLLABLES[k])


_DISPATCH = {"english_g2p": english_g2p, "spanish_map": spanish_map}


def expected_ipa_dedup(syl, lang):
    cache_key = (lang, syl.lower())
    if cache_key in _IPA_CACHE: return _IPA_CACHE[cache_key]
    fn = _DISPATCH[LANG_CONFIGS[lang]["syllable_to_ipa"]]
    raw = fn(syl)
    out = []
    for p in raw:
        if not out or out[-1] != p: out.append(p)
    _IPA_CACHE[cache_key] = out
    return out


# ---------- match logic ----------

def tokens_match(hyp_tok, target_tok, equiv):
    if hyp_tok == target_tok: return True
    if target_tok in equiv and hyp_tok in equiv[target_tok]: return True
    return False


def find_sequence(hyp, target, start_pos, equiv, k=K_SKIP):
    if not target: return start_pos, 0
    i = start_pos
    while i < len(hyp):
        ti = i; match = True
        for tp in target:
            found_at = -1
            for j in range(ti, min(ti + k + 1, len(hyp))):
                if tokens_match(hyp[j], tp, equiv):
                    found_at = j; break
            if found_at == -1:
                match = False; break
            ti = found_at + 1
        if match:
            return ti, ti - i
        i += 1
    return -1, 0


def find_syllable(hyp, syl_name, target, start_pos, lang_cfg):
    r, cons = find_sequence(hyp, target, start_pos, lang_cfg["EQUIV"])
    if r != -1: return r, cons
    # Try per-syllable alternative targets (e.g. cho → [tʃ, a]).
    for alt in lang_cfg.get("ALTERNATIVES", {}).get(syl_name.lower(), []):
        r, cons = find_sequence(hyp, alt, start_pos, lang_cfg["EQUIV"])
        if r != -1: return r, cons
    if syl_name.lower() in lang_cfg["OPTIONAL_PREFIX"] and len(target) > 1:
        return find_sequence(hyp, target[1:], start_pos, lang_cfg["EQUIV"])
    return -1, 0


def build_slots(syllables, threshold):
    out = pathlib.Path(tempfile.mktemp(suffix=".json"))
    singer.DEFAULT_AUTO_MELISMA_DUR = threshold
    singer.build_target_metadata(syllables, out, melisma_mode="default")
    t = json.loads(out.read_text())[0]
    def _split(v): return v.split() if isinstance(v, str) else list(v)
    durs = [float(d) for d in _split(t["duration"])]
    types = [int(x) for x in _split(t["note_type"])]
    texts = _split(t["text"])
    slots, cur = [], 0.0
    for d, ty, tx in zip(durs, types, texts):
        slots.append({"start": cur, "end": cur + d, "type": ty, "text": tx})
        cur += d
    return slots


def expected_syllables_for_window(slots, w_start, w_end, lang):
    out = []
    for s in slots:
        if s["type"] != 2: continue
        if w_start <= s["start"] < w_end:
            out.append((s["text"], expected_ipa_dedup(s["text"], lang)))
    return out


def syllable_completion(hyp, expected_list, lang_cfg):
    """Greedy in-order match. Recall credits the LONGEST CONTIGUOUS RUN in
    UNIQUE-TYPE space.

    Rationale: matches that cover {ve, cho} but skip mu indicate the audio
    rendered the wrong sub-phrase. Matches that cover {mu, cho} are the
    correct end-of-phrase sub-sequence. Cycling melisma where one type
    appears multiple times (e.g. ve, ve, ve) is NOT penalized — repetition
    of a single unique type is still a length-1 contiguous run.
    """
    pos = 0
    per_syl = []  # (expected_idx, consumed_tokens, name)
    for idx, (name, target) in enumerate(expected_list):
        nxt, cons = find_syllable(hyp, name, target, pos, lang_cfg)
        if nxt != -1:
            per_syl.append((idx, cons, name))
            pos = nxt

    # Map each unique syllable name to a sequence index by first appearance.
    type_to_idx: dict[str, int] = {}
    for name, _ in expected_list:
        if name not in type_to_idx:
            type_to_idx[name] = len(type_to_idx)
    total_unique = len(type_to_idx)

    if not per_syl:
        return 0, total_unique, 0, []

    # Sorted distinct unique-type indices touched by matches.
    matched_uniq = sorted({type_to_idx[name] for _, _, name in per_syl})

    # Longest contiguous run in sorted distinct unique-type indices.
    best_len = 0; cur_len = 0; prev = -2
    for i in matched_uniq:
        cur_len = cur_len + 1 if i == prev + 1 else 1
        if cur_len > best_len:
            best_len = cur_len
        prev = i

    consumed = sum(cons for _, cons, _ in per_syl)  # precision counts all matches
    names = [name for _, _, name in per_syl]
    return best_len, total_unique, consumed, names


_STRIP_TRAILING = __import__("re").compile(r"[\d.]+$")

# Map ASCII diphthong spellings (after stress/dot stripping) onto their
# canonical IPA forms so existing EQUIV entries continue to match.
# wav2vec2-phoneme tokenizes "ai5", "ei5", "ou5", etc. as the closest
# IPA contour without the U+026A/U+028A small-cap markers.
_ASCII_TO_IPA = {
    "ai": "aɪ",
    "ei": "eɪ",
    "ou": "oʊ",
    "au": "aʊ",
    "oi": "ɔɪ",
    # Half-IPA half-Latin forms the espeak-cv-ft tokenizer emits for back
    # diphthongs (script ɑ + Latin u/i). Canonicalize to standard IPA so
    # EQUIV entries match.
    "ɑu": "aʊ",
    "ɑi": "aɪ",
}


def normalize_hyp_tokens(tokens):
    """Strip stress digits / dot markers and canonicalize ASCII diphthongs."""
    out = []
    for t in tokens:
        t = _STRIP_TRAILING.sub("", t)
        out.append(_ASCII_TO_IPA.get(t, t))
    return out


def w2v_phon(proc, mdl, chunk, sr=16000):
    if len(chunk) < 1600: return ""
    inputs = proc(chunk, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = mdl(inputs.input_values).logits
    ids = torch.argmax(logits, dim=-1)[0].tolist()
    return proc.batch_decode([ids])[0].strip()


def score(proc, mdl, wav, slots, lang):
    cfg = LANG_CONFIGS[lang]
    audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != 16000:
        import scipy.signal as sps
        audio = sps.resample_poly(audio, 16000, sr); sr = 16000
    win_len = len(audio) / N_W / sr
    win_scores, win_details = [], []
    for w in range(N_W):
        ws, we = w * win_len, (w + 1) * win_len
        s_idx, e_idx = int(ws * sr), int(we * sr) if w < N_W - 1 else len(audio)
        chunk = audio[s_idx:e_idx]
        hyp = normalize_hyp_tokens(w2v_phon(proc, mdl, chunk, sr).split())
        exp_sylls = expected_syllables_for_window(slots, ws, we, lang)
        found, total, consumed, names = syllable_completion(hyp, exp_sylls, cfg)
        recall = found / max(total, 1)
        precision = consumed / max(len(hyp), 1)
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
        win_scores.append(f1)
        win_details.append({
            "found": found, "total": total, "consumed": consumed,
            "hyp_len": len(hyp), "names": names,
            "recall": recall, "precision": precision, "f1": f1,
            "hyp": " ".join(hyp),
        })
    return win_scores, win_details


# ---------- phrase registry ----------

PHRASES = {
    "happy_birthday":  {"lang": "en",     "syllables": ["hap","pee","birth","day"],
                        "prefix": "aichael_happy_birthday"},
    "buenos_dias":     {"lang": "es_cas", "syllables": ["bue","nos","di","as"],
                        "prefix": "aichael_buenos_dias"},
    "mola_mazo":       {"lang": "es_cas", "syllables": ["mo","la","ma","zo"],
                        "prefix": "aichael_mola_mazo_A2"},
    "llueve_mucho":    {"lang": "es_cas", "syllables": ["llue","ve","mu","cho"],
                        # NEW prefix: Spanish-override hypothesis renders.
                        # Old `aichael_llueve_mucho_thr*_seed*.wav` pool was
                        # built with broken English phonemes — excluded.
                        "prefix": "aichael_llue_ve_mu_cho_HYPOTHESIS"},
}


def main(phrase_name):
    cfg = PHRASES[phrase_name]
    lang = cfg["lang"]
    syllables = cfg["syllables"]
    prefix = cfg["prefix"]

    print(f"=== Rerank: {phrase_name}  lang={lang}  syllables={syllables}", flush=True)
    print(f"    expected IPA per syllable: " +
          ", ".join(f"{s}={expected_ipa_dedup(s, lang)}" for s in syllables))
    print()

    print("Loading wav2vec2-phoneme ...", flush=True)
    proc = AutoProcessor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    mdl = AutoModelForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    mdl.eval()
    slot_map = {0.20: build_slots(syllables, 0.20),
                0.30: build_slots(syllables, 0.30),
                0.40: build_slots(syllables, 0.40)}

    files = []
    pattern = re.compile(rf"{re.escape(prefix)}_thr(\d+)_seed(\d+)\.wav")
    for path in sorted(DL.glob(f"{prefix}_thr*_seed*.wav")):
        m = pattern.match(path.name)
        if m:
            thr, seed = int(m.group(1))/100, int(m.group(2))
            files.append((f"SINGLE thr={thr:.2f} seed={seed:>3d}", path,
                          slot_map.get(thr, slot_map[0.20]), thr, seed))
    for tag, p in [("v2_xf",  DL / f"{prefix}_composite_v2_xf.wav"),
                   ("lev_xf", DL / f"{prefix}_composite_lev_xf.wav"),
                   ("ALT_xf", DL / f"{prefix}_composite_ALT_xf.wav")]:
        if p.exists():
            files.append((f"COMP   {tag:24s}", p, slot_map[0.20], None, None))

    results = []
    for label, path, slots, thr, seed in files:
        ws, dets = score(proc, mdl, path, slots, lang)
        w1, mean = ws[0], sum(ws) / N_W
        new = 0.3 * w1 + 0.7 * mean
        results.append((label, path.name, w1, mean, new, ws, dets))
    results.sort(key=lambda r: -r[4])

    # Persist per-file scores so a composite-builder can pick best-per-window
    # without re-running wav2vec2. Output schema is intentionally minimal.
    jsonl_path = Path(f"/tmp/aichael_{phrase_name}_results.jsonl")
    with jsonl_path.open("w") as f:
        for label, fname, w1, mean, new, ws, dets in results:
            f.write(json.dumps({
                "label": label, "fname": fname,
                "w1": w1, "mean": mean, "new": new,
                "win_f1": ws,
            }) + "\n")
    print(f"(scores written to {jsonl_path})")

    print(f"\n{'rank':>4}  {'new':>6s}  {'w1':>5s}  {'w2':>5s}  {'w3':>5s}  {'w4':>5s}   {'label':32s}   per-window (R/P→F1)")
    for i, (label, fname, w1, mean, new, ws, dets) in enumerate(results[:20], 1):
        det = "  ".join(f"{dets[j]['recall']:.2f}/{dets[j]['precision']:.2f}→{dets[j]['f1']:.2f}"
                       for j in range(N_W))
        mark = " ⭐" if i == 1 else ""
        print(f"  #{i:>2}  {new:.3f}   {ws[0]:.2f}   {ws[1]:.2f}   {ws[2]:.2f}   {ws[3]:.2f}    {label:32s}   {det}{mark}")
    print(f"\n(Top 20 of {len(results)} total)")


if __name__ == "__main__":
    phrase = sys.argv[1] if len(sys.argv) > 1 else "mola_mazo"
    main(phrase)
