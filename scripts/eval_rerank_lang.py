"""Language-aware rerank.

Two language configs:
  - "en"     → English (g2p_en + English EQUIV)
  - "es_cas" → Castilian Spanish (manual IPA map + Castilian EQUIV,
               /z/ → /θ/, /v/ → /b/, /h/ silent, /ll/ → /ʎ/)

Per-phrase config selects which language to use. Drop-in compatible
with the singing-aware K=0 adjacency + equal-weight 4-window mean.

Local-only. Doesn't modify singer.py or app.py — those still default
to English. Wiring `lang` through to render is a follow-up.
"""
from __future__ import annotations
import hashlib, json, os, re, sys, tempfile, warnings, pathlib
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
    "ɹ":  {"ɹ", "r", "ɾ"},
    # /ɝ/ stays strict-rhotic. Earlier we accepted /ə/ and /ʌ/ for sung
    # de-rhoticization, but it caused phantom "birth" matches on bare
    # /p ʌ s/ syllables. Singer must actually produce an r-colour for
    # /ɝ/-bearing words ("birth", "her", etc.) to score.
    "ɝ":  {"ɝ", "ɚ", "ɹ"},
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
    "p": {"p", "b"}, "t": {"t", "d", "ts"}, "k": {"k", "g"},
    # Castilian /b/ in singing weakens to bilabial fricative /β/;
    # wav2vec2 often emits ʋ, f, m, or elides
    "b": {"b", "β", "ʋ", "p", "m", "f"},
    "d": {"d", "ð", "t"}, "g": {"g", "ɡ", "ɣ", "k"},
    # /θ/ — Castilian. wav2vec2 transcription bias: the espeak-cv-ft model
    # emits `θ` only 32× across 1620 cached files (~1.7% of files), even
    # when SoulX is asked for `en_TH` and produces audibly good /θ/.
    # Common substitutes wav2vec2 emits at /θ/ positions: `t` (8573 total),
    # `ts` (3029 total). Adding these closes the transcription gap.
    # Phantom risk limited: /θ/ only appears in `zo` (mola_mazo) and `cias`
    # (muchas_gracias) — no other Spanish syllable uses it, so phantoms
    # can only affect those two phrases. Kept `s` out (would conflict
    # with the many /s/-ending syllables) and `f` out (rare substitute).
    "θ": {"θ", "ts", "t"},
    # /h/ → silent. NO equivalents — must be matched via OPTIONAL_PREFIX
    "h": set(),
    # /j/ — yeísmo merger is modern Castilian standard. Spans the full
    # rehilamiento continuum (j → ʝ → ʒ → dʒ) since emphatic/sung
    # delivery commonly affricates. Also /i/ as a vocalic fallback and
    # /ʎ/ for any rural speaker preserving the distinction.
    "j": {"j", "ʝ", "ʎ", "i", "iː", "dʒ", "ʒ"},
    "ɲ": {"ɲ", "n"},
    # Pure 5-vowel system. 2026-05-18: added Latin-letter long forms
    # `aː` and `eː` — espeak-cv-ft emits these for Mandarin-style or
    # English-bias long-vowel realizations that SoulX produces when
    # singing Spanish. Data-driven scan found 1100+ `aː` and 600+ `eː`
    # in windows where /a/- and /e/-bearing syllables were expected
    # but missed. Unambiguously the target vowel quality.
    "a": {"a", "ɑ", "ɑː", "aː", "ʌ"},
    "e": {"e", "ɛ", "eɪ", "eː"},
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
    # 2026-05-18: dropped `n` from m-EQUIV. Phantom-prone — let bare /n V/
    # sequences credit as /m V/ syllables (mo/ma/mu). Observed in mola_mazo
    # seed29 W1 where `n o` was credited as `mo`, inflating cov to 0.75
    # despite no /m/ in the audio. /m/→/n/ assimilation is real but rare
    # enough in sung context that the phantom rate dominates. Bilabial /b/
    # for sung /m/ is retained — well-attested allophone in Castilian.
    "n": {"n", "ŋ", "ɳ", "ɲ", "m"}, "m": {"m", "ŋ", "b"},
    "l": {"l", "ɫ", "ɾ", "ʎ"},
    # tap r. 2026-05-18: added `ʐ` (Mandarin retroflex r). SoulX produces
    # this when reaching for a Spanish tap r; data-driven scan found ~100
    # occurrences across `tar`/`des`/`chas` missed-windows. Phantom risk
    # is low — Spanish doesn't produce `ʐ` natively.
    "ɾ": {"ɾ", "r", "ɹ", "ʐ"},
    # /s/ stays strict-ish. (Note: hyp tokens are normalized to strip
    # trailing dots, so `ts.` from raw output is `ts` by match-time.)
    "s": {"s", "z", "ʃ", "ts"},
    # /tʃ/ — Spanish 'ch'. wav2vec2 commonly emits the fricative /ʃ/
    # alone (t-burst lost in sustained singing). Also tʂ, ʂ, tɕ
    # (alveolopalatal affricate — phonetically same class, more palatalized).
    # 2026-05-16: dropped `ts` — it's an English-leakage token in
    # espeak-cv-ft output (Spanish has no /ts/ phoneme), and combined
    # with the `cho` [tʃ,a]/[tʃ,aʊ]/[tʃ,əɜ] alts it credited phantom
    # `cho` matches in every junk window of mola_mucho seed=9.
    # 2026-05-18: added `dʒ` (voiced palatal affricate). Observed
    # ~3× per window in muchas_gracias renders where chas was expected
    # but missed (`dʒ a s` patterns clearly audible as "chas"). /tʃ/↔/dʒ/
    # is a well-known voicing-allophone in casual/sung speech.
    "tʃ": {"tʃ", "tS", "ʈʂ", "ʃ", "ʂ", "tɕ", "dʒ"},
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
    # buenas tardes
    "nas": ["n", "a", "s"],
    "tar": ["t", "a", "ɾ"],
    "des": ["d", "e", "s"],
    # buenas noches  (bue & nas reused)
    "no":  ["n", "o"],
    "ches":["tʃ", "e", "s"],
    # muchos / muchas
    "chos":["tʃ", "o", "s"],
    "chas":["tʃ", "a", "s"],
    # muchas gracias
    "gra":  ["g", "ɾ", "a"],
    "cias": ["θ", "j", "a", "s"],   # Castilian /θjas/
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
    #   [tʃ, əɜ]  /tʃəɜ/  (ambiguous mid-central vowel — wav2vec2 fallback
    #                      when audio's vowel quality is uncertain; Spanish
    #                      ear maps to /o/ in cho context, 2026-05-16)
    "cho": [["tʃ", "a"], ["tʃ", "aʊ"], ["tʃ", "əɜ"]],
    # mu — bilabial /m/ in sustained singing often weakens to /p/ or /b/.
    # Scoped to "mu" only: adding /p/ to global m-EQUIV would falsely
    # credit /pa/ as "ma" and /po/ as "mo" in mola_mazo.
    "mu":  [["p", "u"], ["b", "u"]],
    # llue — data-driven alternatives from user-labeled audio (2026-05-14).
    # In ear-confirmed "llueve" renders wav2vec2 emits these palatal-onset
    # patterns. The /w/ glide and final /e/ are often inaudible / cut by
    # the 4-second window boundary, but the palatal-vowel onset is the
    # ear-cue Spanish listeners use. Per-syllable scoping: no other
    # current preset contains "llue", so phantom risk is zero.
    "llue": [
        ["j", "ɑ"],    # observed: dʒ ɑ
        ["j", "o"],    # back-rounded variant
        ["j", "iɛ"],   # observed: j iɛ5
        ["j", "iou"],  # observed: j iou2 (×2 in seed=4 thr=0.20)
    ],
    # chos / di — data-driven from muchos_dias w2 labeling (2026-05-16).
    # User heard "muchos dias" multiple times in audio that wav2vec2
    # transcribed as `dʒ aʊ s` and `dʒ oʊ s` (for "chos"), and `t eɪ`
    # for "dia". Spanish ear maps:
    #   /dʒ/ → /tʃ/ (already accepted for llue; structurally consistent)
    #   /eɪ/ → /i/ (English long-e centralizes to Spanish /i/ in singing)
    "chos": [
        ["dʒ", "aʊ", "s"],   # observed at pos 3-5 in w2
        ["dʒ", "oʊ", "s"],   # observed at pos 15-17 in w2
    ],
    "di": [
        ["d", "eɪ"],         # observed: t (d-allo) + eɪ ≈ "di" in Spanish ear
    ],
    # la — SoulX consistently produces a closing front glide after /l/
    # in mola_mucho/mola_mazo renders, transcribed as `l aɪ`. Ear-confirmed
    # as "lai" (closer to English "lie") but user accepts as la for these
    # phrases (2026-05-16). Scoped to la — adding aɪ to global a-EQUIV
    # would phantom-credit "nice"→nas, "my"→ma, "tire"→tar, etc.
    "la":  [["l", "aɪ"]],
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


def syllable_completion(hyp, expected_list, lang_cfg, phrase_unique_count):
    """Unique-type-count recall — counts distinct syllable types matched.

    Returns (unique_types_matched, phrase_unique_count, good_tokens, names).

    `unique_types_matched`: number of distinct target syllables found
        anywhere in hyp (no order, no count of repetitions).
    `phrase_unique_count`: typically 4 — recall denominator.
    `good_tokens`: count of hyp tokens covered by any successful match
        (set union — no double-count when multiple matches overlap).

    Window F1 (computed in score()) = average of:
        recall    = unique_types_matched / phrase_unique_count
        precision = good_tokens / hyp_len
    """
    seen: set[str] = set()
    targets = []
    for name, target in expected_list:
        if name not in seen:
            seen.add(name)
            targets.append((name, target))

    consumed_idx: set[int] = set()
    matched_types: set[str] = set()
    for name, target in targets:
        pos = 0
        while pos < len(hyp):
            nxt, cons = find_syllable(hyp, name, target, pos, lang_cfg)
            if nxt == -1:
                break
            matched_types.add(name)
            consumed_idx.update(range(nxt - cons, nxt))
            pos = nxt

    return len(matched_types), phrase_unique_count, len(consumed_idx), sorted(matched_types)


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


# Per-window wav2vec2 hyp cache.
#
# 2026-05-18: the cache is now self-sufficient — it stores hyp_per_window
# plus audio duration and wav metadata. This means scoring works WITHOUT
# the wav file present, letting us delete bulk wavs while keeping the
# full historical pool's hyp signal for the metric.
#
# Cache filename: <wav_basename>.json (name-based, deterministic).
# Invalidation: cache stores wav mtime_ns+size; if the wav exists and
# its stats differ, the cache is treated as stale and recomputed.
# If the wav is missing (deleted), the cache is trusted as authoritative.
W2V_CACHE_DIR = Path("/tmp/aichael_w2v_cache")
W2V_CACHE_DIR.mkdir(exist_ok=True)


def _w2v_cache_path(wav_path: Path) -> Path:
    return W2V_CACHE_DIR / f"{wav_path.name}.json"


def _load_cached(wav_path: Path):
    """Return cache dict or None. None if missing or stale (wav exists but stats differ)."""
    cache_p = _w2v_cache_path(wav_path)
    if not cache_p.exists():
        return None
    try:
        data = json.loads(cache_p.read_text())
    except Exception:
        return None
    if wav_path.exists():
        st = wav_path.stat()
        if (data.get("mtime_ns") not in (None, st.st_mtime_ns)
            or data.get("size") not in (None, st.st_size)):
            return None
    return data


def _hyp_per_window(proc, mdl, audio, sr, wav_path: Path):
    cached = _load_cached(wav_path)
    if cached is not None and "hyp_per_window" in cached:
        return cached["hyp_per_window"]
    win_len = len(audio) / N_W / sr
    out = []
    for w in range(N_W):
        ws, we = w * win_len, (w + 1) * win_len
        s_idx = int(ws * sr)
        e_idx = int(we * sr) if w < N_W - 1 else len(audio)
        chunk = audio[s_idx:e_idx]
        hyp = normalize_hyp_tokens(w2v_phon(proc, mdl, chunk, sr).split())
        out.append(hyp)
    payload = {"hyp_per_window": out, "duration_sec": len(audio) / sr,
               "wav_name": wav_path.name}
    if wav_path.exists():
        st = wav_path.stat()
        payload["mtime_ns"] = st.st_mtime_ns
        payload["size"] = st.st_size
    _w2v_cache_path(wav_path).write_text(json.dumps(payload))
    return out


def cached_duration(wav_path: Path):
    """Return cached duration_sec, or None if not in cache."""
    cached = _load_cached(wav_path)
    if cached is not None:
        return cached.get("duration_sec")
    return None


def iter_pool_for_phrase(phrase: str, downloads_dir: Path,
                          thresholds: set | None = None):
    """Yield wav Paths for `phrase` from both existing files AND from cache
    entries whose wav has been deleted. Used by scoring code so the pool
    survives bulk wav deletion."""
    cfg = PHRASES[phrase]
    sylls = cfg["syllables"]
    prefixes = {cfg["prefix"], f"aichael_{'_'.join(sylls)}_HYPOTHESIS"}
    thr_pat = re.compile(r"_thr(\d+)_seed(\d+)\.wav$")
    seen = set()
    # Live wavs
    for pre in prefixes:
        for p in sorted(downloads_dir.glob(f"{pre}_thr*_seed*.wav")):
            m = thr_pat.search(p.name)
            if not m: continue
            thr = int(m.group(1)) / 100
            if thresholds is None or thr in thresholds:
                seen.add(p.name)
                yield p, thr
    # Cached-but-deleted wavs
    for cache_p in W2V_CACHE_DIR.glob("aichael_*.json"):
        # cache filename = wav_basename.json
        wav_name = cache_p.name[:-5]  # strip .json
        if wav_name in seen:
            continue
        if not any(wav_name.startswith(pre) for pre in prefixes):
            continue
        m = thr_pat.search(wav_name)
        if not m: continue
        thr = int(m.group(1)) / 100
        if thresholds is None or thr in thresholds:
            yield downloads_dir / wav_name, thr


def score(proc, mdl, wav, slots, lang, phrase_unique_count):
    cfg = LANG_CONFIGS[lang]
    wav_path = Path(wav)
    cached = _load_cached(wav_path)
    # Fast path: hyp + duration both cached → no wav read needed.
    if cached is not None and "hyp_per_window" in cached and "duration_sec" in cached:
        hyp_per_w = cached["hyp_per_window"]
        win_len = cached["duration_sec"] / N_W
    else:
        # Cache miss (or partial) — must read the wav.
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if sr != 16000:
            import scipy.signal as sps
            audio = sps.resample_poly(audio, 16000, sr); sr = 16000
        win_len = len(audio) / N_W / sr
        hyp_per_w = _hyp_per_window(proc, mdl, audio, sr, wav_path)
    win_scores, win_details = [], []
    for w in range(N_W):
        ws, we = w * win_len, (w + 1) * win_len
        hyp = hyp_per_w[w]
        exp_sylls = expected_syllables_for_window(slots, ws, we, lang)
        found, total, consumed, names = syllable_completion(hyp, exp_sylls, cfg, phrase_unique_count)
        # syllable_coverage: unique phrase-syllable types detected in window
        # divided by phrase unique syllable count (=4). NOT classical recall:
        # the denominator is the phrase's type inventory, not the number of
        # expected slot instances in the window (which varies by threshold).
        # Effect: a window with 5 slots and 2 distinct types matched scores
        # 0.50, regardless of how many times each type was supposed to repeat.
        syllable_coverage = found / max(total, 1)
        # syllable_precision: fraction of hyp tokens consumed by valid matches.
        # Denominator = hyp_len (raw). Silent-window guard: if a window has
        # fewer than 6 phonemes, the singer effectively didn't sing in that
        # window — set prec to 0 so the geom aggregate collapses. This catches
        # the "habe birthday" failure mode where a 4-token W3 with 4 matches
        # scored 1.0 and dominated the file's geom mean (2026-05-16).
        if len(hyp) < 6:
            syllable_precision = 0.0
        else:
            syllable_precision = consumed / len(hyp)
        f1 = (syllable_coverage * syllable_precision) ** 0.5
        win_scores.append(f1)
        win_details.append({
            "found": found, "total": total, "consumed": consumed,
            "hyp_len": len(hyp), "names": names,
            "syllable_coverage": syllable_coverage,
            "syllable_precision": syllable_precision,
            "f1": f1, "hyp": " ".join(hyp),
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
    "buenas_tardes":   {"lang": "es_cas", "syllables": ["bue","nas","tar","des"],
                        "prefix": "aichael_bue_nas_tar_des_HYPOTHESIS"},
    "buenas_noches":   {"lang": "es_cas", "syllables": ["bue","nas","no","ches"],
                        "prefix": "aichael_bue_nas_no_ches_HYPOTHESIS"},
    "muchos_dias":     {"lang": "es_cas", "syllables": ["mu","chos","di","as"],
                        "prefix": "aichael_mu_chos_di_as_HYPOTHESIS"},
    "muchas_tardes":   {"lang": "es_cas", "syllables": ["mu","chas","tar","des"],
                        "prefix": "aichael_mu_chas_tar_des_HYPOTHESIS"},
    "mola_mucho":      {"lang": "es_cas", "syllables": ["mo","la","mu","cho"],
                        "prefix": "aichael_mo_la_mu_cho_HYPOTHESIS"},
    "hoy_no_llueve":   {"lang": "es_cas", "syllables": ["hoy","no","llue","ve"],
                        "prefix": "aichael_hoy_no_llue_ve_HYPOTHESIS"},
    "muchas_gracias":  {"lang": "es_cas", "syllables": ["mu","chas","gra","cias"],
                        "prefix": "aichael_mu_chas_gra_cias_HYPOTHESIS"},
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
    # Build slot_map covering all melisma thresholds we use for sweeps.
    # Each threshold's slot timings differ — using the wrong slot_map would
    # mis-align expected syllables to the wrong window times.
    slot_map = {t: build_slots(syllables, t)
                for t in (0.20, 0.25, 0.30, 0.35, 0.40)}

    # Accept files matching the primary prefix OR a HYPOTHESIS variant
    # (the latter is what render_seeds.py emits by default; the former is
    # the older naming for buenos_dias / happy_birthday).
    phrase_tag = "_".join(syllables)
    hyp_prefix = f"aichael_{phrase_tag}_HYPOTHESIS"
    prefixes_to_scan = list({prefix, hyp_prefix})  # dedupe if same

    files = []
    for scan_prefix in prefixes_to_scan:
        pattern = re.compile(rf"{re.escape(scan_prefix)}_thr(\d+)_seed(\d+)\.wav")
        for path in sorted(DL.glob(f"{scan_prefix}_thr*_seed*.wav")):
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
    phrase_unique_count = len(set(syllables))
    for label, path, slots, thr, seed in files:
        ws, dets = score(proc, mdl, path, slots, lang, phrase_unique_count)
        w1, mean = ws[0], sum(ws) / N_W
        # geom-mean across windows with W1 doubled. W1 is systematically the
        # hardest window (SoulX cold-start: no left context, less warm-up).
        # Doubling it in the geom rewards files that overcame the cold start.
        # Mathematically: (w1·w1·w2·w3·w4)^(1/5) — w1 counted twice.
        # Any dead window still collapses the score to 0.
        geom = (ws[0] * ws[0] * ws[1] * ws[2] * ws[3]) ** 0.2
        # 2026-05-16: phrase_coverage multiplier. Computes "how many of the
        # phrase's 4 unique syllables were detected anywhere in the file,"
        # capping the score by phrase-level recall. Catches files where one
        # window has lots of matches of the same 2 syllables repeatedly but
        # other syllables are never produced (mola_mazo: ma+la heard 4× each,
        # mo and zo never appear → 2/4 = 0.50 multiplier).
        unique_seen = set()
        for d in dets:
            unique_seen.update(d["names"])
        phrase_coverage = len(unique_seen) / phrase_unique_count
        new = geom * phrase_coverage
        results.append((label, path.name, w1, mean, new, ws, dets, phrase_coverage))
    results.sort(key=lambda r: -r[4])

    # Persist per-file scores so a composite-builder can pick best-per-window
    # without re-running wav2vec2. Output schema is intentionally minimal.
    jsonl_path = Path(f"/tmp/aichael_{phrase_name}_results.jsonl")
    with jsonl_path.open("w") as f:
        for label, fname, w1, mean, new, ws, dets, pcov in results:
            f.write(json.dumps({
                "label": label, "fname": fname,
                "w1": w1, "mean": mean, "new": new,
                "win_f1": ws, "phrase_coverage": pcov,
            }) + "\n")
    print(f"(scores written to {jsonl_path})")

    print(f"\n{'rank':>4}  {'new':>6s}  {'pcov':>4s}  {'w1':>5s}  {'w2':>5s}  {'w3':>5s}  {'w4':>5s}   {'label':32s}   per-window (cov/prec→F1)")
    for i, (label, fname, w1, mean, new, ws, dets, pcov) in enumerate(results[:20], 1):
        det = "  ".join(f"{dets[j]['syllable_coverage']:.2f}/{dets[j]['syllable_precision']:.2f}→{dets[j]['f1']:.2f}"
                       for j in range(N_W))
        mark = " ⭐" if i == 1 else ""
        print(f"  #{i:>2}  {new:.3f}  {pcov:.2f}  {ws[0]:.2f}   {ws[1]:.2f}   {ws[2]:.2f}   {ws[3]:.2f}    {label:32s}   {det}{mark}")
    print(f"\n(Top 20 of {len(results)} total)")

    # Top-per-threshold helper. Prevents the threshold-bias trap (metric
    # systematically over-rewards thr=0.20 via cycling-melisma cycles, so
    # the global top can mask better-quality candidates at thr=0.30/0.40).
    label_thr_re = re.compile(r"SINGLE\s+thr=(\d+\.\d+)\s+seed=")
    by_thr: dict[float, tuple] = {}
    for rank0, (label, fname, w1, mean, new, ws, dets, pcov) in enumerate(results):
        m = label_thr_re.match(label)
        if not m: continue
        thr = float(m.group(1))
        if thr not in by_thr:
            by_thr[thr] = (rank0 + 1, label, fname, w1, mean, new, ws, dets, pcov)
    if by_thr:
        print(f"\n--- top SINGLE at each threshold (catches threshold-bias) ---")
        for thr in sorted(by_thr.keys()):
            rank, label, fname, _, _, new, ws, _, _ = by_thr[thr]
            print(f"  thr={thr:.2f}  rank=#{rank:<3d}  new={new:.3f}  "
                  f"w1={ws[0]:.2f}  w2={ws[1]:.2f}  w3={ws[2]:.2f}  w4={ws[3]:.2f}  "
                  f"{fname}")


if __name__ == "__main__":
    phrase = sys.argv[1] if len(sys.argv) > 1 else "mola_mazo"
    main(phrase)
