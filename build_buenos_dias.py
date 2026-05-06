"""Map 'buenos dias' (4 syllables: Bue/nos/di/as) cyclically onto the
MJ-derived chorus metadata. Preserves notes, durations, and rest/melisma
structure — only swaps the lyric and phoneme.

For note_type=3 (tied/melisma), reuse the previous syllable's held form
(vowel-only phoneme) so the consonant doesn't re-articulate.
"""
import json

SRC = "/Users/milhouse/claude-code/SingerTranslator/data/mj_chorus_metadata.json"
DST = "/tmp/mj_chorus_buenos_dias.json"

PHON = {"bue": "en_B-W-EH1", "nos": "en_N-OW1-S", "di": "en_D-IY1", "as": "en_AA1-S"}
HELD = {"bue": "en_EH1",     "nos": "en_OW1",     "di": "en_IY1",   "as": "en_AA1"}
CYCLE = ["bue", "nos", "di", "as"]

with open(SRC) as f:
    meta = json.load(f)
item = meta[0]

note_pitch = item["note_pitch"].split() if isinstance(item["note_pitch"], str) else list(item["note_pitch"])
note_type  = item["note_type"].split()  if isinstance(item["note_type"],  str) else list(item["note_type"])
durations  = item["duration"].split()   if isinstance(item["duration"],   str) else list(item["duration"])

assert len(note_pitch) == len(note_type) == len(durations), "len mismatch"

new_text, new_phon = [], []
new_ntype = list(note_type)
syll_idx = 0
last_syll = None

for i, (pitch, ntype) in enumerate(zip(note_pitch, note_type)):
    pitch, ntype = int(pitch), int(ntype)
    if pitch == 0 or ntype == 1:
        new_text.append("<SP>"); new_phon.append("<SP>"); new_ntype[i] = "1"; last_syll = None
        continue
    if ntype == 3 and last_syll is not None:
        new_text.append(last_syll + "_"); new_phon.append(HELD[last_syll])
        continue
    s = CYCLE[syll_idx % len(CYCLE)]
    new_text.append(s); new_phon.append(PHON[s])
    last_syll = s; syll_idx += 1; new_ntype[i] = "2"

item["text"]      = " ".join(new_text)
item["phoneme"]   = " ".join(new_phon)
item["note_type"] = " ".join(new_ntype)

with open(DST, "w") as f:
    json.dump(meta, f, indent=2)

print(f"Wrote {DST}")
print(f"text: {item['text']}")
print(f"phon: {item['phoneme']}")
print(f"types: {item['note_type']}")
print(f"slots:{len(note_pitch)}  syllables filled:{syll_idx}")
