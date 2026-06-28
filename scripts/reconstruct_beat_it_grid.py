"""Reconstruct the buried-syllable notes in the Beat It chorus grid.

The roformer-isolated vocal has no extractable pitch at a handful of syllables
(lead buried under backing/instrumental), so ROSVOT left them as pitch-0 rests
and "defeated" got no notes at all. For the COVER use case we want a slot per
original syllable, so we promote those word-slots to real notes:

  - measured where possible (slot 26 'wrong' had 50% voiced f0 -> MIDI 68)
  - otherwise ESTIMATED from the local melodic contour (clearly not measured)

note_pitch / note_type / duration / text / phoneme are edited, AND the frame f0
track is painted at the reconstructed notes (they sit on unvoiced f0, which SoulX
renders as silence). Total duration (16s) is preserved — the 'defeated' notes are
made by SPLITTING the existing rest, not adding time.

Run: python3 scripts/reconstruct_beat_it_grid.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TGT = ROOT / "assets/beat_it/chorus_target.json"
BAK = ROOT / "assets/beat_it/chorus_target.roformer_raw.json"

# Always rebuild from the measured-only grid so this script is idempotent.
SRC = BAK if BAK.exists() else TGT
g = json.loads(SRC.read_text())
item = g[0]
pit = [int(x) for x in item["note_pitch"].split()]
typ = [int(x) for x in item["note_type"].split()]
dur = [float(x) for x in item["duration"].split()]
txt = item["text"].split()
pho = item["phoneme"].split()
assert len(pit) == len(typ) == len(dur) == len(txt) == len(pho), "array length mismatch"

if not BAK.exists():
    BAK.write_text(TGT.read_text())  # preserve the measured-only grid

# 1) promote pitch-0 word slots to real notes (estimated from contour / measured)
#    slot 5  = 4th hook syllable -> monotone hook pitch 70
#    slot 10 = 'be'              -> 70 (tail of the held 'to')
#    slot 12 = 'show'            -> 70 (line lead-in; f0 hinted ~71)
#    slot 22 = 'it'(L4 lead)     -> 70 (f0 hinted ~70)
#    slot 26 = 'wrong'           -> 68 MEASURED from f0 (50% voiced)
PROMOTE = {5: 70, 10: 70, 12: 70, 22: 70, 26: 68}
for i, p in PROMOTE.items():
    pit[i] = p
    typ[i] = 2

# 2) split the 'defeated' rest (the <SP> at index 11) into de-feat-ed.
#    Estimated arch contour around the surrounding ~69 region.
SPLIT_IDX = 11
assert txt[SPLIT_IDX] in ("<SP>",) and pit[SPLIT_IDX] == 0, f"unexpected slot {SPLIT_IDX}: {txt[SPLIT_IDX]} {pit[SPLIT_IDX]}"
d = dur[SPLIT_IDX]
new_pit = [69, 70, 67]            # de / feat(stressed, higher) / ed
new_dur = [round(d / 3, 4)] * 3
new_txt = ["de", "feat", "ed"]
new_pho = ["en_D-IH0", "en_F-IY1-T", "en_AH0-D"]
new_typ = [2, 2, 2]

def splice(arr, vals):
    return arr[:SPLIT_IDX] + vals + arr[SPLIT_IDX + 1:]

pit = splice(pit, new_pit); typ = splice(typ, new_typ); dur = splice(dur, new_dur)
txt = splice(txt, new_txt); pho = splice(pho, new_pho)

# 3) DO NOT paint the f0 at the reconstructed notes. SoulX voices the target from
#    its f0 track (soulxsinger/utils/data_processor.py reads meta['f0']), and these
#    notes sit on unvoiced f0 (0) because the lead vocal is buried there in the
#    original. We tried painting them with the note pitch — it makes them sound, but
#    as chipmunk-y, robotic artifacts (estimated pitch + flat step-f0 + a vowel
#    forced where the consonant can't articulate). Honest silence is better: we keep
#    f0 = 0 so the buried syllables simply DON'T voice. The note_pitch/splice still
#    give the full-count ORDER its slot structure; those slots just render as rests.
#    This is intentional — the dropped syllables map exactly to the instrumental
#    break, where the song has no singable lead to begin with.

# 4) cap the long held line-2 note. SoulX RE-ARTICULATES a single note longer than
#    ~1s into a doubled syllable ("done" -> "done done"). Cap it and push the excess
#    into the following (silent, rested) slot so total timing/sync is preserved.
CAP_IDX, CAP = 9, 0.65
assert txt[CAP_IDX] == "to", f"cap anchor moved: {txt[CAP_IDX]!r}"
if dur[CAP_IDX] > CAP:
    excess = round(dur[CAP_IDX] - CAP, 4)
    dur[CAP_IDX] = CAP
    dur[CAP_IDX + 1] = round(dur[CAP_IDX + 1] + excess, 4)

# 5) tame the f0 WITHOUT killing the melody. SoulX carries the tune in the f0 track, so
#    it must stay (stripping it -> no melody) and keep its note-to-note motion + natural
#    vibrato (flattening every note to a constant -> robotic). The only thing that hurts
#    is large intra-note excursions (e.g. the line-2 note running to ~578Hz), which SoulX
#    re-articulates -> chipmunk/doubling. So clamp each voiced frame to within ±1.5
#    semitones of its note's median: melody + gentle vibrato survive, only spikes are cut.
import os
import statistics
# clamp width in semitones, from env so the sweep can vary it:
#   "raw"  -> no f0 change (keep extracted contour, spikes and all)
#   "0"    -> flatten each note to its median (fully steady, can be robotic)
#   <n>    -> clamp each voiced frame to within ±n semitones of its note median
_CLAMP = os.environ.get("BEATIT_CLAMP_SEMI", "1.5").strip().lower()
f0 = [float(x) for x in item["f0"].split()]
fps = len(f0) / sum(dur)
onset = 0.0
clamped = 0
if _CLAMP != "raw":
    width = float(_CLAMP)
    SEMI = 2 ** (width / 12) if width > 0 else None
    for i, (p, d) in enumerate(zip(pit, dur)):
        a = int(round(onset * fps)); b = int(round((onset + d) * fps)); onset += d
        if p <= 0:
            continue
        idx = [k for k in range(a, min(b, len(f0))) if f0[k] > 1.0]
        if len(idx) < 2:
            continue
        med = statistics.median(f0[k] for k in idx)
        hit = False
        if SEMI is None:                       # flatten to median
            for k in idx:
                if f0[k] != med:
                    f0[k] = round(med, 1); hit = True
        else:                                  # clamp to +/- width semitones
            lo, hi = med / SEMI, med * SEMI
            for k in idx:
                if f0[k] > hi:
                    f0[k] = round(hi, 1); hit = True
                elif f0[k] < lo:
                    f0[k] = round(lo, 1); hit = True
        clamped += hit
print(f"f0 clamp={_CLAMP!r}: notes adjusted={clamped}")
item["f0"] = " ".join(f"{x:.1f}" for x in f0)
print(f"notes with f0 spikes clamped (+/-1.5 semitone): {clamped}")

item["note_pitch"] = " ".join(str(x) for x in pit)
item["note_type"] = " ".join(str(x) for x in typ)
item["duration"] = " ".join(f"{x:.4f}" for x in dur)
item["text"] = " ".join(txt)
item["phoneme"] = " ".join(pho)
TGT.write_text(json.dumps(g, indent=2))

print(f"slots: {len(pit)} (was {len(pit)-2}) | sung notes: {sum(1 for p in pit if p>0)}")
print(f"total duration preserved: {sum(dur):.3f}s")
print("backup of measured-only grid:", BAK.name)
