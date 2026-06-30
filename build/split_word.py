"""Replace each occurrence of a word with N consecutive slots, each
carrying its own phoneme, duration share, and a copy of the note_pitch.

Goal: force the model to articulate at word boundaries by putting a
plosive (K, P, T) at the end of slot A and the start of slot B.

Example: "broken" → [("brok", "en_B-R-OW1-K"), ("ken", "en_K-AH0-N")]

Usage:
    python split_word.py --in IN.json --out OUT.json --old pretty \
        --pieces 'brok:en_B-R-OW1-K' 'ken:en_K-AH0-N' \
        [--total-duration 0.50]
"""
import argparse
import json


def split_seg(seg: dict, old: str, pieces: list[tuple[str, str]],
              total_duration: float | None = None) -> dict:
    text = seg['text'].split()
    phon = seg['phoneme'].split()
    durs = [float(x) for x in seg['duration'].split()]
    pitches = seg['note_pitch'].split()
    types = seg['note_type'].split()
    n = len(text)
    assert len(phon) == n == len(durs) == len(pitches) == len(types)

    new_text, new_phon, new_durs, new_pitches, new_types = [], [], [], [], []
    n_split = 0

    for i in range(n):
        if text[i].lower() == old.lower():
            # Determine the duration to redistribute. Either fixed (--total-duration)
            # or original duration of this slot, possibly augmented by stealing from
            # the *next* <SP> rest.
            target_dur = total_duration or durs[i]
            steal_idx = None
            if total_duration is not None and total_duration > durs[i]:
                want = total_duration - durs[i]
                if i + 1 < n and text[i + 1] == '<SP>' and durs[i + 1] > want + 0.05:
                    steal_idx = i + 1
                    durs[i + 1] -= want
                    print(f"  slot {i}: stole {want:.2f}s from <SP> rest at {i+1}")
                else:
                    print(f"  WARN slot {i}: cannot reach {total_duration:.2f}s (no slack)")
                    target_dur = durs[i]
            per_piece = target_dur / len(pieces)
            for j, (piece_text, piece_phon) in enumerate(pieces):
                new_text.append(piece_text)
                new_phon.append(piece_phon)
                new_durs.append(per_piece)
                new_pitches.append(pitches[i])  # same pitch for all pieces
                # Articulation hints: 1=onset on first piece, 3=end on last, 2=mid sustain
                if len(pieces) == 1:
                    new_types.append(types[i])
                elif j == 0:
                    new_types.append('1')
                elif j == len(pieces) - 1:
                    new_types.append('3')
                else:
                    new_types.append('2')
            n_split += 1
        else:
            new_text.append(text[i])
            new_phon.append(phon[i])
            new_durs.append(durs[i])
            new_pitches.append(pitches[i])
            new_types.append(types[i])

    print(f"  split {n_split} occurrence(s) of '{old}' into {len(pieces)} pieces")
    print(f"  new slot count: {len(new_text)} (was {n})")

    out = dict(seg)
    out['text'] = ' '.join(new_text)
    out['phoneme'] = ' '.join(new_phon)
    out['duration'] = ' '.join(f"{d:.2f}" for d in new_durs)
    out['note_pitch'] = ' '.join(new_pitches)
    out['note_type'] = ' '.join(new_types)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', dest='out', required=True)
    ap.add_argument('--old', required=True)
    ap.add_argument('--pieces', nargs='+', required=True,
                    help="Each piece as 'text:phoneme', e.g. 'brok:en_B-R-OW1-K'")
    ap.add_argument('--total-duration', type=float, default=None,
                    help="Force total slot duration (steals from next <SP> rest)")
    args = ap.parse_args()

    pieces = []
    for spec in args.pieces:
        text, _, phon = spec.partition(':')
        pieces.append((text, phon))

    data = json.load(open(args.inp))
    edited = [split_seg(s, args.old, pieces, args.total_duration) for s in data]
    json.dump(edited, open(args.out, 'w'), ensure_ascii=False, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
