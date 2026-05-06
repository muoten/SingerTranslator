"""Surgical lyric edit: take a SoulX-Singer metadata JSON, replace one
word everywhere with another, regenerate that word's phoneme group via
g2p_en, and save. All other fields (note_pitch, note_type, duration, f0,
time) are preserved exactly so we test ONLY the lyric/phoneme change.

Usage:
    python swap_word.py --in <in.json> --out <out.json> \
        --old pretty --new broken
"""
import argparse
import json
from g2p_en import G2p

g2p = G2p()


def english_phoneme(word: str) -> str:
    """Convert one English word to SoulX-Singer's phoneme token format,
    e.g. 'pretty' -> 'en_P-R-IH1-T-IY0'."""
    phones = [p for p in g2p(word) if p not in (' ',)]
    return 'en_' + '-'.join(phones)


def swap(seg: dict, old: str, new: str,
         phoneme_override: str | None = None,
         duration_boost: float = 0.0) -> dict:
    text_tokens = seg['text'].split()
    phon_tokens = seg['phoneme'].split()
    if len(text_tokens) != len(phon_tokens):
        raise ValueError(
            f"text/phoneme token-count mismatch: "
            f"{len(text_tokens)} vs {len(phon_tokens)}"
        )
    new_phon = phoneme_override or english_phoneme(new)
    print(f"  '{new}' -> {new_phon}{' (override)' if phoneme_override else ''}")

    duration_tokens = seg.get('duration', '').split()
    has_dur = len(duration_tokens) == len(text_tokens)

    n_swapped = 0
    swapped_indices = []
    for i, t in enumerate(text_tokens):
        if t.lower() == old.lower():
            text_tokens[i] = new
            phon_tokens[i] = new_phon
            swapped_indices.append(i)
            n_swapped += 1
    print(f"  swapped {n_swapped} occurrence(s) of '{old}' -> '{new}'")

    if duration_boost and has_dur:
        for i in swapped_indices:
            old_d = float(duration_tokens[i])
            # Steal from the next <SP> rest if available, else previous
            steal_from = None
            for j in (i + 1, i - 1):
                if 0 <= j < len(text_tokens) and text_tokens[j] == '<SP>':
                    rest_d = float(duration_tokens[j])
                    if rest_d > duration_boost + 0.05:
                        steal_from = j
                        break
            if steal_from is None:
                print(f"  WARN no neighboring <SP> with enough slack at index {i}; skipping boost")
                continue
            duration_tokens[i] = f"{old_d + duration_boost:.2f}"
            duration_tokens[steal_from] = f"{float(duration_tokens[steal_from]) - duration_boost:.2f}"
            print(f"  index {i}: dur {old_d:.2f}s + {duration_boost:.2f}s "
                  f"(stole from <SP> at index {steal_from})")

    out = dict(seg)
    out['text'] = ' '.join(text_tokens)
    out['phoneme'] = ' '.join(phon_tokens)
    if duration_boost and has_dur:
        out['duration'] = ' '.join(duration_tokens)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', dest='out', required=True)
    ap.add_argument('--old', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--phoneme', default=None,
                    help="Override the auto-generated phoneme, e.g. 'en_B-R-OW1-K-K-AH0-N'")
    ap.add_argument('--duration_boost', type=float, default=0.0,
                    help='Add N seconds to swapped-word slot, stealing from a neighboring <SP> rest')
    args = ap.parse_args()

    data = json.load(open(args.inp))
    edited = [swap(s, args.old, args.new,
                   phoneme_override=args.phoneme,
                   duration_boost=args.duration_boost) for s in data]
    json.dump(edited, open(args.out, 'w'), ensure_ascii=False, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
