"""Validation render for the newly-added Beat It song.

Builds the beat_it demo lyric onto the chorus grid and runs the full
SoulX render + accompaniment mix — the end-to-end smoke test that the
assets/beat_it/* assets + soulx_freelyrics registration are wired correctly.

Run from the repo root with the SoulX venv:
  SINGER_DEVICE=cpu vendor/SoulX-Singer/venv/bin/python scripts/render_beat_it_demo.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import singer  # noqa: E402
import soulx_freelyrics as fl  # noqa: E402

SONG = "beat_it"
OUT = ROOT / "_tmp_freelyrics_demo"
OUT.mkdir(exist_ok=True)


def main():
    lines = fl.DEMOS[SONG]
    print("demo lyric:", lines)
    ok, words = fl.check(lines, SONG)
    if not ok:
        raise SystemExit("demo lyric does not fit the beat_it template")

    tgt = fl.build_target(words, OUT / "beat_it_demo_target.json", song=SONG)
    print("built target:", tgt)

    vocal = singer.soulx_render(tgt.resolve(), OUT.resolve(), n_steps=32, song=SONG)
    print("rendered vocal:", vocal)

    mix = singer.mix_with_accompaniment(Path(vocal), OUT / "beat_it_demo_mix.wav", song=SONG)
    print("MIX:", mix)


if __name__ == "__main__":
    main()
