"""Melody-fidelity check: render the beat_it grid on a neutral syllable (la),
so the TUNE can be A/B'd against the original with no lyrics involved."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import singer, soulx_freelyrics as fl
SONG="beat_it"; OUT=ROOT/"_tmp_freelyrics_demo"; OUT.mkdir(exist_ok=True)
# neutral syllable on every sung slot, matching the 4/4/8/7 word counts
counts=fl.word_counts(SONG)
words={i+1:["la"]*n for i,n in enumerate(counts)}
tgt=fl.build_target(words, OUT/"beat_it_neutral_target.json", song=SONG)
vocal=singer.soulx_render(tgt.resolve(), OUT.resolve(), n_steps=32, song=SONG)
mix=singer.mix_with_accompaniment(Path(vocal), OUT/"beat_it_neutral_mix.wav", song=SONG)
print("NEUTRAL_MIX:", mix)
