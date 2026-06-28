"""Syllable-reconstruction check: render the beat_it grid on VARIED nonsense
scat syllables (distinct onsets; 2-syllable scat on 2-syllable reference slots),
so the per-slot articulation, counts and rhythm can be judged — no lyrics used."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import singer, soulx_freelyrics as fl
SONG="beat_it"; OUT=ROOT/"_tmp_freelyrics_demo"; OUT.mkdir(exist_ok=True)
# distinct scat per slot; 2-syllable scat where the reference word is 2 syllables
words = {
    1: ["da", "ba", "do", "bi"],                                  # 4
    2: ["ka", "ti", "ko", "tu"],                                  # 4
    3: ["na", "mo", "baba", "lo", "ni", "su", "ga", "ree"],       # 8 (baba=2syl at 'funky')
    4: ["dada", "koko", "fa", "li", "po", "be", "ta"],            # 7 (dada,koko=2syl)
}
tgt=fl.build_target(words, OUT/"beat_it_scat_target.json", song=SONG)
vocal=singer.soulx_render(tgt.resolve(), OUT.resolve(), n_steps=32, song=SONG)
mix=singer.mix_with_accompaniment(Path(vocal), OUT/"beat_it_scat_mix.wav", song=SONG)
print("SCAT_MIX:", mix)
