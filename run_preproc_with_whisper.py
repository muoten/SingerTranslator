"""Run SoulX-Singer preprocess pipeline with whisper as the English ASR
(bypassing the NeMo dep that's broken on Mac/torch-2.2)."""
import argparse
import os
import re
import sys

sys.path.insert(0, '/Users/milhouse/claude-code/SoulX-Singer')

# IMPORTANT: monkey-patch the English ASR class BEFORE pipeline imports it
import preprocess.tools.lyric_transcription as lt
import numpy as np


def _clean_word(word: str) -> str:
    return re.sub(r"[\?\.,:]", "", word).strip()


INITIAL_PROMPT = os.environ.get('WHISPER_INITIAL_PROMPT', '')


class WhisperASREn:
    """Drop-in replacement for SoulX's _ASREnModel using OpenAI whisper."""
    def __init__(self, model_path: str, device: str):
        import whisper
        size = os.environ.get('WHISPER_MODEL', 'large')
        print(f'[whisper] loading model={size}')
        self.model = whisper.load_model(size)
        self.device = device

    def process(self, wav_fn: str):
        kwargs = dict(language='en', word_timestamps=True)
        if INITIAL_PROMPT:
            kwargs['initial_prompt'] = INITIAL_PROMPT
            print(f'[whisper] using initial_prompt of {len(INITIAL_PROMPT)} chars')
        result = self.model.transcribe(wav_fn, **kwargs)
        print(f'[whisper] text: {result.get("text","").strip()}')

        raw_words = []
        raw_timestamps = []
        for seg in result.get('segments', []):
            for w in seg.get('words', []):
                word = _clean_word(str(w.get('word', '')))
                if not word:
                    continue
                s = float(w.get('start', 0.0))
                e = float(w.get('end', 0.0))
                raw_words.append(word)
                raw_timestamps.append([s, e])

        words, durs = lt._build_words_with_gaps(raw_words, raw_timestamps, wav_fn)

        f0_path = os.path.splitext(wav_fn)[0] + "_f0.npy"
        if os.path.exists(f0_path):
            words, durs = lt._word_dur_post_process(
                words, durs, np.load(f0_path)
            )

        return words, durs


# Patch the class reference in the module
lt._ASREnModel = WhisperASREn


# Now run the pipeline normally
from preprocess.pipeline import main as pipeline_main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--language', default='English')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--vocal_sep', default='False')
    parser.add_argument('--max_merge_duration', type=int, default=60000)
    parser.add_argument('--midi_transcribe', default='True')
    args = parser.parse_args()
    # convert string bools to bools as the pipeline expects
    args.vocal_sep = args.vocal_sep.lower() in ('true', '1', 'yes')
    args.midi_transcribe = args.midi_transcribe.lower() in ('true', '1', 'yes')
    pipeline_main(args)
