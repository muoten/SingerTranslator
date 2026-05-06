#!/usr/bin/env bash
# Wrapper around SoulX-Singer SVS inference. Picks reasonable defaults for
# the SingerTranslator workflow on Mac (CPU, no fp16, auto pitch shift).
#
# Usage:
#   scripts/sing.sh <prompt_wav> <prompt_metadata> <target_metadata> <output_dir>
#
# Example:
#   scripts/sing.sh \
#       /path/to/SoulX-Singer/example/audio/en_prompt.mp3 \
#       /path/to/SoulX-Singer/example/audio/en_prompt.json \
#       /tmp/my_target.json \
#       /tmp/sung_output
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <prompt_wav> <prompt_metadata.json> <target_metadata.json> <output_dir>" >&2
    exit 2
fi

PROMPT_WAV="$1"
PROMPT_META="$2"
TARGET_META="$3"
OUT_DIR="$4"

SOULX_ROOT="${SOULX_ROOT:-/Users/milhouse/claude-code/SoulX-Singer}"
PYBIN="$SOULX_ROOT/venv/bin/python"

cd "$SOULX_ROOT"
# IMPORTANT: --control score uses note_pitch + note_type from metadata.
# Default 'melody' uses frame-level f0 instead, ignoring note_pitch edits.
# For SingerTranslator surgery on melody, score is the right mode.
PYTHONPATH=. "$PYBIN" -m cli.inference \
    --device cpu \
    --control score \
    --model_path pretrained_models/SoulX-Singer/model.pt \
    --config soulxsinger/config/soulxsinger.yaml \
    --prompt_wav_path "$PROMPT_WAV" \
    --prompt_metadata_path "$PROMPT_META" \
    --target_metadata_path "$TARGET_META" \
    --phoneset_path soulxsinger/utils/phoneme/phone_set.json \
    --save_dir "$OUT_DIR" \
    --auto_shift \
    --pitch_shift 0
