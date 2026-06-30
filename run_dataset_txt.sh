#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Optional venv activation
if [[ -f ../dflash_venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source ../dflash_venv/bin/activate
fi

# Keep HF cache on /tmp to avoid /home full
export HF_HOME="${HF_HOME:-./hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-./hf_cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-./hf_cache/hub}"
mkdir -p "$HF_HUB_CACHE"

# Default GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash run_dataset_txt.sh --input-file TXT/copa.txt [options]

Required:
  --input-file PATH           TXT file path, one prompt per line

Optional:
  --dataset-name NAME         Output dataset name (default: txt filename stem)
  --output-dir PATH           Output root dir (default: utils)
  --model PATH                Target model (default: Qwen/Qwen3-4B)
  --draft PATH                Draft model (default: z-lab/Qwen3-4B-DFlash-b16)
  --block-sizes "..."         Space-separated block sizes
                              (default: "1 2 4 8 14 15 16 17 18")
  --max-samples N             Max number of lines to run (default: all)
  --max-new-tokens N          Max generation tokens (default: 2048)
  --temperature FLOAT         Temperature (default: 0.0)
  --seed N                    Seed (default: 0)
  --no-save-hidden-states     Do not save hidden states

Examples:
  bash run_dataset_txt.sh --input-file TXT/copa.txt
  bash run_dataset_txt.sh --input-file TXT/copa.txt --max-samples 32 --seed 0
  bash run_dataset_txt.sh --input-file TXT/copa.txt --block-sizes "4 8 16 18"
EOF
}

INPUT_FILE=""
DATASET_NAME=""
OUTPUT_DIR="utils"
MODEL_NAME="Qwen/Qwen3-4B"
DRAFT_NAME="z-lab/Qwen3-4B-DFlash-b16"
BLOCK_SIZES="1 2 4 8 14 15 16 17 18"
MAX_SAMPLES=""
MAX_NEW_TOKENS="2048"
TEMPERATURE="0.0"
SEED="0"
SAVE_HIDDEN_STATES="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-file)
      INPUT_FILE="$2"
      shift 2
      ;;
    --dataset-name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --draft)
      DRAFT_NAME="$2"
      shift 2
      ;;
    --block-sizes)
      BLOCK_SIZES="$2"
      shift 2
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --no-save-hidden-states)
      SAVE_HIDDEN_STATES="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$INPUT_FILE" ]]; then
  echo "Error: --input-file is required"
  usage
  exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Error: input file does not exist: $INPUT_FILE"
  exit 1
fi

mkdir -p "$OUTPUT_DIR" logs

CMD=(
  python dataset.py
  --input-file "$INPUT_FILE"
  --output-dir "$OUTPUT_DIR"
  --model-name-or-path "$MODEL_NAME"
  --draft-name-or-path "$DRAFT_NAME"
  --block-sizes ${BLOCK_SIZES}
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --seed "$SEED"
)

if [[ -n "$DATASET_NAME" ]]; then
  CMD+=(--dataset-name "$DATASET_NAME")
fi

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi

if [[ "$SAVE_HIDDEN_STATES" == "false" ]]; then
  CMD+=(--no-save-hidden-states)
fi

echo "Running command:"
printf ' %q' "${CMD[@]}"
echo

LOG_FILE="logs/dataset_$(date +%Y%m%d_%H%M%S).log"
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo "Done. Log saved to: $LOG_FILE"
