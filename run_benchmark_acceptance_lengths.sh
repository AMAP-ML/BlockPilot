#!/usr/bin/env bash
set -euo pipefail

# HuggingFace  /tmp， /home 
export HF_HOME="./hf_cache"
export HF_HUB_CACHE="./hf_cache/hub"
export TRANSFORMERS_CACHE="./hf_cache/hub"
mkdir -p "$HF_HUB_CACHE"

#  GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f ../dflash_venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source ../dflash_venv/bin/activate
fi

# （: {OUTPUT_DIR}/{dataset}_0T0.0/{block_size}.json）
OUTPUT_DIR="utils"
SEED=0
TEMPERATURE=0.0

mkdir -p logs

# ：dataset:max_samples
# （ 8/32）
TASKS=(
  "gsm8k:8"
  "math500:8"
  "aime24:8"
  "aime25:8"
  "humaneval:8"
  "mbpp:8"
  "livecodebench:8"
  "swe-bench:8"
  "mt-bench:8"
  "alpaca:8"
)

# 
BLOCK_SIZES=(1 2 4 8 14 15 16 17 18)

# （ run_benchmark.sh ）
MODEL_NAME="Qwen/Qwen3-4B"
DRAFT_NAME="z-lab/Qwen3-4B-DFlash-b16"
MAX_NEW_TOKENS=2048
MASTER_PORT=29606

for task in "${TASKS[@]}"; do
  IFS=':' read -r DATASET_NAME MAX_SAMPLES <<< "$task"

  echo "========================================================"
  echo "Running acceptance-length benchmark: ${DATASET_NAME} (${MAX_SAMPLES} samples)"
  echo "========================================================"

  for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
    LOG_FILE="logs/${DATASET_NAME}_bs${BLOCK_SIZE}.log"
    echo "[run] dataset=${DATASET_NAME} block_size=${BLOCK_SIZE} log=${LOG_FILE}"

    torchrun \
      --nproc_per_node=1 \
      --master_port="${MASTER_PORT}" \
      benchmark.py \
      --dataset "${DATASET_NAME}" \
      --max-samples "${MAX_SAMPLES}" \
      --model-name-or-path "${MODEL_NAME}" \
      --draft-name-or-path "${DRAFT_NAME}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --temperature "${TEMPERATURE}" \
      --seed "${SEED}" \
      --block-size "${BLOCK_SIZE}" \
      --no-save-hidden-states \
      --dir "${OUTPUT_DIR}" \
      2>&1 | tee "${LOG_FILE}"
  done
done

echo "Done. Acceptance-length JSON files saved under: ${OUTPUT_DIR}"
