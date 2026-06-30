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

OUTPUT_DIR="output"
SEED=0
mkdir -p logs "${OUTPUT_DIR}/_hidden_states"

# : dataset:max_samples
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

BLOCK_SIZES=(4 8)

for task in "${TASKS[@]}"; do
  IFS=':' read -r DATASET_NAME MAX_SAMPLES <<< "$task"

  echo "========================================================"
  echo "Running Benchmark: $DATASET_NAME with $MAX_SAMPLES samples"
  echo "========================================================"

  for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
    LOG_FILE="logs/${DATASET_NAME}_bs${BLOCK_SIZE}.log"
    echo "[run] dataset=${DATASET_NAME} block_size=${BLOCK_SIZE} log=${LOG_FILE}"

    torchrun \
      --nproc_per_node=1 \
      --master_port=29606 \
      benchmark.py \
      --dataset "$DATASET_NAME" \
      --max-samples "$MAX_SAMPLES" \
      --model-name-or-path Qwen/Qwen3-4B \
      --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
      --max-new-tokens 2048 \
      --temperature 0.0 \
      --seed "$SEED" \
      --block-size "$BLOCK_SIZE" \
      --dir "$OUTPUT_DIR" \
      2>&1 | tee "$LOG_FILE"
  done
done

echo "Done. Results saved under ${OUTPUT_DIR}/"
