#!/usr/bin/env bash
# Run benchmark.py at fixed block sizes and save {block_size}.json per dataset.
#
# Usage:
#   bash run_benchmark_block_size.sh
#   bash run_benchmark_block_size.sh --block-sizes 14 15 16 17 18
#   bash run_benchmark_block_size.sh --datasets gsm8k math500 --block-sizes 16
#   CUDA_VISIBLE_DEVICES=0 bash run_benchmark_block_size.sh
#
# Output:
#   {OUTPUT_DIR}/{dataset}_{seed}T{temperature}/{block_size}.json
#   {OUTPUT_DIR}/_hidden_states/{dataset}_seed{seed}T{temperature}samples{max_samples}.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ======================== Config (edit here) ========================
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
VENV_PATH="${VENV_PATH:-../dflash_venv/bin/activate}"

MODEL="Qwen/Qwen3-8B"
DRAFT="z-lab/Qwen3-8B-DFlash-b16"
OUTPUT_DIR="utils"

TEMPERATURE="0.0"
SEED="0"
MAX_NEW_TOKENS="2048"
MASTER_PORT="29606"
NPROC="1"

# dataset:max_samples
TASKS=(
  "gsm8k:32"
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

BLOCK_SIZES=(14 15 16 17 18)
SKIP_EXISTING="true"
# ====================================================================

usage() {
  cat <<'EOF'
Usage: bash run_benchmark_block_size.sh [options]

Options:
  --block-sizes  N [N ...]   Block sizes to benchmark (default: 14 15 16 17 18)
  --datasets     D [D ...]   Datasets to run (default: all TASKS in script)
  --max-samples  N           Override max samples for all selected datasets
  --output-dir   DIR         Output directory (default: utils)
  --model        PATH        Target model path
  --draft        PATH        Draft model path
  --temperature  T           Sampling temperature (default: 0.0)
  --seed         N            Random seed (default: 0)
  --gpu          N            CUDA device id
  --force                    Re-run even if {block_size}.json already exists
  -h, --help                 Show this help
EOF
}

SELECTED_DATASETS=()
SELECTED_MAX_SAMPLES=""
CLI_BLOCK_SIZES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --block-sizes)
      shift
      CLI_BLOCK_SIZES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CLI_BLOCK_SIZES+=("$1")
        shift
      done
      ;;
    --datasets)
      shift
      SELECTED_DATASETS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SELECTED_DATASETS+=("$1")
        shift
      done
      ;;
    --max-samples) SELECTED_MAX_SAMPLES="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --draft) DRAFT="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --gpu) CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    --force) SKIP_EXISTING="false"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ${#CLI_BLOCK_SIZES[@]} -gt 0 ]]; then
  BLOCK_SIZES=("${CLI_BLOCK_SIZES[@]}")
fi

export CUDA_VISIBLE_DEVICES
export HF_ENDPOINT

if [[ -f "$VENV_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
fi

mkdir -p logs "$OUTPUT_DIR/_hidden_states"

resolve_tasks() {
  if [[ ${#SELECTED_DATASETS[@]} -eq 0 ]]; then
    printf '%s\n' "${TASKS[@]}"
    return
  fi

  for dataset in "${SELECTED_DATASETS[@]}"; do
    found="false"
    for task in "${TASKS[@]}"; do
      IFS=':' read -r name samples <<< "$task"
      if [[ "$name" == "$dataset" ]]; then
        if [[ -n "$SELECTED_MAX_SAMPLES" ]]; then
          echo "${name}:${SELECTED_MAX_SAMPLES}"
        else
          echo "$task"
        fi
        found="true"
        break
      fi
    done
    if [[ "$found" == "false" ]]; then
      samples="${SELECTED_MAX_SAMPLES:-32}"
      echo "${dataset}:${samples}"
    fi
  done
}

echo "GPU=${CUDA_VISIBLE_DEVICES}"
echo "Model=${MODEL}"
echo "Draft=${DRAFT}"
echo "Output=${OUTPUT_DIR}"
echo "Block sizes=${BLOCK_SIZES[*]}"
echo "Temperature=${TEMPERATURE}, seed=${SEED}"
echo

while IFS= read -r task; do
  [[ -z "$task" ]] && continue
  IFS=':' read -r DATASET_NAME MAX_SAMPLES <<< "$task"

  echo "============================================================"
  echo "Dataset: ${DATASET_NAME}  samples: ${MAX_SAMPLES}"
  echo "============================================================"

  for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
    OUT_JSON="${OUTPUT_DIR}/${DATASET_NAME}_${SEED}T${TEMPERATURE}/${BLOCK_SIZE}.json"
    LOG_FILE="logs/${DATASET_NAME}_bs${BLOCK_SIZE}_T${TEMPERATURE}.log"

    if [[ "$SKIP_EXISTING" == "true" && -f "$OUT_JSON" ]]; then
      echo "[skip] ${OUT_JSON} already exists"
      continue
    fi

    echo "[run ] block_size=${BLOCK_SIZE} -> ${OUT_JSON}"
    torchrun \
      --nproc_per_node="$NPROC" \
      --master_port="$MASTER_PORT" \
      benchmark.py \
      --dataset "$DATASET_NAME" \
      --max-samples "$MAX_SAMPLES" \
      --model-name-or-path "$MODEL" \
      --draft-name-or-path "$DRAFT" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --seed "$SEED" \
      --block-size "$BLOCK_SIZE" \
      --dir "$OUTPUT_DIR" \
      2>&1 | tee "$LOG_FILE"
  done
done < <(resolve_tasks)

echo
echo "Done. Results under ${OUTPUT_DIR}/"
