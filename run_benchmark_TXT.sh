#  GPU
export CUDA_VISIBLE_DEVICES=6
export HF_ENDPOINT=https://hf-mirror.com
source ../dflash_venv/bin/activate

# 
mkdir -p logs

# 
SEEDS=(0)

# 
TASKS=(
  # "gsm8k:8"
  # "math500:8"
  # "aime24:8"
  # "aime25:8"
  # "humaneval:8"
  # "mbpp:8"
  # "livecodebench:8"
  # "swe-bench:8"
  # "mt-bench:8"
  # "alpaca:8"
)

for task in "${TASKS[@]}"; do
  IFS=':' read -r DATASET_NAME MAX_SAMPLES <<< "$task"

  echo "========================================================"
  echo "Running Benchmark: $DATASET_NAME with $MAX_SAMPLES samples"
  echo "========================================================"

  for SEED in "${SEEDS[@]}"; do
      for BLOCK_SIZE in 14 15 16 17 18; do
        torchrun \
          --nproc_per_node=1 \
          --master_port=29605 \
          benchmark_TXT.py \
          --dataset "$DATASET_NAME" \
          --max-samples "$MAX_SAMPLES" \
          --model-name-or-path Qwen/Qwen3-4B \
          --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
          --max-new-tokens 2048 \
          --temperature 0.0 \
          --seed "$SEED" \
          --block-size "$BLOCK_SIZE" \
          --dir _zh_TXT
      done
  done

done