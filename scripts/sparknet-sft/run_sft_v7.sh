#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG_PATH="${1:-$PROJECT_ROOT/configs/sparknet-400m/sft_v7.json}"
RUN_NAME="${RUN_NAME:-sparknet-400m-v2-instruct-v2}"

LOG_DIR="$PROJECT_ROOT/logs/$RUN_NAME"
mkdir -p "$LOG_DIR"

cd "$PROJECT_ROOT"

echo "======================================"
echo "Launching SparkNet-400M SFT v7"
echo "Config: $CONFIG_PATH"
echo "Run name: $RUN_NAME"
echo "Started at: $(date)"
echo "======================================"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "ERROR: Config file not found: $CONFIG_PATH"
  exit 1
fi

if [ ! -d "$PROJECT_ROOT/checkpoints/sparknet-400m-v2-12b/checkpoint-12000" ]; then
  echo "ERROR: Base checkpoint not found: $PROJECT_ROOT/checkpoints/sparknet-400m-v2-12b/checkpoint-12000"
  exit 1
fi

if [ ! -d "$PROJECT_ROOT/datasets/sft_chat_v7" ]; then
  echo "ERROR: Dataset not found: $PROJECT_ROOT/datasets/sft_chat_v7"
  echo "Run build_datasets_v7.sh first."
  exit 1
fi

if [ ! -d "$PROJECT_ROOT/tokenizer-v6" ]; then
  echo "ERROR: Tokenizer dir not found: $PROJECT_ROOT/tokenizer-v6"
  exit 1
fi

{
  echo "Date: $(date)"
  echo "Host: $(hostname)"
  echo "Config: $CONFIG_PATH"
  echo "Git commit:"
  git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo "N/A"
  echo
  echo "nvidia-smi:"
  nvidia-smi 2>/dev/null || echo "nvidia-smi not found"
  echo
} > "$LOG_DIR/launch_info.txt"

python "$PROJECT_ROOT/scripts/sparknet-sft/train_sft_v7.py" \
  --config "$CONFIG_PATH" \
  --run-name "$RUN_NAME" \
  | tee "$LOG_DIR/train.log"

echo "======================================"
echo "SparkNet-400M SFT v7 finished"
echo "Finished at: $(date)"
echo "======================================"
