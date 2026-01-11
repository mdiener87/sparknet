#!/usr/bin/env bash
set -euo pipefail

# Resolve paths robustly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="sparknet-400m-v1-instruct"

DATASET_ROOT="$PROJECT_ROOT/datasets/sft_chat_v1"
CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints/sparknet-400m-v1"
TOKENIZER_DIR="$PROJECT_ROOT/tokenizer-v6"

LOG_DIR="$PROJECT_ROOT/logs/$RUN_NAME"
mkdir -p "$LOG_DIR"

echo "======================================"
echo "Launching SparkNet-400M v1 SFT Phase 1"
echo "Run name: $RUN_NAME"
echo "Started at: $(date)"
echo "======================================"

# Preflight checks
if [ ! -d "$DATASET_ROOT" ]; then
  echo "ERROR: Dataset root not found: $DATASET_ROOT"
  exit 1
fi

SHARD_COUNT="$(ls -d "$DATASET_ROOT"/shard-* 2>/dev/null | wc -l | tr -d ' ')"
if [ "$SHARD_COUNT" -lt 1 ]; then
  echo "ERROR: No shard-* dirs found under $DATASET_ROOT"
  exit 1
fi

if [ ! -d "$CHECKPOINT_ROOT" ]; then
  echo "ERROR: Base checkpoint not found: $CHECKPOINT_ROOT"
  exit 1
fi

if [ ! -d "$TOKENIZER_DIR" ]; then
  echo "ERROR: Tokenizer dir not found: $TOKENIZER_DIR"
  exit 1
fi

echo "Dataset shards found: $SHARD_COUNT"
echo "Disk space for $PROJECT_ROOT:"
df -h "$PROJECT_ROOT" || true

# Optional: record system info up front
{
  echo "Date: $(date)"
  echo "Host: $(hostname)"
  echo "Git commit:"
  git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo "N/A"
  echo
  echo "nvidia-smi:"
  nvidia-smi 2>/dev/null || echo "nvidia-smi not found"
  echo
} > "$LOG_DIR/launch_info.txt"

# Launch training
python "$PROJECT_ROOT/scripts/sparknet-sft/train_sft.py" \
    --run-name "$RUN_NAME" \
  | tee "$LOG_DIR/train.log"

echo "======================================"
echo "SparkNet-400M v1 SFT training finished"
echo "Finished at: $(date)"
echo "======================================"
