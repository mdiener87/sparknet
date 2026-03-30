#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG_PATH="$PROJECT_ROOT/configs/sparknet-400m/pretrain_v2_smoke.json"
RUN_NAME="sparknet-400m-v2-smoke"
DATASET_ROOT="$PROJECT_ROOT/datasets/sparknet-v2-pretrain-smoke"
LOG_DIR="$PROJECT_ROOT/logs/$RUN_NAME"

mkdir -p "$LOG_DIR"

echo "======================================"
echo "Launching SparkNet-400M v2 smoke pretraining"
echo "Config: $CONFIG_PATH"
echo "Dataset: $DATASET_ROOT"
echo "Started at: $(date)"
echo "======================================"

if [ ! -d "$DATASET_ROOT" ]; then
  echo "ERROR: Dataset root not found: $DATASET_ROOT"
  exit 1
fi

SHARD_COUNT="$(find "$DATASET_ROOT" -maxdepth 1 -type d -name 'shard-*' | wc -l | tr -d ' ')"
if [ "$SHARD_COUNT" -lt 1 ]; then
  echo "ERROR: No shard-* directories found under $DATASET_ROOT"
  exit 1
fi

{
  echo "Date: $(date)"
  echo "Host: $(hostname)"
  echo "Git commit:"
  git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo "N/A"
  echo
  echo "nvidia-smi:"
  nvidia-smi || true
  echo
} > "$LOG_DIR/launch_info.txt"

python "$PROJECT_ROOT/scripts/sparknet-400m/train_pretrain.py" \
  --config "$CONFIG_PATH" \
  | tee "$LOG_DIR/train.log"

echo "======================================"
echo "SparkNet-400M v2 smoke pretraining finished"
echo "Finished at: $(date)"
echo "======================================"
