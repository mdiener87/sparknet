#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG_PATH="$PROJECT_ROOT/configs/sparknet-400m/datasets_v2.json"
NUM_SHARDS="${1:-24}"
DATASET_ROOT="$PROJECT_ROOT/datasets/sparknet-v2-pretrain"
LOG_DIR="$PROJECT_ROOT/logs/sparknet-400m-v2-dataset-build"

mkdir -p "$LOG_DIR"

echo "======================================"
echo "Building SparkNet-400M v2 full pretraining dataset"
echo "Config: $CONFIG_PATH"
echo "Target shards: $NUM_SHARDS"
echo "Output root: $DATASET_ROOT"
echo "Started at: $(date)"
echo "======================================"

{
  echo "Date: $(date)"
  echo "Host: $(hostname)"
  echo "Git commit:"
  git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo "N/A"
  echo
} > "$LOG_DIR/launch_info.txt"

bash "$PROJECT_ROOT/scripts/sparknet-400m/build_datasets.sh" \
  "$CONFIG_PATH" \
  "$NUM_SHARDS" \
  | tee "$LOG_DIR/build.log"

echo "======================================"
echo "SparkNet-400M v2 full dataset build finished"
echo "Finished at: $(date)"
echo "======================================"
