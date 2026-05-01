#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG_PATH="$PROJECT_ROOT/configs/sparknet-400m/pretrain_v2_12b.json"
RUN_NAME="sparknet-400m-v2-12b"
DATASET_ROOT="$PROJECT_ROOT/datasets/sparknet-v2-pretrain"
EXPECTED_SHARDS=24
LOG_DIR="$PROJECT_ROOT/logs/$RUN_NAME"

mkdir -p "$LOG_DIR"

echo "======================================"
echo "Launching SparkNet-400M v2 12B pretraining"
echo "Config: $CONFIG_PATH"
echo "Dataset: $DATASET_ROOT"
echo "Started at: $(date)"
echo "======================================"

if [ ! -d "$DATASET_ROOT" ]; then
  echo "ERROR: Dataset root not found: $DATASET_ROOT"
  exit 1
fi

SHARD_COUNT="$(find "$DATASET_ROOT" -maxdepth 1 -type d -name 'shard-*' | wc -l | tr -d ' ')"
if [ "$SHARD_COUNT" -ne "$EXPECTED_SHARDS" ]; then
  echo "ERROR: Expected $EXPECTED_SHARDS shards under $DATASET_ROOT, found $SHARD_COUNT"
  exit 1
fi

CACHE_DIR="$PROJECT_ROOT/cache"
if [ -d "$CACHE_DIR" ]; then
  WIKITEXT_HITS="$(find "$CACHE_DIR" -maxdepth 4 -type d -name '*wikitext*' 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$WIKITEXT_HITS" -eq 0 ]; then
    echo "WARNING: No cached wikitext dataset found under $CACHE_DIR (offline eval may fail)"
  fi
else
  echo "WARNING: Cache dir not found at $CACHE_DIR (offline eval may fail)"
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
echo "SparkNet-400M v2 12B pretraining finished"
echo "Finished at: $(date)"
echo "======================================"
