#!/usr/bin/env bash
set -euo pipefail

# Resolve paths robustly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PREV_RUN_NAME="sparknet-400m-v1"
RUN_NAME="sparknet-400m-v1-expanded"

LOG_DIR="$PROJECT_ROOT/logs/$RUN_NAME"
mkdir -p "$LOG_DIR"

echo "======================================"
echo "Launching SparkNet-400M v1 expanded pretraining"
echo "Run name: $RUN_NAME"
echo "Started at: $(date)"
echo "======================================"

# Preflight checks
DATASET_ROOT="$PROJECT_ROOT/datasets/sparknet-v6-pretrain"
EXPECTED_SHARDS=16
if [ ! -d "$DATASET_ROOT" ]; then
  echo "ERROR: Dataset root not found: $DATASET_ROOT"
  exit 1
fi
SHARD_COUNT="$(ls -d "$DATASET_ROOT"/shard-* 2>/dev/null | wc -l | tr -d ' ')"
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

echo "Disk space for $PROJECT_ROOT:"
df -h "$PROJECT_ROOT" || true

# Find latest checkpoint from the previous run (6B tokens)
PREV_RUN_DIR="$PROJECT_ROOT/checkpoints/$PREV_RUN_NAME"
RESUME_CKPT=""
if [ -d "$PREV_RUN_DIR" ]; then
  RESUME_CKPT="$(ls -d "$PREV_RUN_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
fi
if [ -z "$RESUME_CKPT" ]; then
  echo "ERROR: Could not find a checkpoint under $PREV_RUN_DIR"
  exit 1
fi

# Optional: record system info up front
{
  echo "Date: $(date)"
  echo "Host: $(hostname)"
  echo "Git commit:"
  git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo "N/A"
  echo
  echo "nvidia-smi:"
  nvidia-smi
  echo
} > "$LOG_DIR/launch_info.txt"

# Launch training
python "$PROJECT_ROOT/scripts/sparknet-400m/train_pretrain.py" \
    --resume "$RESUME_CKPT" \
    --run-name "$RUN_NAME" \
  | tee "$LOG_DIR/train.log"

echo "======================================"
echo "SparkNet-400M v1 training finished"
echo "Finished at: $(date)"
echo "======================================"
