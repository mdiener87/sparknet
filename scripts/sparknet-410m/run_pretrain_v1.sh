#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="sparknet-410m-v1"
CONFIG="$PROJECT_ROOT/configs/sparknet-410m/pretrain_v1.json"

LOG_DIR="$PROJECT_ROOT/logs/$RUN_NAME"
mkdir -p "$LOG_DIR"

echo "======================================"
echo "SparkNet-410M v1 pretraining"
echo "Run name : $RUN_NAME"
echo "Config   : $CONFIG"
echo "Started  : $(date)"
echo "======================================"

# --- Preflight checks ---

TOKENIZER_DIR="$PROJECT_ROOT/tokenizer-v7"
if [ ! -f "$TOKENIZER_DIR/tokenizer.json" ]; then
  echo "ERROR: tokenizer-v7 not found at $TOKENIZER_DIR"
  echo "       Run: python scripts/sparknet-410m/build_tokenizer.py"
  exit 1
fi

TRAIN_ROOT="$PROJECT_ROOT/datasets/sparknet-v3-pretrain"
if [ ! -d "$TRAIN_ROOT" ]; then
  echo "ERROR: Training dataset not found: $TRAIN_ROOT"
  echo "       Run build_dataset.py with configs/sparknet-410m/datasets_v1.json"
  exit 1
fi
SHARD_COUNT="$(ls -d "$TRAIN_ROOT"/shard-* 2>/dev/null | wc -l | tr -d ' ')"
if [ "$SHARD_COUNT" -eq 0 ]; then
  echo "ERROR: No shards found under $TRAIN_ROOT"
  exit 1
fi
echo "Training shards: $SHARD_COUNT"

EVAL_ROOT="$PROJECT_ROOT/datasets/sparknet-v3-pretrain-eval"
if [ ! -d "$EVAL_ROOT/shard-000" ]; then
  echo "ERROR: Eval shard not found: $EVAL_ROOT/shard-000"
  echo "       Run build_dataset.py with configs/sparknet-410m/datasets_v1_eval.json"
  exit 1
fi
echo "Eval shard: OK"

echo "Disk space for $PROJECT_ROOT:"
df -h "$PROJECT_ROOT" || true

{
  echo "Date       : $(date)"
  echo "Host       : $(hostname)"
  echo "Git commit :"
  git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo "N/A"
  echo
  echo "nvidia-smi :"
  nvidia-smi
  echo
} > "$LOG_DIR/launch_info.txt"

# --- Launch ---

python "$PROJECT_ROOT/scripts/sparknet-410m/train_pretrain.py" \
    --config "$CONFIG" \
  | tee "$LOG_DIR/train.log"

echo "======================================"
echo "SparkNet-410M v1 finished"
echo "Finished : $(date)"
echo "======================================"
