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

TOKENIZER_DIR="$PROJECT_ROOT/tokenizer-v8"
if [ ! -f "$TOKENIZER_DIR/tokenizer.json" ]; then
  echo "ERROR: tokenizer-v8 not found at $TOKENIZER_DIR"
  echo "       Run: python scripts/sparknet-410m/build_tokenizer.py"
  exit 1
fi

CONFIG_TOKENIZER="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokenizer_path"])' "$CONFIG")"
CONFIG_TRAIN_ROOT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["train_root"])' "$CONFIG")"
CONFIG_EVAL_ROOT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["eval_root"])' "$CONFIG")"
if [ "$CONFIG_TOKENIZER" != "./tokenizer-v8" ]; then
  echo "ERROR: Config tokenizer_path must be ./tokenizer-v8, got $CONFIG_TOKENIZER"
  exit 1
fi
if [ "$CONFIG_TRAIN_ROOT" != "datasets/sparknet-410m-v1-pretrain" ]; then
  echo "ERROR: Config train_root must be datasets/sparknet-410m-v1-pretrain, got $CONFIG_TRAIN_ROOT"
  exit 1
fi
if [ "$CONFIG_EVAL_ROOT" != "datasets/sparknet-410m-v1-pretrain-eval" ]; then
  echo "ERROR: Config eval_root must be datasets/sparknet-410m-v1-pretrain-eval, got $CONFIG_EVAL_ROOT"
  exit 1
fi

TRAIN_ROOT="$PROJECT_ROOT/datasets/sparknet-410m-v1-pretrain"
if [ ! -d "$TRAIN_ROOT" ]; then
  echo "ERROR: Training dataset not found: $TRAIN_ROOT"
  echo "       Run build_dataset.py with configs/sparknet-410m/datasets_v1.json"
  exit 1
fi
SHARD_COUNT=0
for shard in "$TRAIN_ROOT"/shard-*; do
  [ -d "$shard" ] || continue
  if [ -f "$shard/dataset_info.json" ] && [ -f "$shard/state.json" ]; then
    SHARD_COUNT=$((SHARD_COUNT + 1))
  else
    echo "ERROR: Incomplete train shard: $shard"
    exit 1
  fi
done
if [ "$SHARD_COUNT" -ne 21 ]; then
  echo "ERROR: Expected exactly 21 complete train shards under $TRAIN_ROOT, found $SHARD_COUNT"
  exit 1
fi
echo "Training shards: $SHARD_COUNT"

EVAL_ROOT="$PROJECT_ROOT/datasets/sparknet-410m-v1-pretrain-eval"
if [ ! -f "$EVAL_ROOT/shard-000/dataset_info.json" ] || [ ! -f "$EVAL_ROOT/shard-000/state.json" ]; then
  echo "ERROR: Eval shard not found: $EVAL_ROOT/shard-000"
  echo "       Run build_dataset.py with configs/sparknet-410m/datasets_v1_eval.json"
  exit 1
fi
echo "Eval shard: OK"

echo "Disk space for $PROJECT_ROOT:"
df -h "$PROJECT_ROOT" || true
AVAIL_KB="$(df -Pk "$PROJECT_ROOT" | awk 'NR==2 {print $4}')"
MIN_AVAIL_KB=209715200
if [ "$AVAIL_KB" -lt "$MIN_AVAIL_KB" ]; then
  echo "ERROR: Less than 200 GiB free under $PROJECT_ROOT"
  exit 1
fi

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
