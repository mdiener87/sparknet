#!/usr/bin/env bash
set -e

# Absolute path to this script’s directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Project root = one level up
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUNS=(
  "tput_gc_on_b8x64 8 64"
  "tput_gc_off_b8x64 8 64 --no-grad-checkpointing"
  "tput_gc_off_b16x32 16 32 --no-grad-checkpointing"
  "tput_gc_off_b32x16 32 16 --no-grad-checkpointing"
)

for r in "${RUNS[@]}"; do
  set -- $r
  NAME=$1
  BS=$2
  GA=$3
  EXTRA=${@:4}

  echo "=============================="
  echo "Starting $NAME"
  echo "=============================="

  python "$PROJECT_ROOT/scripts/sparknet-400m/train_throughput_test.py" \
    --run-name $NAME \
    --batch-size $BS \
    --grad-accum $GA \
    --target-tokens 80000000 \
    --no-eval \
    $EXTRA | tee "$PROJECT_ROOT/logs/${NAME}.log"
done
