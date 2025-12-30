#!/usr/bin/env bash
set -euo pipefail

# Resolve paths robustly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="sparknet-400m-v1-resumed"

LOG_DIR="$PROJECT_ROOT/logs/$RUN_NAME"
mkdir -p "$LOG_DIR"

echo "======================================"
echo "Launching SparkNet-400M v1 pretraining"
echo "Run name: $RUN_NAME"
echo "Started at: $(date)"
echo "======================================"

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
    --resume "latest" \
  | tee "$LOG_DIR/train.log"

echo "======================================"
echo "SparkNet-400M v1 training finished"
echo "Finished at: $(date)"
echo "======================================"
