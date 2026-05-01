#!/usr/bin/env bash
# Build the sft_chat_v6 dataset.
#
# Default: 3 shards × 50 M tokens = 150 M tokens (~146 K rows at block_size=1024).
# Paired with target_tokens=450M in sft_v6.json this gives ~3 training epochs.
#
# Override any variable at the command line, e.g.:
#   NUM_SHARDS=2 bash build_datasets_v6.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NUM_SHARDS="${NUM_SHARDS:-3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/datasets/sft_chat_v6}"
SHARD_TOKENS="${SHARD_TOKENS:-}"
MAX_ASSISTANT_TOKENS="${MAX_ASSISTANT_TOKENS:-}"
MAX_MESSAGES="${MAX_MESSAGES:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
POOL_BATCH_SIZE="${POOL_BATCH_SIZE:-}"

cd "$PROJECT_ROOT"

for i in $(seq 1 "$NUM_SHARDS"); do
  echo "=== Building v6 shard $i/$NUM_SHARDS into $OUTPUT_ROOT ==="
  cmd=(
    python "$PROJECT_ROOT/scripts/sparknet-sft/build_dataset_v6.py"
    --output-root "$OUTPUT_ROOT"
    --num-shards 1
  )

  if [ -n "$SHARD_TOKENS" ]; then
    cmd+=(--shard-tokens "$SHARD_TOKENS")
  fi
  if [ -n "$MAX_ASSISTANT_TOKENS" ]; then
    cmd+=(--max-assistant-tokens "$MAX_ASSISTANT_TOKENS")
  fi
  if [ -n "$MAX_MESSAGES" ]; then
    cmd+=(--max-messages "$MAX_MESSAGES")
  fi
  if [ -n "$NUM_WORKERS" ]; then
    cmd+=(--num-workers "$NUM_WORKERS")
  fi
  if [ -n "$POOL_BATCH_SIZE" ]; then
    cmd+=(--pool-batch-size "$POOL_BATCH_SIZE")
  fi

  "${cmd[@]}"
done
