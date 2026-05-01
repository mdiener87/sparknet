#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NUM_SHARDS="${NUM_SHARDS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/datasets/sft_chat_v5}"
SHARD_TOKENS="${SHARD_TOKENS:-}"
MAX_ASSISTANT_TOKENS="${MAX_ASSISTANT_TOKENS:-}"
MAX_MESSAGES="${MAX_MESSAGES:-}"
ULTRACHAT_STREAMING="${ULTRACHAT_STREAMING:-0}"
NUM_WORKERS="${NUM_WORKERS:-}"
POOL_BATCH_SIZE="${POOL_BATCH_SIZE:-}"

cd "$PROJECT_ROOT"

for i in $(seq 1 "$NUM_SHARDS"); do
  echo "=== Building v5 shard $i/$NUM_SHARDS into $OUTPUT_ROOT ==="
  cmd=(
    python "$PROJECT_ROOT/scripts/sparknet-sft/build_dataset_v5.py"
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
  if [ "$ULTRACHAT_STREAMING" = "1" ]; then
    cmd+=(--ultrachat-streaming)
  fi
  if [ -n "$NUM_WORKERS" ]; then
    cmd+=(--num-workers "$NUM_WORKERS")
  fi
  if [ -n "$POOL_BATCH_SIZE" ]; then
    cmd+=(--pool-batch-size "$POOL_BATCH_SIZE")
  fi

  "${cmd[@]}"
done
