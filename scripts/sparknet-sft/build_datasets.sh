#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NUM_SHARDS="${NUM_SHARDS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/datasets/sft_chat_v4}"

cd "$PROJECT_ROOT"

for i in $(seq 1 "$NUM_SHARDS"); do
  echo "=== Building shard $i/$NUM_SHARDS into $OUTPUT_ROOT ==="
  python "$PROJECT_ROOT/scripts/sparknet-sft/build_dataset.py" \
    --output-root "$OUTPUT_ROOT" \
    --num-shards 1
done
