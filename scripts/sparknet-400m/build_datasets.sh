#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/sparknet-400m/datasets_v2.json}"
NUM_SHARDS="${2:-16}"

echo "Config: ${CONFIG_PATH}"
echo "Shards: ${NUM_SHARDS}"

for i in $(seq 1 "${NUM_SHARDS}"); do
  echo "=== Building shard ${i}/${NUM_SHARDS} ==="
  python scripts/sparknet-400m/build_dataset.py --config "${CONFIG_PATH}"
done
