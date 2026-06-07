#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/sparknet-125m/datasets_v1.json}"
NUM_SHARDS="${2:-5}"

echo "Config: ${CONFIG_PATH}"
echo "Shards: ${NUM_SHARDS}"

for i in $(seq 1 "${NUM_SHARDS}"); do
  echo "=== Building shard ${i}/${NUM_SHARDS} ==="
  python scripts/sparknet-125m/build_dataset.py --config "${CONFIG_PATH}"
done
