#!/usr/bin/env bash
set -e

NUM_SHARDS=4

for i in $(seq 1 $NUM_SHARDS); do
  echo "=== Building shard $i ==="
  python scripts/sparknet-sft/build_dataset.py
done
