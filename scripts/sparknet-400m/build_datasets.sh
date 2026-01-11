#!/usr/bin/env bash
set -e

NUM_SHARDS=16

for i in $(seq 1 $NUM_SHARDS); do
  echo "=== Building shard $i ==="
  python scripts/sparknet-400m/build_dataset.py
done
