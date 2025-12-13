#!/usr/bin/env bash
set -e

NUM_SHARDS=8

for i in $(seq 1 $NUM_SHARDS); do
  echo "=== Building shard $i ==="
  python build_dataset_v6_sharded.py
done
