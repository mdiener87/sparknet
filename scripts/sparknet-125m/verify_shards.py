#!/usr/bin/env python3
"""
verify_shards.py — fast cross-shard duplication guard for the pretraining set.

This is the detection half of the 410m-v1 fix. The hash-partitioned
build_dataset.py makes shard duplication *structurally* impossible, but the
whole point of the 410m post-mortem is that a silent data bug cost 16 days.
So we also *check*, directly on the packed output, before every run: sample
input_ids blocks from each built shard, fingerprint them, and assert that the
shards barely overlap.

Unlike analyze_v1_dedup.py (which re-streams the HF sources for an exhaustive
audit), this reads only the local Arrow shards and finishes in seconds — cheap
enough to wire into the launch preflight.

Exit status:
  0  duplication factor <= --max-dup-factor (healthy)
  1  duplication exceeds the threshold (suspicious — investigate before training)
  2  usage / shard-count / IO error

Usage:
  python verify_shards.py --root datasets/sparknet-125m-v1-pretrain --expect-shards 5
  python verify_shards.py --root datasets/sparknet-125m-v1-pretrain --sample-per-shard 50000
"""

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True,
                   help="Dataset root containing shard-* dirs (relative to repo root or absolute).")
    p.add_argument("--expect-shards", type=int, default=None,
                   help="If set, fail unless exactly this many complete shards are present.")
    p.add_argument("--sample-per-shard", type=int, default=20000, metavar="N",
                   help="Rows to fingerprint per shard, sampled evenly (default 20000).")
    p.add_argument("--max-dup-factor", type=float, default=1.01, metavar="F",
                   help="Fail if sampled_blocks / distinct_blocks exceeds F (default 1.01).")
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def is_complete_shard(path: Path) -> bool:
    return (path / "dataset_info.json").exists() and (path / "state.json").exists()


def _block_hash(input_ids) -> bytes:
    # input_ids is a torch tensor or list of ints; bytes() of the int list is
    # enough to fingerprint a fixed-length block. 16 bytes keeps the set light.
    return hashlib.sha256(bytes(int(t) & 0xFFFF for t in input_ids).hex().encode()).digest()[:16]


def sample_indices(n_rows: int, sample: int):
    if sample >= n_rows:
        return range(n_rows)
    step = n_rows / sample
    return (int(i * step) for i in range(sample))


def main() -> int:
    args = parse_args()
    root = resolve(args.root)
    if not root.is_dir():
        print(f"ERROR: dataset root not found: {root}", file=sys.stderr)
        return 2

    shards = sorted(p for p in root.glob("shard-*") if p.is_dir())
    complete = [p for p in shards if is_complete_shard(p)]
    incomplete = [p for p in shards if not is_complete_shard(p)]
    if incomplete:
        print("ERROR: incomplete shards present: " + ", ".join(p.name for p in incomplete),
              file=sys.stderr)
        return 2
    if not complete:
        print(f"ERROR: no complete shards under {root}", file=sys.stderr)
        return 2
    if args.expect_shards is not None and len(complete) != args.expect_shards:
        print(f"ERROR: expected {args.expect_shards} shards, found {len(complete)}",
              file=sys.stderr)
        return 2

    print(f"Verifying {len(complete)} shard(s) under {root}")
    print(f"Sampling up to {args.sample_per_shard:,} blocks/shard "
          f"(threshold dup factor <= {args.max_dup_factor})\n")

    global_hashes: set = set()
    per_shard_distinct = defaultdict(int)
    sampled_total = 0
    cross_shard_collisions = 0

    for shard in complete:
        ds = Dataset.load_from_disk(str(shard))
        n_rows = len(ds)
        idxs = list(sample_indices(n_rows, args.sample_per_shard))
        local: set = set()
        for i in idxs:
            h = _block_hash(ds[i]["input_ids"])
            sampled_total += 1
            if h in local:
                continue  # within-shard repeat (don't double count)
            local.add(h)
            if h in global_hashes:
                cross_shard_collisions += 1
            else:
                global_hashes.add(h)
        per_shard_distinct[shard.name] = len(local)
        print(f"  {shard.name}: rows={n_rows:,}  sampled={len(idxs):,}  distinct={len(local):,}")

    distinct_total = len(global_hashes)
    dup_factor = sampled_total / max(1, distinct_total)

    print()
    print(f"  Sampled blocks (all shards) : {sampled_total:,}")
    print(f"  Distinct blocks             : {distinct_total:,}")
    print(f"  Cross-shard collisions      : {cross_shard_collisions:,}")
    print(f"  Duplication factor          : {dup_factor:.4f}x")
    print()

    if dup_factor > args.max_dup_factor:
        print(f"FAIL: duplication factor {dup_factor:.4f} exceeds {args.max_dup_factor}. "
              f"Shards are NOT disjoint — investigate before training.", file=sys.stderr)
        return 1

    print("OK: shards are disjoint within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
