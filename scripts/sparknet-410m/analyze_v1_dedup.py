#!/usr/bin/env python3
"""
analyze_v1_dedup.py — Replay the v1 dataset build to measure document duplication.

For each completed shard-* directory under output_root, replay the exact same
source-iteration and sampling logic that build_dataset.py used — but without
tokenizing.  For every document chunk that passes _accept_for_partition, compute
a content hash and accumulate it into a global set.

What this measures
------------------
All shards start their source iterators at position 0, so the same documents
near the beginning of each HuggingFace dataset are pulled repeatedly.  That's
the duplication the training loss curve reflects.

Outputs
-------
  • Total accepted chunks across all shards (= training "steps" × block_size)
  • Distinct chunks (unique-content estimate)
  • Effective duplication factor
  • Per-source breakdown of total vs. distinct
  • Estimated unique-token count (chars ÷ 4.0 heuristic, no tokenizer needed)
  • JSON results file (optional)

Stopping condition
------------------
Each shard's replay stops after exactly text_chunks_emitted steps (read from
metadata.json), mirroring the point where shard_generator hit shard_tokens.

Usage
-----
  cd /home/mdiener/projects/sparknet

  # Full analysis (slow — streams 21× from HF datasets)
  python scripts/sparknet-410m/analyze_v1_dedup.py \\
      --config configs/sparknet-410m/datasets_v1.json \\
      --output results/v1_dedup.json

  # Quick smoke-test with 5 000 chunks per shard
  python scripts/sparknet-410m/analyze_v1_dedup.py \\
      --config configs/sparknet-410m/datasets_v1.json \\
      --limit-per-shard 5000 \\
      --shards 0-2

  # Single-shard run for debugging
  python scripts/sparknet-410m/analyze_v1_dedup.py \\
      --config configs/sparknet-410m/datasets_v1.json \\
      --shards 0 --limit-per-shard 50000
"""

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# ── repo root → sys.path so we can import build_dataset cleanly ───────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

# Heavy HF imports live inside build_dataset; this import brings them in too.
from build_dataset import (
    _accept_for_partition,
    _chatml_wrap,
    _hash_ratio,
    _should_chatml_wrap,
    _text_chunks,
    extract_texts,
    is_complete_shard,
    is_transient_error,
    load_config,
    load_source_dataset,
    source_name,
    SOURCE_RETRY_BASE_DELAY,
    SOURCE_RETRY_LIMIT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure document duplication in the v1 pretraining dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config",
        default="configs/sparknet-410m/datasets_v1.json",
        help="Path to the v1 dataset config (default: %(default)s).",
    )
    p.add_argument(
        "--shards",
        default=None,
        help=(
            "Shard IDs to analyze: range '0-19', comma list '0,3,7', "
            "or omit for all completed train shards."
        ),
    )
    p.add_argument(
        "--limit-per-shard",
        type=int,
        default=None,
        metavar="N",
        help="Stop each shard replay after N chunks (for quick tests).",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        metavar="N",
        help="Max chunks to sample for mean-length estimation (default 2000).",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write JSON results to this path.",
    )
    return p.parse_args()


def parse_shard_ids(spec: Optional[str], available: List[int]) -> List[int]:
    """Parse a shard spec like '0-19' or '0,3,7' against the available list."""
    if spec is None:
        return list(available)
    ids: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    available_set = set(available)
    missing = [i for i in ids if i not in available_set]
    if missing:
        raise ValueError(f"Requested shard IDs not found as completed train shards: {missing}")
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented iterators (no tokenization, yields (chunk, source_label))
# ─────────────────────────────────────────────────────────────────────────────

def _source_iter_instrumented(source: Dict) -> Iterator[Tuple[str, str]]:
    """
    Mirrors source_text_iterator from build_dataset.py but:
      - yields (chunk, source_label) tuples instead of raw chunks
      - skips the multiprocess tokenizer pool (read-only replay)
      - includes the same retry logic for transient network errors

    Correctness anchors:
      • _accept_for_partition called with identical arguments → same train/eval split
      • _text_chunks with same long_text_chunk_chars → same chunk boundaries
      • _should_chatml_wrap / _chatml_wrap → same text seen by the model
    """
    src_label = source_name(source)
    attempt = 0

    while True:
        try:
            dataset = load_source_dataset(source)
            row_iter = iter(dataset)
            rows_seen = 0

            while True:
                try:
                    row = next(row_iter)
                except StopIteration:
                    return  # source exhausted normally

                if not isinstance(row, dict):
                    continue
                rows_seen += 1

                for text in extract_texts(row, source):
                    stripped = text.strip()
                    if not stripped:
                        continue
                    if not _accept_for_partition(stripped, source):
                        continue
                    for chunk in _text_chunks(stripped, int(source["long_text_chunk_chars"])):
                        if _should_chatml_wrap(chunk, source):
                            yield _chatml_wrap(chunk), src_label
                        else:
                            yield chunk, src_label

        except StopIteration:
            # Propagated from the outer while-True; shouldn't happen but guard it.
            return
        except Exception as err:
            if not is_transient_error(err):
                raise
            attempt += 1
            if attempt > SOURCE_RETRY_LIMIT:
                raise RuntimeError(
                    f"Source {src_label} failed after {SOURCE_RETRY_LIMIT} retries"
                ) from err
            delay = SOURCE_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(
                f"  [Retry {attempt}/{SOURCE_RETRY_LIMIT}] {src_label}: {err}  "
                f"(sleeping {delay:.0f}s)",
                flush=True,
            )
            time.sleep(delay)


def _text_stream_instrumented(
    sources: List[Dict],
    rng: random.Random,
) -> Iterator[Tuple[str, str]]:
    """
    Mirrors text_stream from build_dataset.py.
    Uses the same rng.choices(keys, weights) sampling so the per-shard document
    interleaving is identical to what the real build produced.

    Yields (chunk, source_label).
    """
    keys = [str(i) for i in range(len(sources))]
    weights = [float(s["prob"]) for s in sources]
    iters = {k: _source_iter_instrumented(sources[int(k)]) for k in keys}

    while keys:
        key = rng.choices(keys, weights=weights, k=1)[0]
        try:
            yield next(iters[key])
        except StopIteration:
            idx = keys.index(key)
            print(f"  [SourceDone] {source_name(sources[int(key)])}", flush=True)
            keys.pop(idx)
            weights.pop(idx)
            iters.pop(key, None)


# ─────────────────────────────────────────────────────────────────────────────
# Per-chunk content hash (separate from _hash_ratio which is used for splits)
# ─────────────────────────────────────────────────────────────────────────────

def _content_hash(text: str) -> bytes:
    """
    16-byte (128-bit) content fingerprint for deduplication.
    Using 16 bytes keeps memory reasonable (~400 bytes/item in a Python set)
    while making accidental collisions astronomically unlikely across ~8M chunks.
    """
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).digest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Shard replay
# ─────────────────────────────────────────────────────────────────────────────

def replay_shard(
    shard_id: int,
    cfg: Dict,
    metadata: Dict,
    global_hashes: Dict[str, set],   # source_label -> set[bytes], mutated in place
    sample_chunks: List[str],
    sample_target: int,
    limit: Optional[int],
) -> Dict:
    """
    Replay one shard's document stream and update global_hashes.

    Returns a per-shard stats dict.
    """
    build_stats = metadata["build_stats"]
    chunks_target = int(build_stats["text_chunks_emitted"])
    if limit is not None:
        chunks_target = min(chunks_target, limit)

    per_source_total: Dict[str, int] = defaultdict(int)

    rng = random.Random(int(cfg["seed"]) + shard_id)

    print(f"\n[Shard {shard_id:03d}] target={chunks_target:,} chunks  "
          f"(metadata says {int(build_stats['text_chunks_emitted']):,})", flush=True)
    t0 = time.perf_counter()

    stream = _text_stream_instrumented(cfg["sources"], rng)
    report_every = max(1, chunks_target // 20)   # ~20 progress ticks
    n = 0

    for chunk, src_label in stream:
        h = _content_hash(chunk)
        per_source_total[src_label] += 1
        global_hashes[src_label].add(h)

        if len(sample_chunks) < sample_target:
            sample_chunks.append(chunk)

        n += 1
        if n % report_every == 0:
            elapsed = time.perf_counter() - t0
            rate = n / max(1e-6, elapsed)
            eta = (chunks_target - n) / max(1e-6, rate)
            print(
                f"  {n:>9,} / {chunks_target:,}  "
                f"({100*n/chunks_target:.0f}%)  "
                f"{rate:,.0f} chunks/s  ETA {eta:.0f}s",
                flush=True,
            )

        if n >= chunks_target:
            break

    # Explicitly close the generator so GeneratorExit propagates into
    # _source_iter_instrumented and signals the HuggingFace streaming
    # prefetch threads to stop.  Without this, those threads are still
    # alive when Python's shutdown finalizer runs, causing a
    # "PyGILState_Release: thread state must be current" crash.
    stream.close()

    elapsed = time.perf_counter() - t0
    print(f"  Done  {n:,} chunks in {elapsed:.0f}s  ({n/max(1,elapsed):,.0f} chunks/s)", flush=True)

    return {
        "shard_id": shard_id,
        "chunks_replayed": n,
        "chunks_target": chunks_target,
        "elapsed_seconds": round(elapsed, 1),
        "per_source_total": dict(per_source_total),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _discover_train_shards(output_root: Path) -> List[int]:
    """Return sorted list of shard IDs that are complete and are train partition."""
    ids = []
    for p in sorted(output_root.glob("shard-*")):
        if not p.is_dir() or not is_complete_shard(p):
            continue
        meta_path = p / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        if meta.get("partition", "train") == "train":
            ids.append(int(p.name.split("-")[1]))
    return ids


def _mean_chars(texts: List[str]) -> float:
    return sum(len(t) for t in texts) / max(1, len(texts))


def _fmt_billions(n: float) -> str:
    return f"{n/1e9:.2f}B"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg["output_root"])

    all_train_shards = _discover_train_shards(output_root)
    if not all_train_shards:
        sys.exit(f"No completed train shards found under {output_root}")

    shard_ids = parse_shard_ids(args.shards, all_train_shards)

    print("=" * 70)
    print("SparkNet v1 dataset deduplication analysis")
    print("=" * 70)
    print(f"  Config      : {args.config}")
    print(f"  Output root : {output_root}")
    print(f"  Shards      : {shard_ids[0]}–{shard_ids[-1]}  ({len(shard_ids)} total)")
    print(f"  Sources     : {len(cfg['sources'])}")
    if args.limit_per_shard:
        print(f"  Limit/shard : {args.limit_per_shard:,}  ← TEST MODE")
    print()

    # Per-source global hash sets: source_label -> set of 16-byte fingerprints
    global_hashes: Dict[str, set] = defaultdict(set)

    # Sample chunks for length estimation
    sample_chunks: List[str] = []

    all_shard_results: List[Dict] = []
    wall_t0 = time.perf_counter()

    for shard_id in shard_ids:
        shard_dir = output_root / f"shard-{shard_id:03d}"
        metadata = json.load(open(shard_dir / "metadata.json"))
        result = replay_shard(
            shard_id=shard_id,
            cfg=cfg,
            metadata=metadata,
            global_hashes=global_hashes,
            sample_chunks=sample_chunks,
            sample_target=args.sample_size,
            limit=args.limit_per_shard,
        )
        all_shard_results.append(result)

        # Running totals after each shard
        total_so_far = sum(r["chunks_replayed"] for r in all_shard_results)
        distinct_so_far = sum(len(v) for v in global_hashes.values())
        elapsed_total = time.perf_counter() - wall_t0
        remaining_shards = len(shard_ids) - len(all_shard_results)
        eta_total = (elapsed_total / len(all_shard_results)) * remaining_shards if remaining_shards else 0
        print(
            f"  Running totals: accepted={total_so_far:,}  distinct={distinct_so_far:,}  "
            f"dup={total_so_far/max(1,distinct_so_far):.2f}×  "
            f"ETA {eta_total/60:.0f}min",
            flush=True,
        )

    # ── Aggregate ────────────────────────────────────────────────────────────

    total_accepted = sum(r["chunks_replayed"] for r in all_shard_results)
    per_source_distinct: Dict[str, int] = {k: len(v) for k, v in global_hashes.items()}
    total_distinct = sum(per_source_distinct.values())

    per_source_total: Dict[str, int] = defaultdict(int)
    for r in all_shard_results:
        for src, cnt in r["per_source_total"].items():
            per_source_total[src] += cnt

    duplication_factor = total_accepted / max(1, total_distinct)

    # Length / token estimates
    mean_ch = _mean_chars(sample_chunks)
    # ~4.0 chars/token is a reasonable BPE estimate for mixed English/code text.
    # Adjust CHARS_PER_TOKEN if you want to use a different heuristic.
    CHARS_PER_TOKEN = 4.0
    mean_tok = mean_ch / CHARS_PER_TOKEN
    unique_tokens_est = total_distinct * mean_tok
    training_tokens = len(shard_ids) * int(cfg["shard_tokens"])
    effective_epochs = training_tokens / max(1, unique_tokens_est)

    total_wall = time.perf_counter() - wall_t0

    # ── Report ───────────────────────────────────────────────────────────────

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Shards analyzed              : {len(shard_ids)}")
    print(f"  Total accepted chunks        : {total_accepted:>14,}")
    print(f"  Distinct chunks (unique)     : {total_distinct:>14,}")
    print(f"  Effective duplication factor : {duplication_factor:>14.2f}×")
    print()
    print(f"  Sampled {len(sample_chunks):,} chunks for length estimation:")
    print(f"    Mean chars/chunk           : {mean_ch:>14,.1f}")
    print(f"    Chars/token (heuristic)    : {CHARS_PER_TOKEN:>14.1f}")
    print(f"    Mean tokens/chunk (est.)   : {mean_tok:>14,.0f}")
    print(f"    Est. unique tokens         : {_fmt_billions(unique_tokens_est):>14}  ({unique_tokens_est:,.0f})")
    print()
    print(f"  Training tokens total        : {_fmt_billions(training_tokens):>14}  ({training_tokens:,})")
    print(f"  Effective epochs             : {effective_epochs:>14.2f}×")
    print()
    if args.limit_per_shard:
        print("  ⚠  TEST MODE: counts above are extrapolated from a partial replay.")
        print()

    # Per-source table
    col_w = max(len(s) for s in per_source_total) + 2
    col_w = max(col_w, 52)
    header = f"  {'SOURCE':<{col_w}}  {'TOTAL':>9}  {'DISTINCT':>9}  {'DUP':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for src in sorted(per_source_total, key=lambda k: -per_source_total[k]):
        tot = per_source_total[src]
        dist = per_source_distinct.get(src, 0)
        dup = tot / max(1, dist)
        short = src if len(src) <= col_w else "…" + src[-(col_w - 1):]
        print(f"  {short:<{col_w}}  {tot:>9,}  {dist:>9,}  {dup:>5.2f}×")
    print()
    print(f"  Total wall time: {total_wall/60:.1f} min")

    # ── JSON output ──────────────────────────────────────────────────────────

    results = {
        "config": args.config,
        "shards_analyzed": shard_ids,
        "limit_per_shard": args.limit_per_shard,
        "total_accepted_chunks": total_accepted,
        "total_distinct_chunks": total_distinct,
        "duplication_factor": round(duplication_factor, 4),
        "sample_size": len(sample_chunks),
        "mean_chars_per_chunk": round(mean_ch, 1),
        "chars_per_token_heuristic": CHARS_PER_TOKEN,
        "mean_tokens_per_chunk_est": round(mean_tok, 1),
        "estimated_unique_tokens": round(unique_tokens_est),
        "training_tokens_total": training_tokens,
        "effective_epochs": round(effective_epochs, 4),
        "per_source": {
            src: {
                "total_chunks": per_source_total[src],
                "distinct_chunks": per_source_distinct.get(src, 0),
                "duplication_factor": round(
                    per_source_total[src] / max(1, per_source_distinct.get(src, 0)), 4
                ),
            }
            for src in sorted(per_source_total, key=lambda k: -per_source_total[k])
        },
        "shard_results": all_shard_results,
        "wall_time_seconds": round(total_wall, 1),
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results written → {out_path}")

    # Release the global hash sets (potentially hundreds of MB) before exit,
    # then force a GC cycle so the HuggingFace streaming iterators' finalizers
    # run while the GIL is still properly owned.  os._exit skips Python's
    # atexit / threading finalizer path entirely, which is where the
    # "PyGILState_Release: thread state must be current" crash lives.
    del global_hashes, sample_chunks, all_shard_results
    gc.collect()
    os._exit(0)


if __name__ == "__main__":
    main()
