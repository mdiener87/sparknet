#!/usr/bin/env python3
"""
build_eval_suite.py — build the per-dataset evaluation suite for sparknet-125m.

The main eval shard (datasets/sparknet-125m-v1-pretrain-eval) is a *mixed*
held-out set that drives best-checkpoint selection. It cannot be split by
source after the fact, because build_dataset.py concatenates documents across
sources before slicing into blocks — a single block straddles source
boundaries. So per-domain evaluation needs single-source eval shards built
separately, which is what this script does.

It produces, under <eval_suite_root>/:
  <slug>/shard-000   one per HF source, each packed ONLY from that source's
                     held-out (eval-partition) documents, ~per_source_tokens each
  canary/shard-000   a small curated probe set (configs/.../canary_probes.jsonl),
                     packed verbatim — a clean, low-noise human-review signal

These are loaded by train_pretrain.py as a dict eval_dataset, so the Trainer
emits eval_<slug>_loss for each. They are review-only; eval_all_loss (the mixed
set) still drives checkpoint selection.

Idempotent: a source whose shard already exists is skipped.

Usage:
  python scripts/sparknet-125m/build_eval_suite.py \
      --eval-config configs/sparknet-125m/datasets_v1_eval.json \
      --canary configs/sparknet-125m/canary_probes.jsonl
"""

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import build_dataset as bd  # noqa: E402  (heavy HF/torch imports live here)
from datasets import Dataset, Features, Sequence, Value  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

DEFAULT_EVAL_SUITE_ROOT = "datasets/sparknet-125m-v1-pretrain-eval-bysource"
DEFAULT_PER_SOURCE_TOKENS = 1_000_000

# Stable, tidy metric slugs (become eval_<slug>_loss). Falls back to the last
# path component of the source name if a source is not listed here.
SOURCE_SLUGS = {
    "HuggingFaceFW/fineweb-edu": "fineweb_edu",
    "mlfoundations/dclm-baseline-1.0": "dclm",
    "HuggingFaceFW/finepdfs": "finepdfs",
    "HuggingFaceFW/finewiki": "finewiki",
    "HuggingFaceTB/smollm-corpus": "cosmopedia",
    "kejian/codesearchnet-python-raw": "code",
    "open-web-math/open-web-math": "math",
}

FEATURES = Features({
    "input_ids":      Sequence(Value("int32")),
    "labels":         Sequence(Value("int32")),
    "attention_mask": Sequence(Value("int8")),
})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-config", default="configs/sparknet-125m/datasets_v1_eval.json")
    p.add_argument("--canary", default="configs/sparknet-125m/canary_probes.jsonl")
    p.add_argument("--eval-suite-root", default=DEFAULT_EVAL_SUITE_ROOT)
    p.add_argument("--per-source-tokens", type=int, default=DEFAULT_PER_SOURCE_TOKENS)
    return p.parse_args()


def slug_for(source: dict) -> str:
    name = str(source.get("name", "json"))
    if name in SOURCE_SLUGS:
        return SOURCE_SLUGS[name]
    return name.split("/")[-1].replace("-", "_").replace(".", "_")


def write_shard(ds: Dataset, out_dir: Path, metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "shard-000"
    ds.save_to_disk(str(shard_dir))
    with open(shard_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def build_source_eval(base_cfg: dict, source: dict, out_dir: Path, per_source_tokens: int) -> None:
    """Build a single-source held-out eval shard via build_dataset.shard_generator."""
    cfg = dict(base_cfg)
    cfg["sources"] = [source]
    cfg["output_root"] = str(out_dir)
    cfg["shard_tokens"] = int(per_source_tokens)
    cfg["report_tokens_every"] = max(1, int(per_source_tokens) // 4)

    ds = Dataset.from_generator(
        lambda: bd.shard_generator(cfg, 0),
        features=FEATURES,
        writer_batch_size=int(cfg["writer_batch_size"]),
    )
    write_shard(ds, out_dir, {
        "kind": "per_source_eval",
        "source": source.get("name"),
        "config_name": source.get("config_name"),
        "partition": "eval",
        "holdout_rate": float(cfg["holdout_rate"]),
        "holdout_salt": cfg["holdout_salt"],
        "block_size": int(cfg["block_size"]),
        "target_tokens": int(per_source_tokens),
        "build_stats": cfg.get("_build_stats", {}),
        "seed": int(cfg["seed"]),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
    })


def build_canary(tokenizer_path: str, jsonl_path: Path, out_dir: Path, block_size: int) -> int:
    """Pack the curated probe set into fixed-size blocks.

    Each probe is encoded as [BOS] ids [EOS] (matching training), concatenated,
    and sliced into block_size blocks. The trailing partial block is padded with
    EOS and its label positions set to -100 so the pad tokens do not contribute
    to the loss — this keeps every curated token in the signal without diluting
    perplexity with padding.
    """
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    tok.model_max_length = int(1e30)
    bos, eos = tok.bos_token_id, tok.eos_token_id

    buffer: list[int] = []
    n_docs = 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get("text", "").strip()
            if not text:
                continue
            ids = tok.encode(text, add_special_tokens=False)
            if not ids:
                continue
            buffer.extend([bos] + ids + [eos])
            n_docs += 1

    rows = []
    for start in range(0, len(buffer), block_size):
        block = buffer[start:start + block_size]
        pad = block_size - len(block)
        if pad:
            labels = block + [-100] * pad
            attn = [1] * len(block) + [0] * pad
            block = block + [eos] * pad
        else:
            labels = list(block)
            attn = [1] * block_size
        rows.append({"input_ids": block, "labels": labels, "attention_mask": attn})

    ds = Dataset.from_list(rows, features=FEATURES)
    write_shard(ds, out_dir, {
        "kind": "canary_probe",
        "source_file": str(jsonl_path),
        "n_docs": n_docs,
        "n_blocks": len(rows),
        "n_tokens": len(buffer),
        "block_size": block_size,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
    })
    return len(rows)


def main() -> int:
    args = parse_args()
    base_cfg = bd.load_config(args.eval_config)
    if base_cfg["partition"] != "eval":
        print(f"ERROR: {args.eval_config} must be an eval-partition config.", file=sys.stderr)
        return 2

    suite_root = bd.resolve_repo_path(args.eval_suite_root)
    canary_path = bd.resolve_repo_path(args.canary)
    block_size = int(base_cfg["block_size"])

    print(f"Eval suite root : {suite_root}")
    print(f"Per-source target: {args.per_source_tokens:,} tokens")
    print(f"Sources          : {len(base_cfg['sources'])}\n")

    for source in base_cfg["sources"]:
        slug = slug_for(source)
        out_dir = suite_root / slug
        if bd.is_complete_shard(out_dir / "shard-000"):
            print(f"[skip] {slug}: already built")
            continue
        print(f"[build] {slug}: {bd.source_name(source)}")
        t0 = time.perf_counter()
        build_source_eval(base_cfg, source, out_dir, args.per_source_tokens)
        print(f"[done] {slug} in {time.perf_counter() - t0:.0f}s\n")

    # Canary probe set
    canary_dir = suite_root / "canary"
    if bd.is_complete_shard(canary_dir / "shard-000"):
        print("[skip] canary: already built")
    else:
        print(f"[build] canary: {canary_path}")
        n_blocks = build_canary(base_cfg["tokenizer_path"], canary_path, canary_dir, block_size)
        print(f"[done] canary: {n_blocks} blocks\n")

    print("Eval suite complete. Subsets:")
    for sub in sorted(suite_root.glob("*")):
        if (sub / "shard-000").is_dir():
            print(f"  - {sub.name}")
    return 0


if __name__ == "__main__":
    rc = main()
    # HuggingFace streaming iterators leave prefetch threads whose finalizers
    # crash with "PyGILState_Release: thread state must be current" during the
    # normal interpreter shutdown. The work is already done and flushed, so skip
    # Python's finalizer path entirely (same workaround as analyze_v1_dedup.py).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
