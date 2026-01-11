#!/usr/bin/env python3
"""
build_sft_chat_v1.py

Builds packed SFT chat shards (input_ids / labels / attention_mask) for SparkNet.
Key behaviors / improvements vs prior version:
- Masks assistant *prefix* tokens in labels (trains assistant content, not the scaffold).
- Caches prefix/newline token IDs per worker (big speedup).
- WORLD: Deterministic-ish seeding knobs; still nondeterministic with streaming + multiproc.
- Produces fixed-size blocks (BLOCK_SIZE) and packs multiple conversations per block.
- Writes shard-level metadata.json and dataset-level meta.json.
"""

import os
# Prefer turning off HF dataset caching; we save-to-disk shards ourselves.
os.environ.setdefault("HF_DATASETS_DISABLE_CACHE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import json
import random
import socket
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from datasets import (
    Dataset,
    Features,
    Sequence,
    Value,
    load_dataset,
    disable_caching,
)

from transformers import AutoTokenizer

disable_caching()

# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
DEFAULT_TOKENIZER_DIR = "./tokenizer-v6"
DEFAULT_OUTPUT_ROOT = "datasets/sft_chat_v1"

DEFAULT_BLOCK_SIZE = 1024
DEFAULT_SHARD_TOKENS = 50_000_000
DEFAULT_WRITER_BATCH_SIZE = 50_000
DEFAULT_REPORT_TOKENS_EVERY = 5_000_000

CHAT_TEMPLATE_VERSION = "sparknet_chat_v1"
ROLE_PREFIX = {
    "system": "### System:\n",
    "user": "### User:\n",
    "assistant": "### Assistant:\n",
}

DEFAULT_MAX_ASSISTANT_TOKENS = 384
DEFAULT_MAX_MESSAGES = 8  # excludes system

DATA_SOURCES = [
    ("HuggingFaceH4/ultrachat_200k", 0.85),
    ("OpenAssistant/oasst1", 0.15),
]

# ---------------------------------------------------------------------
# Worker globals (initialized once per process)
# ---------------------------------------------------------------------
_WORKER_TOK = None
_WORKER_EOS = None
_WORKER_CFG = None
_WORKER_PREFIX_IDS = None
_WORKER_NL_IDS = None


def _init_worker(tokenizer_dir: str, block_size: int, max_assistant_tokens: int):
    """Initialize tokenizer + cached token IDs in each multiprocessing worker."""
    global _WORKER_TOK, _WORKER_EOS, _WORKER_CFG, _WORKER_PREFIX_IDS, _WORKER_NL_IDS

    tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    # Ensure EOS exists
    if tok.eos_token_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id; required for packing boundaries.")

    _WORKER_TOK = tok
    _WORKER_EOS = tok.eos_token_id
    _WORKER_CFG = {"block_size": int(block_size), "max_assistant_tokens": int(max_assistant_tokens)}

    # Cache commonly tokenized strings
    _WORKER_PREFIX_IDS = {
        role: tok(prefix, add_special_tokens=False)["input_ids"]
        for role, prefix in ROLE_PREFIX.items()
    }
    _WORKER_NL_IDS = tok("\n", add_special_tokens=False)["input_ids"]


def _normalize_role(role: Optional[str]) -> Optional[str]:
    if role is None:
        return None
    role = role.strip().lower()
    if role in {"system"}:
        return "system"
    if role in {"user", "human", "prompter"}:
        return "user"
    if role in {"assistant", "bot", "gpt"}:
        return "assistant"
    return None


def _normalize_messages(raw_messages: List[dict], max_messages: int) -> List[dict]:
    """Normalize arbitrary message formats into {role, content} list; enforce constraints."""
    messages: List[dict] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = _normalize_role(msg.get("role") or msg.get("from") or msg.get("speaker"))
        content = msg.get("content") or msg.get("value") or msg.get("text")
        if role is None or not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        messages.append({"role": role, "content": content})

    if not messages:
        return []

    # Merge consecutive same-role messages (helps reduce fragmentation)
    merged: List[dict] = []
    for msg in messages:
        if not merged or merged[-1]["role"] != msg["role"]:
            merged.append(msg)
        else:
            merged[-1]["content"] += "\n\n" + msg["content"]

    # Extract optional system, limit other messages
    if merged and merged[0]["role"] == "system":
        system_msg = merged[0]
        rest = merged[1:]
    else:
        system_msg = None
        rest = merged

    non_system = [m for m in rest if m["role"] in {"user", "assistant"}]
    if max_messages is not None and len(non_system) > max_messages:
        non_system = non_system[-max_messages:]
    merged = ([system_msg] if system_msg else []) + non_system

    # Ensure the conversation ends with assistant (we supervise only assistant outputs)
    while merged and merged[-1]["role"] != "assistant":
        merged.pop()

    roles = {m["role"] for m in merged}
    if "assistant" not in roles or "user" not in roles:
        return []

    return merged


def _tokenize_conversation(item: Tuple[str, List[dict]]):
    """
    Returns (source, input_ids, labels, assistant_supervised_tokens) or None.

    Important behavior:
    - input_ids includes role prefixes for all roles.
    - labels are -100 except assistant *content* (+ newline + optional eos). Assistant prefix tokens are masked.
      This avoids training the model to emit "### Assistant:\n" scaffold.
    """
    source, messages = item
    if not messages:
        return None

    tok = _WORKER_TOK
    eos_id = _WORKER_EOS
    block_size = _WORKER_CFG["block_size"]
    max_assistant_tokens = _WORKER_CFG["max_assistant_tokens"]

    input_ids: List[int] = []
    labels: List[int] = []
    assistant_supervised = 0

    for msg in messages:
        role = msg["role"]
        prefix_ids = _WORKER_PREFIX_IDS.get(role)
        if prefix_ids is None:
            continue

        content_ids = tok(msg["content"], add_special_tokens=False)["input_ids"]

        # Cap assistant response length (optional)
        if role == "assistant" and max_assistant_tokens:
            content_ids = content_ids[:max_assistant_tokens]

        seg_ids = prefix_ids + content_ids + _WORKER_NL_IDS
        input_ids.extend(seg_ids)

        if role == "assistant":
            # Mask assistant scaffold; supervise only content + newline.
            labels.extend([-100] * len(prefix_ids))
            labels.extend(content_ids + _WORKER_NL_IDS)
            assistant_supervised += len(content_ids) + len(_WORKER_NL_IDS)
        else:
            labels.extend([-100] * len(seg_ids))

    if not input_ids or assistant_supervised == 0:
        return None

    # Ensure a boundary token; supervise EOS only if assistant was the last role (it is by construction).
    if input_ids[-1] != eos_id:
        input_ids.append(eos_id)
        labels.append(eos_id)
        assistant_supervised += 1

    # If too long, truncate to last block_size tokens (keeps most recent context/answer).
    if len(input_ids) > block_size:
        input_ids = input_ids[-block_size:]
        labels = labels[-block_size:]

        # If truncation removed all supervised labels, drop sample (rare, but possible)
        if all(x == -100 for x in labels):
            return None

    return source, input_ids, labels, assistant_supervised


# ---------------------------------------------------------------------
# Source streams
# ---------------------------------------------------------------------
def _load_split(name: str, split_candidates: List[str], streaming: bool):
    last_err = None
    for split in split_candidates:
        try:
            return load_dataset(name, split=split, streaming=streaming)
        except Exception as err:
            last_err = err
    if last_err:
        raise last_err
    raise RuntimeError(f"Failed to load dataset: {name}")


def ultrachat_stream(max_messages: int, streaming: bool = True) -> Iterator[List[dict]]:
    ds = _load_split(
        "HuggingFaceH4/ultrachat_200k",
        split_candidates=["train", "train_sft", "train_sft_filtered"],
        streaming=streaming,
    )
    for row in ds:
        raw_messages = None
        if isinstance(row, dict):
            raw_messages = row.get("messages") or row.get("conversation")
        if not isinstance(raw_messages, list):
            continue
        messages = _normalize_messages(raw_messages, max_messages=max_messages)
        if messages:
            yield messages


def _pick_best_candidate(candidates: List[dict]) -> dict:
    """Pick a preferred child response if ranking metadata is present."""
    if not candidates:
        raise ValueError("No candidates to pick from")
    if "rank" in candidates[0]:
        ranked = [c for c in candidates if c.get("rank") is not None]
        if ranked:
            return min(ranked, key=lambda c: c["rank"])
    if "accepted" in candidates[0]:
        accepted = [c for c in candidates if c.get("accepted") is True]
        if accepted:
            return accepted[0]
    return candidates[0]


def oasst_stream(max_messages: int, only_en: bool = True) -> Iterator[List[dict]]:
    # Note: not streaming in oasst1 because we build a tree. For scale, consider streaming + incremental indexing.
    ds = _load_split("OpenAssistant/oasst1", ["train"], streaming=False)

    if "messages" in ds.column_names:
        for row in ds:
            raw_messages = row.get("messages")
            if not isinstance(raw_messages, list):
                continue
            messages = _normalize_messages(raw_messages, max_messages=max_messages)
            if messages:
                yield messages
        return

    # Build a parent/child tree
    by_id: Dict[str, dict] = {}
    children: Dict[str, List[dict]] = defaultdict(list)
    roots: List[dict] = []

    for row in ds:
        if only_en and row.get("lang") not in {None, "en"}:
            continue
        mid = row.get("message_id")
        if not mid:
            continue
        by_id[mid] = row
        pid = row.get("parent_id")
        if pid:
            children[pid].append(row)
        else:
            roots.append(row)

    for root in roots:
        msgs: List[dict] = []
        current = root
        while current:
            role = _normalize_role(current.get("role"))
            text = current.get("text")
            if role is None or not isinstance(text, str) or not text.strip():
                break
            msgs.append({"role": role, "content": text.strip()})

            # Limit non-system messages as we build
            non_system = [m for m in msgs if m["role"] != "system"]
            if max_messages is not None and len(non_system) >= max_messages:
                break

            next_role = "assistant" if role != "assistant" else "user"
            candidates = [
                c for c in children.get(current["message_id"], [])
                if _normalize_role(c.get("role")) == next_role
            ]
            if not candidates:
                break
            current = _pick_best_candidate(candidates)

        messages = _normalize_messages(msgs, max_messages=max_messages)
        if messages:
            yield messages


def _init_source_iterators(max_messages: int, ultrachat_streaming: bool):
    streams = {
        "HuggingFaceH4/ultrachat_200k": ultrachat_stream(
            max_messages=max_messages, streaming=ultrachat_streaming
        ),
        "OpenAssistant/oasst1": oasst_stream(max_messages=max_messages, only_en=True),
    }
    return {name: iter(stream) for name, stream in streams.items()}


# ---------------------------------------------------------------------
# Sharded packed sequence generator
# ---------------------------------------------------------------------
def shard_generator(
    shard_tokens: int,
    shard_id: int,
    block_size: int,
    report_tokens_every: int,
    max_messages: int,
    ultrachat_streaming: bool,
):
    buffer_ids: List[int] = []
    buffer_labels: List[int] = []
    produced_tokens = 0
    total_seen_tokens = 0
    next_report = report_tokens_every
    start = time.perf_counter()

    source_supervised = {name: 0 for name, _ in DATA_SOURCES}
    source_weights = {name: float(weight) for name, weight in DATA_SOURCES}

    num_workers = max(1, cpu_count() - 1)
    pool = Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(TOKENIZER_DIR, block_size, MAX_ASSISTANT_TOKENS),
        maxtasksperchild=500,  # helps long runs with occasional tokenizer leaks
    )

    try:
        iterators = _init_source_iterators(max_messages=max_messages, ultrachat_streaming=ultrachat_streaming)

        def stream_for_pool() -> Iterator[Tuple[str, List[dict]]]:
            """Yield conversations while nudging toward target assistant-token mix."""
            names = list(source_weights.keys())
            weights = [source_weights[n] for n in names]
            total_supervised = 0

            while True:
                if total_supervised < 10_000:
                    source = random.choices(names, weights=weights, k=1)[0]
                else:
                    deficits = {}
                    for name, w in source_weights.items():
                        current = source_supervised[name] / max(1, total_supervised)
                        deficits[name] = w - current
                    source = max(deficits, key=deficits.get)

                try:
                    messages = next(iterators[source])
                except StopIteration:
                    if source == "HuggingFaceH4/ultrachat_200k":
                        iterators[source] = iter(
                            ultrachat_stream(max_messages=max_messages, streaming=ultrachat_streaming)
                        )
                    else:
                        iterators[source] = iter(oasst_stream(max_messages=max_messages, only_en=True))
                    continue

                yield source, messages

                # Update total_supervised using the best available running count.
                total_supervised = max(1, sum(source_supervised.values()))

        # chunksize: tune; larger reduces IPC overhead, smaller reduces tail latency variance
        for item in pool.imap_unordered(_tokenize_conversation, stream_for_pool(), chunksize=8):
            if not item:
                continue

            source, ids, labels, assistant_supervised = item
            if assistant_supervised <= 0:
                continue

            # Probabilistic downsampling to keep assistant-token ratios near target.
            total_sup = max(1, sum(source_supervised.values()))
            if total_sup >= 10_000:
                current = source_supervised[source] / total_sup
                target = source_weights.get(source, 0.0)
                if target > 0 and current > target:
                    accept_prob = max(0.1, target / current)
                    if random.random() > accept_prob:
                        continue

            total_seen_tokens += len(ids)
            source_supervised[source] += int(assistant_supervised)

            buffer_ids.extend(ids)
            buffer_labels.extend(labels)

            while len(buffer_ids) >= block_size:
                chunk_ids = buffer_ids[:block_size]
                chunk_labels = buffer_labels[:block_size]
                buffer_ids = buffer_ids[block_size:]
                buffer_labels = buffer_labels[block_size:]

                if all(x == -100 for x in chunk_labels):
                    continue

                produced_tokens += block_size
                yield {
                    "input_ids": chunk_ids,
                    "labels": chunk_labels,
                    "attention_mask": [1] * block_size,
                }

                if produced_tokens >= shard_tokens:
                    return

            if total_seen_tokens >= next_report:
                elapsed = time.perf_counter() - start
                rate = total_seen_tokens / max(1e-6, elapsed)
                total_sup = max(1, sum(source_supervised.values()))
                ratios = []
                for name, _w in DATA_SOURCES:
                    ratios.append(f"{name.split('/')[-1]}={source_supervised[name]/total_sup:.2%}")
                print(
                    f"[Shard {shard_id:03d}] Seen {total_seen_tokens:,} tok | "
                    f"Emitted {produced_tokens:,}/{shard_tokens:,} | "
                    f"{rate:,.0f} tok/s | supervised mix {' '.join(ratios)}"
                )
                next_report += report_tokens_every

    finally:
        pool.terminate()
        pool.join()


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    global TOKENIZER_DIR, OUTPUT_ROOT, BLOCK_SIZE, SHARD_TOKENS, MAX_ASSISTANT_TOKENS, MAX_MESSAGES

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--shard-tokens", type=int, default=DEFAULT_SHARD_TOKENS)
    parser.add_argument("--num-shards", type=int, default=1, help="How many shards to build in this run.")
    parser.add_argument("--max-assistant-tokens", type=int, default=DEFAULT_MAX_ASSISTANT_TOKENS)
    parser.add_argument("--max-messages", type=int, default=DEFAULT_MAX_MESSAGES)
    parser.add_argument("--writer-batch-size", type=int, default=DEFAULT_WRITER_BATCH_SIZE)
    parser.add_argument("--report-tokens-every", type=int, default=DEFAULT_REPORT_TOKENS_EVERY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ultrachat-streaming",
        action="store_true",
        help="Use streaming=True for UltraChat (faster startup, less RAM, less deterministic).",
    )
    args = parser.parse_args()

    TOKENIZER_DIR = args.tokenizer_dir
    OUTPUT_ROOT = args.output_root
    BLOCK_SIZE = args.block_size
    SHARD_TOKENS = args.shard_tokens
    MAX_ASSISTANT_TOKENS = args.max_assistant_tokens
    MAX_MESSAGES = args.max_messages

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Basic tokenizer sanity check (in main process)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
    if tok.eos_token_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id; cannot build packed shards.")
    tok_model_path = Path(TOKENIZER_DIR) / "tokenizer.model"
    tokenizer_hash = _hash_file(str(tok_model_path)) if tok_model_path.exists() else None

    # Determine next shard id from existing shard-* directories
    existing = sorted(d for d in os.listdir(OUTPUT_ROOT) if d.startswith("shard-") and os.path.isdir(os.path.join(OUTPUT_ROOT, d)))
    shard_id = len(existing)

    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "labels": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
        }
    )

    for _ in range(args.num_shards):
        shard_dir = os.path.join(OUTPUT_ROOT, f"shard-{shard_id:03d}")
        os.makedirs(shard_dir, exist_ok=False)

        print(f"Building shard {shard_id:03d} → {shard_dir}")
        print(f"Target tokens: {SHARD_TOKENS:,} | block_size={BLOCK_SIZE} | max_asst={MAX_ASSISTANT_TOKENS} | max_msgs={MAX_MESSAGES}")

        ds = Dataset.from_generator(
            lambda: shard_generator(
                shard_tokens=SHARD_TOKENS,
                shard_id=shard_id,
                block_size=BLOCK_SIZE,
                report_tokens_every=args.report_tokens_every,
                max_messages=MAX_MESSAGES,
                ultrachat_streaming=args.ultrachat_streaming,
            ),
            features=features,
            writer_batch_size=args.writer_batch_size,
        )
        ds.save_to_disk(shard_dir)

        shard_metadata = {
            "shard_id": shard_id,
            "dataset": "sft_chat_v1",
            "tokenizer": "sparknet-v6",
            "tokenizer_path": TOKENIZER_DIR,
            "tokenizer_hash": tokenizer_hash,
            "block_size": BLOCK_SIZE,
            "target_tokens": SHARD_TOKENS,
            "chat_template_version": CHAT_TEMPLATE_VERSION,
            "role_prefix": ROLE_PREFIX,
            "max_assistant_tokens": MAX_ASSISTANT_TOKENS,
            "max_messages": MAX_MESSAGES,
            "note": "Assistant prefix tokens are masked in labels; only assistant content (+newline+eos) is supervised.",
            "data_sources": [{"name": name, "weight": weight} for name, weight in DATA_SOURCES],
            "created_at": datetime.utcnow().isoformat() + "Z",
            "hostname": socket.gethostname(),
            "ultrachat_streaming": bool(args.ultrachat_streaming),
        }
        with open(os.path.join(shard_dir, "metadata.json"), "w") as f:
            json.dump(shard_metadata, f, indent=2)

        # Update dataset-level meta.json
        meta_path = os.path.join(OUTPUT_ROOT, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)

        meta.setdefault("dataset", "sft_chat_v1")
        meta.setdefault("tokenizer", "sparknet-v6")
        meta.setdefault("tokenizer_path", TOKENIZER_DIR)
        meta.setdefault("tokenizer_hash", tokenizer_hash)
        meta.setdefault("block_size", BLOCK_SIZE)
        meta.setdefault("chat_template_version", CHAT_TEMPLATE_VERSION)
        meta.setdefault("role_prefix", ROLE_PREFIX)
        meta.setdefault("max_assistant_tokens", MAX_ASSISTANT_TOKENS)
        meta.setdefault("max_messages", MAX_MESSAGES)
        meta.setdefault("data_sources", shard_metadata["data_sources"])
        meta.setdefault("labeling_note", shard_metadata["note"])
        meta["updated_at"] = datetime.utcnow().isoformat() + "Z"
        meta["shards"] = sorted(list(set(meta.get("shards", [])) | {f"shard-{shard_id:03d}"}))

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Shard {shard_id:03d} complete\n")
        shard_id += 1


if __name__ == "__main__":
    main()
