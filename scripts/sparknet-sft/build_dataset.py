import os
os.environ["HF_DATASETS_DISABLE_CACHE"] = "1"

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
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from datasets import Dataset, Features, Sequence, Value, load_dataset, disable_caching
disable_caching()

from transformers import LlamaTokenizer

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
TOKENIZER_DIR = "./tokenizer-v6"
OUTPUT_ROOT = "datasets/sft_chat_v1"

BLOCK_SIZE = 1024
SHARD_TOKENS = 50_000_000
WRITER_BATCH_SIZE = 50_000
REPORT_TOKENS_EVERY = 5_000_000

CHAT_TEMPLATE_VERSION = "sparknet_chat_v1"
ROLE_PREFIX = {
    "system": "### System:\n",
    "user": "### User:\n",
    "assistant": "### Assistant:\n",
}

MAX_ASSISTANT_TOKENS = 384
MAX_MESSAGES = 8  # short-to-medium dialogs (excludes system)

DATA_SOURCES = [
    ("HuggingFaceH4/ultrachat_200k", 0.85),
    ("OpenAssistant/oasst1", 0.15),
]

# ---------------------------------------------------------------------
# Worker tokenizer
# ---------------------------------------------------------------------
_WORKER_TOK = None
_WORKER_EOS = None
_WORKER_CFG = None


def _init_worker(tokenizer_dir: str, block_size: int, max_assistant_tokens: int):
    global _WORKER_TOK, _WORKER_EOS, _WORKER_CFG
    tok = LlamaTokenizer.from_pretrained(tokenizer_dir)
    _WORKER_TOK = tok
    _WORKER_EOS = tok.eos_token_id
    _WORKER_CFG = {
        "block_size": block_size,
        "max_assistant_tokens": max_assistant_tokens,
    }


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


def _normalize_messages(raw_messages: List[dict]) -> List[dict]:
    messages: List[dict] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = _normalize_role(
            msg.get("role") or msg.get("from") or msg.get("speaker")
        )
        content = msg.get("content") or msg.get("value") or msg.get("text")
        if role is None or not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        messages.append({"role": role, "content": content})

    if not messages:
        return []

    merged: List[dict] = []
    for msg in messages:
        if not merged or merged[-1]["role"] != msg["role"]:
            merged.append(msg)
        else:
            merged[-1]["content"] += "\n\n" + msg["content"]

    if merged and merged[0]["role"] == "system":
        system_msg = merged[0]
        rest = merged[1:]
    else:
        system_msg = None
        rest = merged

    non_system = [m for m in rest if m["role"] in {"user", "assistant"}]
    if MAX_MESSAGES is not None and len(non_system) > MAX_MESSAGES:
        non_system = non_system[-MAX_MESSAGES:]
    merged = ([system_msg] if system_msg else []) + non_system

    while merged and merged[-1]["role"] != "assistant":
        merged.pop()

    roles = {m["role"] for m in merged}
    if "assistant" not in roles or "user" not in roles:
        return []

    return merged


def _tokenize_conversation(item: Tuple[str, List[dict]]):
    source, messages = item
    if not messages:
        return None

    tok = _WORKER_TOK
    eos_id = _WORKER_EOS
    max_assistant_tokens = _WORKER_CFG["max_assistant_tokens"]

    input_ids: List[int] = []
    labels: List[int] = []
    assistant_tokens = 0

    for msg in messages:
        role = msg["role"]
        prefix = ROLE_PREFIX.get(role)
        if prefix is None:
            continue
        prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        content_ids = tok(msg["content"], add_special_tokens=False)["input_ids"]

        if role == "assistant" and max_assistant_tokens:
            content_ids = content_ids[:max_assistant_tokens]

        seg_ids = prefix_ids + content_ids + tok("\n", add_special_tokens=False)["input_ids"]
        input_ids.extend(seg_ids)

        if role == "assistant":
            labels.extend(seg_ids)
            assistant_tokens += len(seg_ids)
        else:
            labels.extend([-100] * len(seg_ids))

    if not input_ids or assistant_tokens == 0:
        return None

    if input_ids[-1] != eos_id:
        input_ids.append(eos_id)
        if messages[-1]["role"] == "assistant":
            labels.append(eos_id)
            assistant_tokens += 1
        else:
            labels.append(-100)

    block_size = _WORKER_CFG["block_size"]
    if len(input_ids) > block_size:
        input_ids = input_ids[-block_size:]
        labels = labels[-block_size:]

    return source, input_ids, labels, assistant_tokens


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


def ultrachat_stream() -> Iterator[List[dict]]:
    ds = _load_split(
        "HuggingFaceH4/ultrachat_200k",
        split_candidates=["train", "train_sft", "train_sft_filtered"],
        streaming=True,
    )
    for row in ds:
        raw_messages = None
        if isinstance(row, dict):
            raw_messages = row.get("messages") or row.get("conversation")
        if not isinstance(raw_messages, list):
            continue
        messages = _normalize_messages(raw_messages)
        if messages:
            yield messages


def _pick_best_candidate(candidates: List[dict]) -> dict:
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


def oasst_stream() -> Iterator[List[dict]]:
    ds = _load_split("OpenAssistant/oasst1", ["train"], streaming=False)

    if "messages" in ds.column_names:
        for row in ds:
            raw_messages = row.get("messages")
            if not isinstance(raw_messages, list):
                continue
            messages = _normalize_messages(raw_messages)
            if messages:
                yield messages
        return

    rows = [
        row for row in ds
        if row.get("lang") in {None, "en"}
    ]
    by_id: Dict[str, dict] = {}
    children: Dict[str, List[dict]] = defaultdict(list)
    roots: List[dict] = []

    for row in rows:
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
            if MAX_MESSAGES is not None and len(
                [m for m in msgs if m["role"] != "system"]
            ) >= MAX_MESSAGES:
                break
            next_role = "assistant" if role != "assistant" else "user"
            candidates = [
                c for c in children.get(current["message_id"], [])
                if _normalize_role(c.get("role")) == next_role
            ]
            if not candidates:
                break
            current = _pick_best_candidate(candidates)

        messages = _normalize_messages(msgs)
        if messages:
            yield messages


def conversation_stream() -> Iterator[Tuple[str, List[dict]]]:
    streams = {
        "HuggingFaceH4/ultrachat_200k": ultrachat_stream(),
        "OpenAssistant/oasst1": oasst_stream(),
    }
    iterators = {name: iter(stream) for name, stream in streams.items()}
    source_weights = {name: weight for name, weight in DATA_SOURCES}

    while True:
        source = random.choices(
            list(source_weights.keys()),
            weights=list(source_weights.values()),
        )[0]

        try:
            messages = next(iterators[source])
        except StopIteration:
            if source == "HuggingFaceH4/ultrachat_200k":
                iterators[source] = iter(ultrachat_stream())
            else:
                iterators[source] = iter(oasst_stream())
            continue

        yield source, messages


# ---------------------------------------------------------------------
# Sharded packed sequence generator
# ---------------------------------------------------------------------
def shard_generator(shard_tokens: int, shard_id: int):
    buffer_ids: List[int] = []
    buffer_labels: List[int] = []
    produced_tokens = 0
    total_seen = 0
    next_report = REPORT_TOKENS_EVERY
    start = time.perf_counter()

    source_tokens = {name: 0 for name, _ in DATA_SOURCES}

    num_workers = max(1, cpu_count() - 1)
    pool = Pool(
        num_workers,
        initializer=_init_worker,
        initargs=(TOKENIZER_DIR, BLOCK_SIZE, MAX_ASSISTANT_TOKENS),
    )

    try:
        for item in pool.imap_unordered(
            _tokenize_conversation, conversation_stream(), chunksize=4
        ):
            if not item:
                continue

            source, ids, labels, assistant_tokens = item
            total_seen += len(ids)
            source_tokens[source] += assistant_tokens

            buffer_ids.extend(ids)
            buffer_labels.extend(labels)

            while len(buffer_ids) >= BLOCK_SIZE:
                chunk_ids = buffer_ids[:BLOCK_SIZE]
                chunk_labels = buffer_labels[:BLOCK_SIZE]
                buffer_ids = buffer_ids[BLOCK_SIZE:]
                buffer_labels = buffer_labels[BLOCK_SIZE:]

                produced_tokens += BLOCK_SIZE
                yield {
                    "input_ids": chunk_ids,
                    "labels": chunk_labels,
                    "attention_mask": [1] * BLOCK_SIZE,
                }

                if produced_tokens >= shard_tokens:
                    return

            if total_seen >= next_report:
                elapsed = time.perf_counter() - start
                rate = total_seen / max(1e-6, elapsed)
                ratios = []
                for name, weight in DATA_SOURCES:
                    ratio = source_tokens[name] / max(1, sum(source_tokens.values()))
                    ratios.append(f"{name.split('/')[-1]}={ratio:.2%}")
                print(
                    f"Seen {total_seen:,} tokens | "
                    f"Emitted {produced_tokens:,}/{shard_tokens:,} | "
                    f"{rate:,.0f} tok/s | "
                    f"mix {' '.join(ratios)}"
                )
                next_report += REPORT_TOKENS_EVERY

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
    global TOKENIZER_DIR, OUTPUT_ROOT, BLOCK_SIZE, SHARD_TOKENS, MAX_ASSISTANT_TOKENS
    global MAX_MESSAGES

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--tokenizer-dir", default=TOKENIZER_DIR)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--shard-tokens", type=int, default=SHARD_TOKENS)
    parser.add_argument("--max-assistant-tokens", type=int, default=MAX_ASSISTANT_TOKENS)
    parser.add_argument("--max-messages", type=int, default=MAX_MESSAGES)
    parser.add_argument("--seed", type=int, default=42)
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

    existing = sorted(
        d for d in os.listdir(OUTPUT_ROOT) if d.startswith("shard-")
    )
    shard_id = len(existing)

    shard_dir = os.path.join(OUTPUT_ROOT, f"shard-{shard_id:03d}")
    os.makedirs(shard_dir, exist_ok=False)

    print(f"Building shard {shard_id:03d} → {shard_dir}")
    print(f"Target tokens: {SHARD_TOKENS:,}")

    features = Features({
        "input_ids": Sequence(Value("int32")),
        "labels": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
    })

    ds = Dataset.from_generator(
        lambda: shard_generator(SHARD_TOKENS, shard_id),
        features=features,
        writer_batch_size=WRITER_BATCH_SIZE,
    )
    ds.save_to_disk(shard_dir)

    tok_model_path = Path(TOKENIZER_DIR) / "tokenizer.model"
    tokenizer_hash = _hash_file(str(tok_model_path)) if tok_model_path.exists() else None

    shard_metadata = {
        "shard_id": shard_id,
        "tokenizer": "sparknet-v6",
        "tokenizer_path": TOKENIZER_DIR,
        "tokenizer_hash": tokenizer_hash,
        "block_size": BLOCK_SIZE,
        "target_tokens": SHARD_TOKENS,
        "chat_template_version": CHAT_TEMPLATE_VERSION,
        "max_assistant_tokens": MAX_ASSISTANT_TOKENS,
        "max_messages": MAX_MESSAGES,
        "data_sources": [
            {"name": name, "weight": weight}
            for name, weight in DATA_SOURCES
        ],
        "created_at": datetime.utcnow().isoformat() + "Z",
        "hostname": socket.gethostname(),
    }

    with open(os.path.join(shard_dir, "metadata.json"), "w") as f:
        json.dump(shard_metadata, f, indent=2)

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
    meta.setdefault("max_assistant_tokens", MAX_ASSISTANT_TOKENS)
    meta.setdefault("max_messages", MAX_MESSAGES)
    meta.setdefault("data_sources", shard_metadata["data_sources"])
    meta["updated_at"] = datetime.utcnow().isoformat() + "Z"
    meta["shards"] = sorted(
        list(set(meta.get("shards", [])) | {f"shard-{shard_id:03d}"}))

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Shard {shard_id:03d} complete")


if __name__ == "__main__":
    main()
