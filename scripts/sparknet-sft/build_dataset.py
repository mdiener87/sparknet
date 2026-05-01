#!/usr/bin/env python3
"""
build_sft_chat.py

Builds fixed-length SFT chat shards (input_ids / labels / attention_mask) for SparkNet.

Key fix vs v2:
- If a conversation is too long, DO NOT token-truncate the tail.
  Instead drop oldest user+assistant turns and re-tokenize until it fits.
  This keeps chat conditioning stable and aligns with inference prompts.

Other:
- One conversation per row, right-padded to BLOCK_SIZE.
- Assistant prefix tokens masked in labels; supervise assistant content (+newline+eos).
- Uses tokenizer pad_token_id (no hardcoded pad id).
- Writes shard metadata.json + dataset meta.json.
"""

import os
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
from datasets import Dataset, Features, Sequence, Value, load_dataset, disable_caching
from transformers import LlamaTokenizer

disable_caching()

DEFAULT_TOKENIZER_DIR = "./tokenizer-v6"
DEFAULT_OUTPUT_ROOT = "datasets/sft_chat_v4"

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
    ("HuggingFaceH4/ultrachat_200k", 0.75),
    ("OpenAssistant/oasst1", 0.25),
]

# Worker globals
_WORKER_TOK = None
_WORKER_EOS = None
_WORKER_PAD = None
_WORKER_CFG = None
_WORKER_PREFIX_IDS = None
_WORKER_NL_IDS = None


def load_sparknet_tokenizer(tokenizer_path: str):
    path = Path(tokenizer_path).expanduser()
    model_path = path / "tokenizer.model" if path.is_dir() else path
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")

    tok = LlamaTokenizer(vocab_file=str(model_path), legacy=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok

def _init_worker(tokenizer_dir: str, block_size: int, max_assistant_tokens: int):
    global _WORKER_TOK, _WORKER_EOS, _WORKER_PAD, _WORKER_CFG, _WORKER_PREFIX_IDS, _WORKER_NL_IDS
    tok = load_sparknet_tokenizer(tokenizer_dir)
    if tok.eos_token_id is None:
        raise RuntimeError("Tokenizer missing eos_token_id.")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    _WORKER_TOK = tok
    _WORKER_EOS = tok.eos_token_id
    _WORKER_PAD = tok.pad_token_id
    _WORKER_CFG = {"block_size": int(block_size), "max_assistant_tokens": int(max_assistant_tokens)}

    _WORKER_PREFIX_IDS = {role: tok(prefix, add_special_tokens=False)["input_ids"]
                          for role, prefix in ROLE_PREFIX.items()}
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

    # merge consecutive same-role
    merged: List[dict] = []
    for m in messages:
        if not merged or merged[-1]["role"] != m["role"]:
            merged.append(m)
        else:
            merged[-1]["content"] += "\n\n" + m["content"]

    # keep optional system, cap others
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

    # ensure ends on assistant
    while merged and merged[-1]["role"] != "assistant":
        merged.pop()

    roles = {m["role"] for m in merged}
    if "assistant" not in roles or "user" not in roles:
        return []

    return merged

def _drop_oldest_turn(messages: List[dict]) -> bool:
    """Drop oldest user turn + its following assistant turn (keeps optional system)."""
    if not messages:
        return False
    start_idx = 1 if messages[0]["role"] == "system" else 0
    if start_idx >= len(messages):
        return False

    # drop oldest (should be user, but be tolerant)
    messages.pop(start_idx)

    # if next is assistant, drop it too
    if start_idx < len(messages) and messages[start_idx]["role"] == "assistant":
        messages.pop(start_idx)
    return True

def _tokenize_messages(messages: List[dict]) -> Tuple[List[int], List[int], int]:
    """Tokenize full conversation; returns (input_ids, labels, supervised_tokens)."""
    tok = _WORKER_TOK
    eos_id = _WORKER_EOS
    max_asst = _WORKER_CFG["max_assistant_tokens"]

    input_ids: List[int] = []
    labels: List[int] = []
    supervised = 0

    for msg in messages:
        role = msg["role"]
        prefix_ids = _WORKER_PREFIX_IDS.get(role)
        if prefix_ids is None:
            continue

        content_ids = tok(msg["content"], add_special_tokens=False)["input_ids"]
        if role == "assistant" and max_asst:
            content_ids = content_ids[:max_asst]

        seg_ids = prefix_ids + content_ids + _WORKER_NL_IDS
        input_ids.extend(seg_ids)

        if role == "assistant":
            labels.extend([-100] * len(prefix_ids))
            labels.extend(content_ids + _WORKER_NL_IDS)
            supervised += len(content_ids) + len(_WORKER_NL_IDS)
        else:
            labels.extend([-100] * len(seg_ids))

    if not input_ids or supervised == 0:
        return [], [], 0

    if input_ids[-1] != eos_id:
        input_ids.append(eos_id)
        labels.append(eos_id)
        supervised += 1

    return input_ids, labels, supervised

def _tokenize_conversation(item: Tuple[str, List[dict]]):
    """
    One conversation => one row.
    If too long, drop oldest turns until it fits (no token-tail truncation).
    """
    source, messages = item
    if not messages:
        return None

    block_size = _WORKER_CFG["block_size"]
    pad_id = _WORKER_PAD

    msgs = list(messages)

    # drop oldest turns until tokenized length fits
    while True:
        ids, labels, supervised = _tokenize_messages(msgs)
        if supervised == 0:
            return None
        if len(ids) <= block_size:
            break
        if not _drop_oldest_turn(msgs):
            # can't drop anything else; give up
            return None

    seq_len = len(ids)
    if seq_len < block_size:
        pad_len = block_size - seq_len
        input_ids = ids + [pad_id] * pad_len
        labels_out = labels + [-100] * pad_len
        attn = [1] * seq_len + [0] * pad_len
    else:
        input_ids = ids
        labels_out = labels
        attn = [1] * block_size

    # Must not start with supervised labels (should be true now)
    if any(l != -100 for l in labels_out[:32]):
        return None

    return source, input_ids, labels_out, attn, supervised

def _load_split(name: str, split_candidates: List[str], streaming: bool):
    last_err = None
    for split in split_candidates:
        try:
            return load_dataset(name, split=split, streaming=streaming)
        except Exception as err:
            last_err = err
    raise last_err or RuntimeError(f"Failed to load dataset: {name}")

def ultrachat_stream(max_messages: int, streaming: bool = True) -> Iterator[List[dict]]:
    ds = _load_split(
        "HuggingFaceH4/ultrachat_200k",
        ["train", "train_sft", "train_sft_filtered"],
        streaming=streaming,
    )
    for row in ds:
        raw = None
        if isinstance(row, dict):
            raw = row.get("messages") or row.get("conversation")
        if not isinstance(raw, list):
            continue
        msgs = _normalize_messages(raw, max_messages=max_messages)
        if msgs:
            yield msgs

def _pick_best_candidate(candidates: List[dict]) -> dict:
    if not candidates:
        raise ValueError("No candidates")
    if "rank" in candidates[0]:
        ranked = [c for c in candidates if c.get("rank") is not None]
        if ranked:
            return min(ranked, key=lambda c: c["rank"])
    if "accepted" in candidates[0]:
        acc = [c for c in candidates if c.get("accepted") is True]
        if acc:
            return acc[0]
    return candidates[0]

def oasst_stream(max_messages: int, only_en: bool = True) -> Iterator[List[dict]]:
    ds = _load_split("OpenAssistant/oasst1", ["train"], streaming=False)

    if "messages" in ds.column_names:
        for row in ds:
            raw = row.get("messages")
            if not isinstance(raw, list):
                continue
            msgs = _normalize_messages(raw, max_messages=max_messages)
            if msgs:
                yield msgs
        return

    children: Dict[str, List[dict]] = defaultdict(list)
    roots: List[dict] = []

    for row in ds:
        if only_en and row.get("lang") not in {None, "en"}:
            continue
        mid = row.get("message_id")
        if not mid:
            continue
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

            non_system = [m for m in msgs if m["role"] != "system"]
            if max_messages is not None and len(non_system) >= max_messages:
                break

            next_role = "assistant" if role != "assistant" else "user"
            cands = [c for c in children.get(current["message_id"], [])
                     if _normalize_role(c.get("role")) == next_role]
            if not cands:
                break
            current = _pick_best_candidate(cands)

        norm = _normalize_messages(msgs, max_messages=max_messages)
        if norm:
            yield norm

def _init_source_iterators(max_messages: int, ultrachat_streaming: bool):
    streams = {
        "HuggingFaceH4/ultrachat_200k": ultrachat_stream(max_messages=max_messages, streaming=ultrachat_streaming),
        "OpenAssistant/oasst1": oasst_stream(max_messages=max_messages, only_en=True),
    }
    return {name: iter(stream) for name, stream in streams.items()}

def shard_generator(
    shard_tokens: int,
    shard_id: int,
    block_size: int,
    report_tokens_every: int,
    max_messages: int,
    ultrachat_streaming: bool,
    seed: int,
):
    produced_tokens = 0
    total_seen_tokens = 0
    next_report = report_tokens_every
    start = time.perf_counter()

    rng = random.Random(seed + shard_id)

    source_weights = {name: float(weight) for name, weight in DATA_SOURCES}
    names = list(source_weights.keys())
    weights = [source_weights[n] for n in names]

    num_workers = max(1, cpu_count() - 1)
    pool = Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(TOKENIZER_DIR, block_size, MAX_ASSISTANT_TOKENS),
        maxtasksperchild=500,
    )

    try:
        iters = _init_source_iterators(max_messages=max_messages, ultrachat_streaming=ultrachat_streaming)

        def stream_for_pool():
            while True:
                src = rng.choices(names, weights=weights, k=1)[0]
                try:
                    msgs = next(iters[src])
                except StopIteration:
                    if src == "HuggingFaceH4/ultrachat_200k":
                        iters[src] = iter(ultrachat_stream(max_messages=max_messages, streaming=ultrachat_streaming))
                    else:
                        iters[src] = iter(oasst_stream(max_messages=max_messages, only_en=True))
                    continue
                yield src, msgs

        for item in pool.imap_unordered(_tokenize_conversation, stream_for_pool(), chunksize=16):
            if not item:
                continue
            source, input_ids, labels, attn, supervised = item

            total_seen_tokens += sum(1 for _ in input_ids)  # == block_size
            produced_tokens += block_size

            yield {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

            if produced_tokens >= shard_tokens:
                return

            if produced_tokens >= next_report:
                elapsed = time.perf_counter() - start
                rate = produced_tokens / max(1e-6, elapsed)
                print(
                    f"[Shard {shard_id:03d}] Emitted {produced_tokens:,}/{shard_tokens:,} tok | "
                    f"{rate:,.0f} tok/s"
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

def main():
    global TOKENIZER_DIR, OUTPUT_ROOT, BLOCK_SIZE, SHARD_TOKENS, MAX_ASSISTANT_TOKENS, MAX_MESSAGES

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--shard-tokens", type=int, default=DEFAULT_SHARD_TOKENS)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-assistant-tokens", type=int, default=DEFAULT_MAX_ASSISTANT_TOKENS)
    parser.add_argument("--max-messages", type=int, default=DEFAULT_MAX_MESSAGES)
    parser.add_argument("--writer-batch-size", type=int, default=DEFAULT_WRITER_BATCH_SIZE)
    parser.add_argument("--report-tokens-every", type=int, default=DEFAULT_REPORT_TOKENS_EVERY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ultrachat-streaming", action="store_true")
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

    tok = load_sparknet_tokenizer(TOKENIZER_DIR)
    if tok.eos_token_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id.")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"[Tok] vocab={len(tok)} eos={tok.eos_token_id} pad={tok.pad_token_id} bos={tok.bos_token_id}")

    tok_model_path = Path(TOKENIZER_DIR) / "tokenizer.model"
    tokenizer_hash = _hash_file(str(tok_model_path)) if tok_model_path.exists() else None

    existing = sorted(
        d for d in os.listdir(OUTPUT_ROOT)
        if d.startswith("shard-") and os.path.isdir(os.path.join(OUTPUT_ROOT, d))
    )
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
                seed=args.seed,
            ),
            features=features,
            writer_batch_size=args.writer_batch_size,
        )
        ds.save_to_disk(shard_dir)

        shard_metadata = {
            "shard_id": shard_id,
            "dataset": "sft_chat_v4",
            "tokenizer": "sparknet-v6",
            "tokenizer_path": TOKENIZER_DIR,
            "tokenizer_hash": tokenizer_hash,
            "block_size": BLOCK_SIZE,
            "target_tokens": SHARD_TOKENS,
            "rows_expected": int(SHARD_TOKENS // BLOCK_SIZE),
            "chat_template_version": CHAT_TEMPLATE_VERSION,
            "role_prefix": ROLE_PREFIX,
            "max_assistant_tokens": MAX_ASSISTANT_TOKENS,
            "max_messages": MAX_MESSAGES,
            "note": "One conversation per row. If too long, drop oldest turns until it fits (no token-tail truncation). Assistant prefix masked; supervise assistant content (+newline+eos). Right-pad to block_size.",
            "data_sources": [{"name": name, "weight": weight} for name, weight in DATA_SOURCES],
            "created_at": datetime.utcnow().isoformat() + "Z",
            "hostname": socket.gethostname(),
            "ultrachat_streaming": bool(args.ultrachat_streaming),
            "seed": args.seed,
        }
        with open(os.path.join(shard_dir, "metadata.json"), "w") as f:
            json.dump(shard_metadata, f, indent=2)

        meta_path = os.path.join(OUTPUT_ROOT, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)

        meta.setdefault("dataset", "sft_chat_v4")
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
