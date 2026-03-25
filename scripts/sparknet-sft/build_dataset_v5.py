#!/usr/bin/env python3
"""
build_dataset_v5.py

Build a narrower SFT dataset for SparkNet-400M v5.

The v5 goal is not "general assistant for everything". It is a smaller, cleaner
chatbot distribution with short helpful answers, simple plans, rewrites, and
cautious uncertainty behavior.

Compared with v4, this builder adds:
- Stronger filtering for code-like / JSON-like / URL-heavy outputs.
- Repetition and placeholder spam checks.
- Shorter assistant targets.
- Fewer turns per conversation.
"""

import os

os.environ.setdefault("HF_DATASETS_DISABLE_CACHE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import json
import random
import re
import socket
import shutil
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from datasets import Dataset, Features, Sequence, Value, disable_caching, load_dataset
from transformers import LlamaTokenizer

disable_caching()

DEFAULT_TOKENIZER_DIR = "./tokenizer-v6"
DEFAULT_OUTPUT_ROOT = "datasets/sft_chat_v5"

DEFAULT_BLOCK_SIZE = 1024
DEFAULT_SHARD_TOKENS = 50_000_000
DEFAULT_WRITER_BATCH_SIZE = 50_000
DEFAULT_REPORT_TOKENS_EVERY = 5_000_000
DEFAULT_NUM_WORKERS = max(1, cpu_count() - 1)
DEFAULT_POOL_BATCH_SIZE = max(64, DEFAULT_NUM_WORKERS * 8)

CHAT_TEMPLATE_VERSION = "sparknet_chat_v1"
ROLE_PREFIX = {
    "system": "### System:\n",
    "user": "### User:\n",
    "assistant": "### Assistant:\n",
}

DEFAULT_MAX_ASSISTANT_TOKENS = 192
DEFAULT_MAX_MESSAGES = 6  # excludes system

DATA_SOURCES = [
    ("HuggingFaceH4/ultrachat_200k", 0.60),
    ("OpenAssistant/oasst1", 0.40),
]

CODE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"```",
        r"\bimport\s+[a-z0-9_.]+",
        r"\bfrom\s+[a-z0-9_.]+\s+import\b",
        r"\bdef\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(",
        r"\bclass\s+[A-Z][a-zA-Z0-9_]*\b",
        r"\bpublic\s+class\b",
        r"\bconsole\.log\s*\(",
        r"<(?:!DOCTYPE|html|body|script)\b",
        r"#include\s*<",
        r"\bSELECT\b.+\bFROM\b",
        r"\bfunction\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(",
        r"\breturn\s+\{",
        r"\bTraceback\b",
        r"\bException\b",
    ]
]

PROGRAMMING_REQUEST_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"```",
        r"\bwrite\s+(?:a|an|the)?\s*(?:function|script|program|class)\b",
        r"\bdebug\b.+\b(?:code|script|function|program|query)\b",
        r"\bjson\s+object\b",
        r"\breturn\s+json\b",
        r"\bstack\s+trace\b",
        r"\btraceback\b",
        r"\bregex\b",
        r"\bapi\b.+\bendpoint\b",
    ]
]

PLACEHOLDER_RE = re.compile(r"\[[^\]]{1,40}\]")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z']+")


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

    _WORKER_PREFIX_IDS = {
        role: tok(prefix, add_special_tokens=False)["input_ids"] for role, prefix in ROLE_PREFIX.items()
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


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _looks_like_code(text: str) -> bool:
    if any(pattern.search(text) for pattern in CODE_PATTERNS):
        return True
    symbol_count = sum(1 for ch in text if ch in "{}[]();<>`\\")
    return symbol_count >= 10 and (symbol_count / max(1, len(text))) > 0.06


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return True
    if stripped.startswith("[") and stripped.endswith("]"):
        return True
    if re.search(r'"[A-Za-z0-9_ -]+"\s*:\s*', stripped):
        return True
    return False


def _has_bad_repetition(text: str) -> bool:
    words = [w.lower() for w in WORD_RE.findall(text)]
    if len(words) < 24:
        return False

    trigrams = list(zip(words, words[1:], words[2:]))
    if trigrams:
        unique_ratio = len(set(trigrams)) / len(trigrams)
        if unique_ratio < 0.55:
            return True

    repeated_bigram_run = 0
    last_bigram = None
    for bigram in zip(words, words[1:]):
        if bigram == last_bigram:
            repeated_bigram_run += 1
            if repeated_bigram_run >= 3:
                return True
        else:
            repeated_bigram_run = 0
            last_bigram = bigram

    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if len(lines) >= 4 and len(set(lines)) <= len(lines) * 0.6:
        return True
    return False


def _looks_like_placeholder_spam(text: str) -> bool:
    return len(PLACEHOLDER_RE.findall(text)) >= 2


def _looks_like_form_letter(text: str) -> bool:
    lower = text.lower()
    return lower.startswith("dear ") or "i hope this message finds you well" in lower


def _is_programming_request(text: str) -> bool:
    return any(pattern.search(text) for pattern in PROGRAMMING_REQUEST_PATTERNS)


def _assistant_message_is_usable(text: str) -> bool:
    wc = _word_count(text)
    if wc < 4:
        return False
    if wc > 140:
        return False
    if URL_RE.search(text):
        return False
    if _looks_like_code(text) or _looks_like_json(text):
        return False
    if _has_bad_repetition(text):
        return False
    if _looks_like_placeholder_spam(text) and wc > 40:
        return False
    if _looks_like_form_letter(text) and wc > 90:
        return False
    return True


def _conversation_is_usable(messages: List[dict]) -> bool:
    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    user_messages = [m for m in messages if m["role"] == "user"]

    if not assistant_messages or not user_messages:
        return False

    for msg in user_messages:
        if _is_programming_request(msg["content"]):
            return False

    for msg in assistant_messages:
        if not _assistant_message_is_usable(msg["content"]):
            return False

    last_assistant = assistant_messages[-1]["content"]
    if _word_count(last_assistant) < 8:
        return False

    return True


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

    merged: List[dict] = []
    for message in messages:
        if not merged or merged[-1]["role"] != message["role"]:
            merged.append(message)
        else:
            merged[-1]["content"] += "\n\n" + message["content"]

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

    while merged and merged[-1]["role"] != "assistant":
        merged.pop()

    roles = {m["role"] for m in merged}
    if "assistant" not in roles or "user" not in roles:
        return []

    if not _conversation_is_usable(merged):
        return []

    return merged


def _drop_oldest_turn(messages: List[dict]) -> bool:
    if not messages:
        return False
    start_idx = 1 if messages[0]["role"] == "system" else 0
    if start_idx >= len(messages):
        return False

    messages.pop(start_idx)
    if start_idx < len(messages) and messages[start_idx]["role"] == "assistant":
        messages.pop(start_idx)
    return True


def _tokenize_messages(messages: List[dict]) -> Tuple[List[int], List[int], int]:
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
    source, messages = item
    if not messages:
        return None

    block_size = _WORKER_CFG["block_size"]
    pad_id = _WORKER_PAD
    msgs = list(messages)

    while True:
        ids, labels, supervised = _tokenize_messages(msgs)
        if supervised == 0:
            return None
        if len(ids) <= block_size:
            break
        if not _drop_oldest_turn(msgs):
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

    if any(label != -100 for label in labels_out[:32]):
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
        ranked = [candidate for candidate in candidates if candidate.get("rank") is not None]
        if ranked:
            return min(ranked, key=lambda candidate: candidate["rank"])
    if "accepted" in candidates[0]:
        accepted = [candidate for candidate in candidates if candidate.get("accepted") is True]
        if accepted:
            return accepted[0]
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
            candidates = [
                candidate
                for candidate in children.get(current["message_id"], [])
                if _normalize_role(candidate.get("role")) == next_role
            ]
            if not candidates:
                break
            current = _pick_best_candidate(candidates)

        norm = _normalize_messages(msgs, max_messages=max_messages)
        if norm:
            yield norm


def _init_source_iterators(max_messages: int, ultrachat_streaming: bool):
    streams = {
        "HuggingFaceH4/ultrachat_200k": ultrachat_stream(
            max_messages=max_messages, streaming=ultrachat_streaming
        ),
        "OpenAssistant/oasst1": oasst_stream(max_messages=max_messages, only_en=True),
    }
    return {name: iter(stream) for name, stream in streams.items()}


def _next_source_item(iters, names, weights, rng, max_messages: int, ultrachat_streaming: bool):
    while True:
        source = rng.choices(names, weights=weights, k=1)[0]
        try:
            messages = next(iters[source])
        except StopIteration:
            if source == "HuggingFaceH4/ultrachat_200k":
                iters[source] = iter(ultrachat_stream(max_messages=max_messages, streaming=ultrachat_streaming))
            else:
                iters[source] = iter(oasst_stream(max_messages=max_messages, only_en=True))
            continue
        return source, messages


def shard_generator(
    shard_tokens: int,
    shard_id: int,
    block_size: int,
    report_tokens_every: int,
    max_messages: int,
    ultrachat_streaming: bool,
    seed: int,
    num_workers: int,
    pool_batch_size: int,
):
    produced_tokens = 0
    next_report = report_tokens_every
    start = time.perf_counter()
    rng = random.Random(seed + shard_id)

    source_weights = {name: float(weight) for name, weight in DATA_SOURCES}
    names = list(source_weights.keys())
    weights = [source_weights[name] for name in names]
    iters = _init_source_iterators(max_messages=max_messages, ultrachat_streaming=ultrachat_streaming)

    if num_workers <= 1:
        _init_worker(TOKENIZER_DIR, block_size, MAX_ASSISTANT_TOKENS)
        while produced_tokens < shard_tokens:
            item = _tokenize_conversation(
                _next_source_item(iters, names, weights, rng, max_messages, ultrachat_streaming)
            )
            if not item:
                continue
            _source, input_ids, labels, attention_mask, _supervised = item
            produced_tokens += block_size
            yield {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

            if produced_tokens >= next_report:
                elapsed = time.perf_counter() - start
                rate = produced_tokens / max(1e-6, elapsed)
                print(
                    f"[Shard {shard_id:03d}] Emitted {produced_tokens:,}/{shard_tokens:,} tok | "
                    f"{rate:,.0f} tok/s"
                )
                next_report += report_tokens_every
        return

    pool = Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(TOKENIZER_DIR, block_size, MAX_ASSISTANT_TOKENS),
        maxtasksperchild=500,
    )

    try:
        while produced_tokens < shard_tokens:
            batch = [
                _next_source_item(iters, names, weights, rng, max_messages, ultrachat_streaming)
                for _ in range(pool_batch_size)
            ]
            for item in pool.imap_unordered(_tokenize_conversation, batch, chunksize=16):
                if not item:
                    continue
                _source, input_ids, labels, attention_mask, _supervised = item

                produced_tokens += block_size
                yield {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

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
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_complete_shard_dir(path: Path) -> bool:
    return (path / "dataset_info.json").exists() and (path / "state.json").exists()


def _shard_index(path: Path) -> int:
    return int(path.name.split("-")[-1])


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
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--pool-batch-size", type=int, default=DEFAULT_POOL_BATCH_SIZE)
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

    shard_dirs = sorted(path for path in Path(OUTPUT_ROOT).glob("shard-*") if path.is_dir())
    incomplete_dirs = [path for path in shard_dirs if not _is_complete_shard_dir(path)]
    for path in incomplete_dirs:
        print(f"[Cleanup] Removing incomplete shard directory: {path}")
        shutil.rmtree(path, ignore_errors=True)

    complete_dirs = sorted(path for path in Path(OUTPUT_ROOT).glob("shard-*") if _is_complete_shard_dir(path))
    shard_id = max((_shard_index(path) for path in complete_dirs), default=-1) + 1

    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "labels": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
        }
    )

    filter_summary = {
        "drop_code_like_assistant": True,
        "drop_json_like_assistant": True,
        "drop_programming_requests": True,
        "drop_url_heavy_examples": True,
        "drop_repetitive_examples": True,
        "drop_placeholder_spam": True,
        "assistant_word_cap": 140,
        "max_assistant_tokens": MAX_ASSISTANT_TOKENS,
        "max_messages": MAX_MESSAGES,
    }

    for _ in range(args.num_shards):
        shard_dir = os.path.join(OUTPUT_ROOT, f"shard-{shard_id:03d}")
        os.makedirs(shard_dir, exist_ok=False)
        print(f"Building shard {shard_id:03d} -> {shard_dir}")
        print(
            f"Target tokens: {SHARD_TOKENS:,} | block_size={BLOCK_SIZE} | "
            f"max_asst={MAX_ASSISTANT_TOKENS} | max_msgs={MAX_MESSAGES}"
        )

        try:
            ds = Dataset.from_generator(
            lambda: shard_generator(
                shard_tokens=SHARD_TOKENS,
                shard_id=shard_id,
                block_size=BLOCK_SIZE,
                report_tokens_every=args.report_tokens_every,
                max_messages=MAX_MESSAGES,
                ultrachat_streaming=args.ultrachat_streaming,
                seed=args.seed,
                num_workers=max(1, int(args.num_workers)),
                pool_batch_size=max(16, int(args.pool_batch_size)),
            ),
            features=features,
            writer_batch_size=args.writer_batch_size,
        )
            ds.save_to_disk(shard_dir)
        except Exception:
            print(f"[Cleanup] Build failed for shard {shard_id:03d}; removing partial directory {shard_dir}")
            shutil.rmtree(shard_dir, ignore_errors=True)
            raise

        shard_metadata = {
            "shard_id": shard_id,
            "dataset": "sft_chat_v5",
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
            "note": (
                "One conversation per row. If too long, drop oldest turns until it fits. "
                "Assistant prefix masked; supervise assistant content (+newline+eos). "
                "Right-pad to block_size. v5 additionally filters code-like, JSON-like, "
                "URL-heavy, repetitive, and programming-heavy examples."
            ),
            "filters": filter_summary,
            "data_sources": [{"name": name, "weight": weight} for name, weight in DATA_SOURCES],
            "created_at": datetime.utcnow().isoformat() + "Z",
            "hostname": socket.gethostname(),
            "ultrachat_streaming": bool(args.ultrachat_streaming),
            "seed": args.seed,
        }
        with open(os.path.join(shard_dir, "metadata.json"), "w") as handle:
            json.dump(shard_metadata, handle, indent=2)

        meta_path = os.path.join(OUTPUT_ROOT, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as handle:
                meta = json.load(handle)

        meta.setdefault("dataset", "sft_chat_v5")
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
        meta.setdefault("filters", filter_summary)
        meta["updated_at"] = datetime.utcnow().isoformat() + "Z"
        meta["shards"] = [
            path.name
            for path in sorted(Path(OUTPUT_ROOT).glob("shard-*"), key=_shard_index)
            if _is_complete_shard_dir(path)
        ]

        with open(meta_path, "w") as handle:
            json.dump(meta, handle, indent=2)

        print(f"Shard {shard_id:03d} complete\n")
        shard_id += 1


if __name__ == "__main__":
    main()
