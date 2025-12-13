import os
import random
import time
from multiprocessing import Pool, cpu_count
from typing import Iterator, List

import torch
from datasets import Dataset, Features, Sequence, Value, load_dataset
from transformers import LlamaTokenizer

import json
import socket
from datetime import datetime

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
TOKENIZER_DIR = "./tokenizer-v6"
OUTPUT_ROOT = "datasets/sparknet-v6-pretrain"

BLOCK_SIZE = 1024
SHARD_TOKENS = 500_000_000      # tokens per shard (e.g. 500M)
WRITER_BATCH_SIZE = 50_000
REPORT_TOKENS_EVERY = 10_000_000

DATA_SOURCES = [
    ("codelion/fineweb-edu-1B", 0.55),
    ("codelion/dclm-baseline-1B", 0.25),
    ("codelion/finepdfs-1B", 0.15),
    ("eli5", 0.04),
    ("data/diener_blog.jsonl", 0.01),
]

# ---------------------------------------------------------------------
# Worker tokenizer
# ---------------------------------------------------------------------
_WORKER_TOK = None
_WORKER_EOS = None


def _init_worker():
    global _WORKER_TOK, _WORKER_EOS
    tok = LlamaTokenizer.from_pretrained(TOKENIZER_DIR)
    _WORKER_TOK = tok
    _WORKER_EOS = tok.eos_token_id


def _tokenize(text: str) -> List[int] | None:
    if not isinstance(text, str):
        return None
    ids = _WORKER_TOK(text, add_special_tokens=False)["input_ids"]
    if not ids:
        return None
    ids.append(_WORKER_EOS)
    return ids


# ---------------------------------------------------------------------
# Text stream
# ---------------------------------------------------------------------
def text_stream() -> Iterator[str]:
    while True:
        src = random.choices(
            [s for s, _ in DATA_SOURCES],
            weights=[w for _, w in DATA_SOURCES],
        )[0]

        if src == "eli5":
            ds = load_dataset("eli5", split="train", streaming=True)
            for row in ds:
                yield row["question"]
                yield row["answer"]

        elif src.endswith(".jsonl"):
            ds = load_dataset("json", data_files=src, split="train", streaming=True)
            for row in ds:
                for v in row.values():
                    if isinstance(v, str):
                        yield v

        else:
            ds = load_dataset(src, split="train", streaming=True)
            for row in ds:
                if "text" in row:
                    yield row["text"]


# ---------------------------------------------------------------------
# Sharded packed sequence generator
# ---------------------------------------------------------------------
def shard_generator(shard_tokens: int):
    buffer: List[int] = []
    produced_tokens = 0
    total_seen = 0
    next_report = REPORT_TOKENS_EVERY
    start = time.perf_counter()

    num_workers = max(1, cpu_count() - 1)
    pool = Pool(num_workers, initializer=_init_worker)
    stream = text_stream()

    try:
        for ids in pool.imap_unordered(_tokenize, stream, chunksize=8):
            if not ids:
                continue

            buffer.extend(ids)
            total_seen += len(ids)

            while len(buffer) >= BLOCK_SIZE:
                chunk = buffer[:BLOCK_SIZE]
                buffer = buffer[BLOCK_SIZE:]

                produced_tokens += BLOCK_SIZE
                yield {
                    "input_ids": chunk,
                    "labels": chunk.copy(),
                    "attention_mask": [1] * BLOCK_SIZE,
                }

                if produced_tokens >= shard_tokens:
                    return

            if total_seen >= next_report:
                elapsed = time.perf_counter() - start
                rate = total_seen / max(1e-6, elapsed)
                print(
                    f"Seen {total_seen:,} tokens | "
                    f"Emitted {produced_tokens:,}/{shard_tokens:,} | "
                    f"{rate:,.0f} tok/s"
                )
                next_report += REPORT_TOKENS_EVERY

    finally:
        pool.terminate()
        pool.join()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    random.seed(42)
    torch.manual_seed(42)

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
        lambda: shard_generator(SHARD_TOKENS),
        features=features,
        writer_batch_size=WRITER_BATCH_SIZE,
    )
    ds.save_to_disk(shard_dir)

    metadata = {
        "shard_id": shard_id,
        "tokenizer": "sparknet-v6",
        "tokenizer_path": TOKENIZER_DIR,
        "block_size": BLOCK_SIZE,
        "target_tokens": SHARD_TOKENS,
        "data_sources": [
            {"name": name, "weight": weight}
            for name, weight in DATA_SOURCES
        ],
        "created_at": datetime.utcnow().isoformat() + "Z",
        "hostname": socket.gethostname(),
    }

    with open(os.path.join(shard_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Shard {shard_id:03d} complete")



if __name__ == "__main__":
    main()
