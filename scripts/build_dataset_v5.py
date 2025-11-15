import random
import time
from multiprocessing import Pool, cpu_count

import torch
from datasets import Dataset, Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer

TOKENIZER_DIR = "./tokenizer-v5"
SAVE_DIR = "datasets/sparknet-v5-1b"
TARGET_TOKENS = 1_000_000_000
BLOCK_SIZE = 1024
PROGRESS_TOKENS_STEP = 10_000_000
PACKING_PROGRESS_STEPS = 20  # report roughly this many times while packing
SAVE_WRITER_BATCH_SIZE = 50_000

# Weighted sampling just like CodeLion
DATA_SOURCES = [
    ("codelion/finepdfs-1B",       0.50),
    ("codelion/dclm-baseline-1B",  0.30),
    ("codelion/fineweb-edu-1B",    0.19),
    ("data/diener_blog.jsonl",    0.01),
]

_WORKER_TOKENIZER = None
_WORKER_EOS_ID = None


def _init_worker(tokenizer_dir: str):
    """Load tokenizer inside each worker process once."""
    global _WORKER_TOKENIZER, _WORKER_EOS_ID
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    _WORKER_TOKENIZER = tok
    _WORKER_EOS_ID = tok.eos_token_id


def _tokenize_text(text: str):
    """Return token ids (plus EOS) for a single text sample."""
    if _WORKER_TOKENIZER is None or not isinstance(text, str):
        return None

    ids = _WORKER_TOKENIZER(text, add_special_tokens=False)["input_ids"]
    if not ids:
        return None

    ids.append(_WORKER_EOS_ID)
    return ids


def text_stream():
    """Infinite generator producing text samples based on weighted sampling."""
    while True:
        name = random.choices(
            [src for src, _ in DATA_SOURCES],
            weights=[w for _, w in DATA_SOURCES]
        )[0]

        ds = load_dataset(name, split="train", streaming=True)

        for row in ds:
            if "text" in row and isinstance(row["text"], str):
                yield row["text"]
            else:
                # fallback: any string-like field
                for value in row.values():
                    if isinstance(value, str):
                        yield value


def accumulate_tokens():
    """Tokenize stream samples in parallel until TARGET_TOKENS is reached."""
    raw_tokens = []
    total = 0
    next_report = PROGRESS_TOKENS_STEP
    start_time = time.perf_counter()
    stream = text_stream()

    num_workers = max(1, (cpu_count() or 1) - 1)
    print(f"Beginning token accumulation with {num_workers} workers…")

    pool = Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(TOKENIZER_DIR,)
    )
    iterator = pool.imap_unordered(_tokenize_text, stream, chunksize=8)

    try:
        for ids in iterator:
            if not ids:
                continue

            raw_tokens.extend(ids)
            total += len(ids)

            if total >= next_report:
                elapsed = max(1e-6, time.perf_counter() - start_time)
                rate = total / elapsed
                pct = min(100.0, (total / TARGET_TOKENS) * 100)
                print(
                    f"Tokenized {total:,}/{TARGET_TOKENS:,} tokens "
                    f"({pct:.2f}%) at ~{rate:,.0f} tok/s"
                )
                next_report += PROGRESS_TOKENS_STEP

            if total >= TARGET_TOKENS:
                print(f"Reached target: {total:,} tokens.")
                break
    finally:
        pool.terminate()
        pool.join()

    return raw_tokens


def pack_sequences(raw_tokens):
    """Yield packed sequences to avoid holding the entire dataset in RAM."""
    print("Packing into 1024-token sequences…")
    total_sequences = max(0, len(raw_tokens) // BLOCK_SIZE)
    report_every = max(1, total_sequences // PACKING_PROGRESS_STEPS) if total_sequences else 0

    def generator():
        packed = 0
        for i in range(0, len(raw_tokens) - BLOCK_SIZE + 1, BLOCK_SIZE):
            chunk = raw_tokens[i:i + BLOCK_SIZE]
            if len(chunk) < BLOCK_SIZE:
                break

            packed += 1
            if report_every and packed % report_every == 0:
                pct = (packed / total_sequences) * 100 if total_sequences else 100.0
                print(f"Packed {packed:,}/{total_sequences:,} sequences ({pct:.1f}%).")

            yield {
                "input_ids": chunk,
                "labels": chunk.copy(),
                "attention_mask": [1] * BLOCK_SIZE
            }

        print(f"Created {packed:,} sequences of block size {BLOCK_SIZE}.")

    return generator


def save_dataset_chunked(generator_fn):
    """Stream packed sequences to disk using smaller writer batches."""
    print("Saving dataset with chunked writer…")
    features = Features({
        "input_ids": Sequence(Value("int32")),
        "labels": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8"))
    })
    ds = Dataset.from_generator(
        generator_fn,
        features=features,
        writer_batch_size=SAVE_WRITER_BATCH_SIZE
    )
    ds.save_to_disk(SAVE_DIR)
    print(f"Dataset saved to: {SAVE_DIR}/")


def main():
    random.seed(42)
    torch.manual_seed(42)

    print("Loading tokenizer…")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    print(f"Tokenizer vocab size: {len(tok)} | EOS id: {tok.eos_token_id}")

    raw_tokens = accumulate_tokens()
    sequence_generator = pack_sequences(raw_tokens)
    save_dataset_chunked(sequence_generator)

    print("Done.")


if __name__ == "__main__":
    main()
