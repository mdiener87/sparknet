#!/usr/bin/env python3
"""
Pack a pretraining shard from the v3 corpus.

Each document is tokenized as:
  [BOS] token_ids... [EOS]

Documents are concatenated and sliced into fixed block_size chunks. The
BOS/EOS boundaries teach the model document structure without any extra
machinery in the training loop.

Run once per shard. For 10B training tokens at 500M tokens/shard, run 20×.
For the eval shard, use datasets_v1_eval.json and run once.

Usage:
  # Build one training shard
  python build_dataset.py --config configs/sparknet-410m/datasets_v1.json

  # Build the eval shard (run once before training)
  python build_dataset.py --config configs/sparknet-410m/datasets_v1_eval.json
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import random
import signal
import socket
import threading
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterator, List, Optional

os.environ["HF_DATASETS_DISABLE_CACHE"] = "1"
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

import torch
from datasets import Dataset, Features, Sequence, Value, disable_caching, load_dataset
from datasets.exceptions import DatasetGenerationError
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import RequestException
from transformers import PreTrainedTokenizerFast

disable_caching()

DEFAULT_CONFIG_PATH = "configs/sparknet-410m/datasets_v1.json"
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BLOCK_SIZE = 1024
DEFAULT_SHARD_TOKENS = 500_000_000
DEFAULT_WRITER_BATCH_SIZE = 50_000
DEFAULT_REPORT_TOKENS_EVERY = 10_000_000
DEFAULT_OUTPUT_ROOT = "datasets/sparknet-410m-v1-pretrain"

FALLBACK_TEXT_FIELDS = ("text", "content", "body", "page_content")
SOURCE_RETRY_LIMIT = 8
SOURCE_RETRY_BASE_DELAY = 2.0
DEFAULT_SOURCE_TIMEOUT_SECONDS = 180
DEFAULT_HOLDOUT_RATE = 0.02
DEFAULT_HOLDOUT_SALT = "sparknet-410m-v1"
DEFAULT_CHATML_WRAP_RATE = 0.005
DEFAULT_LONG_TEXT_CHUNK_CHARS = 200_000

CHATML_USER_PROMPT = "Please continue this document."

_WORKER_TOK = None
_WORKER_BOS = None
_WORKER_EOS = None


class SourceTimeoutError(TimeoutError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def load_config(path: str) -> Dict:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r") as f:
        cfg = json.load(f)

    cfg.setdefault("seed", 42)
    cfg.setdefault("tokenizer_path", "./tokenizer-v8")
    cfg.setdefault("output_root", DEFAULT_OUTPUT_ROOT)
    cfg.setdefault("partition", "train")
    if cfg["partition"] not in ("train", "eval"):
        raise ValueError("Config `partition` must be 'train' or 'eval'.")
    if "context_length" in cfg and "block_size" not in cfg:
        cfg["block_size"] = int(cfg["context_length"])
    cfg.setdefault("block_size", DEFAULT_BLOCK_SIZE)
    cfg.setdefault("shard_tokens", DEFAULT_SHARD_TOKENS)
    cfg.setdefault("writer_batch_size", DEFAULT_WRITER_BATCH_SIZE)
    cfg.setdefault("report_tokens_every", DEFAULT_REPORT_TOKENS_EVERY)
    cfg.setdefault("source_timeout_seconds", DEFAULT_SOURCE_TIMEOUT_SECONDS)
    cfg.setdefault("holdout_rate", DEFAULT_HOLDOUT_RATE)
    cfg.setdefault("holdout_salt", DEFAULT_HOLDOUT_SALT)
    cfg.setdefault("chatml_wrap_rate", DEFAULT_CHATML_WRAP_RATE)
    cfg.setdefault("long_text_chunk_chars", DEFAULT_LONG_TEXT_CHUNK_CHARS)
    cfg["_config_path"] = str(config_path)

    if "mix" in cfg and "sources" not in cfg:
        cfg["sources"] = cfg["mix"]

    sources = cfg.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"Config must define a non-empty `sources` list: {path}")

    normalized: List[Dict] = []
    for idx, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"Source {idx} must be an object.")
        if "name" not in source and "data_files" not in source:
            raise ValueError(f"Source {idx} must define `name` or `data_files`.")
        item = dict(source)
        item.setdefault("split", "train")
        item.setdefault("prob", item.get("weight"))
        if item.get("prob") is None:
            raise ValueError(f"Source {idx} is missing `prob`/`weight`.")
        item["prob"] = float(item["prob"])
        if item["prob"] <= 0:
            raise ValueError(f"Source {idx} must have positive `prob`/`weight`.")
        item.setdefault("streaming", True)
        item.setdefault("timeout_seconds", int(cfg["source_timeout_seconds"]))
        item.setdefault("partition", cfg["partition"])
        item.setdefault("holdout_rate", float(cfg["holdout_rate"]))
        item.setdefault("holdout_salt", str(cfg["holdout_salt"]))
        item.setdefault("chatml_wrap_rate", float(cfg["chatml_wrap_rate"]))
        item.setdefault("long_text_chunk_chars", int(cfg["long_text_chunk_chars"]))
        normalized.append(item)

    cfg["tokenizer_path"] = str(resolve_repo_path(str(cfg["tokenizer_path"])))
    cfg["output_root"] = str(resolve_repo_path(str(cfg["output_root"])))
    cfg["sources"] = normalized
    for source in cfg["sources"]:
        data_files = source.get("data_files")
        if isinstance(data_files, dict):
            source["data_files"] = {
                split: str(resolve_repo_path(str(fp)))
                for split, fp in data_files.items()
            }
        elif isinstance(data_files, str):
            source["data_files"] = str(resolve_repo_path(data_files))
    return cfg


def _init_worker(tokenizer_path: str):
    global _WORKER_TOK, _WORKER_BOS, _WORKER_EOS
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    # Suppress HF's "sequence longer than model_max_length" warning. Documents
    # are intentionally encoded at full length and sliced into blocks downstream.
    tok.model_max_length = int(1e30)
    _WORKER_TOK = tok
    _WORKER_BOS = tok.bos_token_id
    _WORKER_EOS = tok.eos_token_id


def _tokenize(text: str) -> Optional[List[int]]:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    ids = _WORKER_TOK.encode(text, add_special_tokens=False)
    if not ids:
        return None
    # Wrap each document with BOS/EOS so the model learns document boundaries
    return [_WORKER_BOS] + ids + [_WORKER_EOS]


def extract_texts(row: Dict, source: Dict) -> Iterator[str]:
    text_fields = source.get("text_fields")
    fields = tuple(str(f) for f in text_fields) if isinstance(text_fields, list) and text_fields else FALLBACK_TEXT_FIELDS

    for field in fields:
        if field == "*":
            for value in row.values():
                if isinstance(value, str) and value.strip():
                    yield value
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            yield item
            return
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            yield value
            return
        if isinstance(value, list):
            emitted = False
            for item in value:
                if isinstance(item, str) and item.strip():
                    emitted = True
                    yield item
            if emitted:
                return


def load_source_dataset(source: Dict):
    split = str(source.get("split", "train"))
    streaming = bool(source.get("streaming", True))
    config_name = source.get("config_name")
    if source.get("data_files") is not None:
        return load_dataset("json", data_files=source["data_files"], split=split, streaming=streaming)
    return load_dataset(str(source["name"]), name=str(config_name) if config_name else None,
                        split=split, streaming=streaming)


def source_name(source: Dict) -> str:
    name = str(source.get("name", "json"))
    config_name = source.get("config_name")
    return f"{name}[{config_name}]" if config_name else name


def _hash_ratio(*parts: str) -> float:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8", errors="surrogatepass"))
        h.update(b"\0")
    return int.from_bytes(h.digest()[:8], "big") / float(1 << 64)


def _accept_for_partition(text: str, source: Dict) -> bool:
    ratio = _hash_ratio(source_name(source), text, str(source["holdout_salt"]))
    in_holdout = ratio < float(source["holdout_rate"])
    return in_holdout if source["partition"] == "eval" else not in_holdout


def _should_chatml_wrap(text: str, source: Dict) -> bool:
    if source["partition"] != "train":
        return False
    return _hash_ratio(source_name(source), text, str(source["holdout_salt"]), "chatml") < float(source["chatml_wrap_rate"])


def _chatml_wrap(text: str) -> str:
    return (
        f"<|im_start|>user\n{CHATML_USER_PROMPT}<|im_end|>\n"
        f"<|im_start|>assistant\n{text}<|im_end|>"
    )


def _text_chunks(text: str, max_chars: int) -> Iterator[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        yield text
        return
    for start in range(0, len(text), max_chars):
        chunk = text[start:start + max_chars].strip()
        if chunk:
            yield chunk


def _stats(source: Dict) -> Optional[Dict]:
    stats = source.get("_build_stats")
    return stats if isinstance(stats, dict) else None


def _bump(source: Dict, key: str, amount: int = 1):
    stats = _stats(source)
    if stats is not None:
        stats[key] = int(stats.get(key, 0)) + amount


@contextmanager
def source_timeout(seconds: int, label: str):
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise_timeout(signum, frame):
        raise SourceTimeoutError(f"{label} exceeded {seconds}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])
        signal.signal(signal.SIGALRM, old_handler)


def is_transient_error(err: Exception) -> bool:
    if isinstance(err, (HfHubHTTPError, RequestException, ConnectionError, TimeoutError, DatasetGenerationError)):
        return True
    text = str(err).lower()
    return any(needle in text for needle in
               ("502", "503", "504", "bad gateway", "gateway timeout",
                "read timed out", "temporarily unavailable",
                "connection aborted", "connection reset"))


def source_text_iterator(source: Dict) -> Iterator[str]:
    attempt = 0
    name = source_name(source)
    timeout_seconds = int(source.get("timeout_seconds", DEFAULT_SOURCE_TIMEOUT_SECONDS))
    while True:
        try:
            print(f"[SourceOpen] {name}", flush=True)
            with source_timeout(timeout_seconds, f"Opening {name}"):
                dataset = load_source_dataset(source)
            row_iter = iter(dataset)
            while True:
                with source_timeout(timeout_seconds, f"Reading {name}"):
                    row = next(row_iter)
                if isinstance(row, dict):
                    for text in extract_texts(row, source):
                        stripped = text.strip()
                        if not stripped:
                            continue
                        _bump(source, "documents_seen")
                        if not _accept_for_partition(stripped, source):
                            skip_key = "documents_skipped_train" if source["partition"] == "eval" else "documents_skipped_eval"
                            _bump(source, skip_key)
                            continue
                        _bump(source, "documents_accepted")
                        chunks = list(_text_chunks(stripped, int(source["long_text_chunk_chars"])))
                        if len(chunks) > 1:
                            _bump(source, "documents_chunked")
                        for chunk in chunks:
                            _bump(source, "text_chunks_emitted")
                            if _should_chatml_wrap(chunk, source):
                                _bump(source, "chatml_wrapped_chunks")
                                yield _chatml_wrap(chunk)
                            else:
                                yield chunk
        except StopIteration:
            return
        except Exception as err:
            if not is_transient_error(err):
                raise
            attempt += 1
            if attempt > SOURCE_RETRY_LIMIT:
                raise RuntimeError(f"Source {name} failed after {SOURCE_RETRY_LIMIT} retries") from err
            delay = SOURCE_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(
                f"[SourceRetry] {name} transient error: {err}. "
                f"Retrying in {delay:.0f}s ({attempt}/{SOURCE_RETRY_LIMIT})",
                flush=True,
            )
            time.sleep(delay)


def text_stream(sources: List[Dict], rng: random.Random) -> Iterator[str]:
    keys = [str(i) for i in range(len(sources))]
    weights = [float(s["prob"]) for s in sources]
    iters = {k: iter(source_text_iterator(sources[int(k)])) for k in keys}
    while True:
        key = rng.choices(keys, weights=weights, k=1)[0]
        try:
            yield next(iters[key])
        except StopIteration:
            iters[key] = iter(source_text_iterator(sources[int(key)]))


def tokenized_stream(pool: Pool, stream: Iterator[str], max_pending: int) -> Iterator[Optional[List[int]]]:
    pending = []
    while True:
        while len(pending) < max_pending:
            pending.append(pool.apply_async(_tokenize, (next(stream),)))

        ready_idx = next((idx for idx, result in enumerate(pending) if result.ready()), None)
        if ready_idx is None:
            time.sleep(0.01)
            continue

        result = pending.pop(ready_idx)
        yield result.get()


def is_complete_shard(path: Path) -> bool:
    return (path / "dataset_info.json").exists() and (path / "state.json").exists()


def next_shard_dir(output_root: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in root.glob("shard-*") if p.is_dir())
    incomplete = [p for p in existing if not is_complete_shard(p)]
    if incomplete:
        raise RuntimeError(
            "Found incomplete shard directories — clean them up before continuing: "
            + ", ".join(str(p) for p in incomplete)
        )
    shard_dir = root / f"shard-{len(existing):03d}"
    shard_dir.mkdir(exist_ok=False)
    return shard_dir


def shard_generator(cfg: Dict, shard_id: int):
    block_size = int(cfg["block_size"])
    shard_tokens = int(cfg["shard_tokens"])
    report_every = int(cfg["report_tokens_every"])
    sources = list(cfg["sources"])
    stats = {
        "partition": cfg["partition"],
        "holdout_rate": float(cfg["holdout_rate"]),
        "chatml_wrap_rate": float(cfg["chatml_wrap_rate"]),
        "long_text_chunk_chars": int(cfg["long_text_chunk_chars"]),
        "documents_seen": 0,
        "documents_accepted": 0,
        "documents_skipped_eval": 0,
        "documents_skipped_train": 0,
        "documents_chunked": 0,
        "text_chunks_emitted": 0,
        "chatml_wrapped_chunks": 0,
    }
    cfg["_build_stats"] = stats
    for source in sources:
        source["_build_stats"] = stats

    buffer: List[int] = []
    produced = 0
    seen = 0
    next_report = report_every
    t0 = time.perf_counter()

    rng = random.Random(int(cfg["seed"]) + shard_id)
    stream = text_stream(sources, rng)
    num_workers = max(1, cpu_count() - 1)
    pool = Pool(num_workers, initializer=_init_worker, initargs=(cfg["tokenizer_path"],))

    try:
        for ids in tokenized_stream(pool, stream, max_pending=num_workers * 4):
            if not ids:
                continue
            buffer.extend(ids)
            seen += len(ids)

            while len(buffer) >= block_size:
                chunk = buffer[:block_size]
                buffer = buffer[block_size:]
                produced += block_size
                yield {
                    "input_ids": chunk,
                    "labels": chunk.copy(),
                    "attention_mask": [1] * block_size,
                }
                if produced >= shard_tokens:
                    return

            if seen >= next_report:
                rate = seen / max(1e-6, time.perf_counter() - t0)
                print(f"Seen {seen:,} | Emitted {produced:,}/{shard_tokens:,} | {rate:,.0f} tok/s")
                next_report += report_every
    finally:
        pool.terminate()
        for w in pool._pool:
            try:
                w.kill()
            except Exception:
                pass
        pool.join()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))

    shard_dir = next_shard_dir(str(cfg["output_root"]))
    shard_id = int(shard_dir.name.split("-")[-1])

    print(f"Config:        {args.config}")
    print(f"Output shard:  {shard_dir}")
    print(f"Target tokens: {int(cfg['shard_tokens']):,}")

    features = Features({
        "input_ids":      Sequence(Value("int32")),
        "labels":         Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
    })

    ds = Dataset.from_generator(
        lambda: shard_generator(cfg, shard_id),
        features=features,
        writer_batch_size=int(cfg["writer_batch_size"]),
    )
    ds.save_to_disk(str(shard_dir))

    metadata = {
        "shard_id": shard_id,
        "config_path": args.config,
        "tokenizer_path": cfg["tokenizer_path"],
        "block_size": int(cfg["block_size"]),
        "target_tokens": int(cfg["shard_tokens"]),
        "partition": cfg["partition"],
        "holdout_rate": float(cfg["holdout_rate"]),
        "holdout_salt": cfg["holdout_salt"],
        "chatml_wrap_rate": float(cfg["chatml_wrap_rate"]),
        "long_text_chunk_chars": int(cfg["long_text_chunk_chars"]),
        "build_stats": cfg.get("_build_stats", {}),
        "sources": [
            {
                "name": s.get("name", "json"),
                "split": s.get("split", "train"),
                "prob": float(s["prob"]),
                "config_name": s.get("config_name"),
                "text_fields": s.get("text_fields"),
            }
            for s in cfg["sources"]
        ],
        "seed": int(cfg["seed"]),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
    }
    with open(shard_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Shard {shard_id:03d} complete → {shard_dir}")


if __name__ == "__main__":
    main()
