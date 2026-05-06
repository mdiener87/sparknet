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
import json
import os
import random
import socket
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from datasets import Dataset, Features, Sequence, Value, disable_caching, load_dataset
from datasets.exceptions import DatasetGenerationError
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import RequestException
from transformers import PreTrainedTokenizerFast

os.environ["HF_DATASETS_DISABLE_CACHE"] = "1"
disable_caching()

DEFAULT_CONFIG_PATH = "configs/sparknet-410m/datasets_v1.json"
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BLOCK_SIZE = 1024
DEFAULT_SHARD_TOKENS = 500_000_000
DEFAULT_WRITER_BATCH_SIZE = 50_000
DEFAULT_REPORT_TOKENS_EVERY = 10_000_000

FALLBACK_TEXT_FIELDS = ("text", "content", "body", "page_content")
SOURCE_RETRY_LIMIT = 8
SOURCE_RETRY_BASE_DELAY = 2.0

_WORKER_TOK = None
_WORKER_BOS = None
_WORKER_EOS = None


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
    cfg.setdefault("tokenizer_path", "./tokenizer-v7")
    cfg.setdefault("output_root", "datasets/sparknet-v3-pretrain")
    if "context_length" in cfg and "block_size" not in cfg:
        cfg["block_size"] = int(cfg["context_length"])
    cfg.setdefault("block_size", DEFAULT_BLOCK_SIZE)
    cfg.setdefault("shard_tokens", DEFAULT_SHARD_TOKENS)
    cfg.setdefault("writer_batch_size", DEFAULT_WRITER_BATCH_SIZE)
    cfg.setdefault("report_tokens_every", DEFAULT_REPORT_TOKENS_EVERY)
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
    while True:
        try:
            for row in load_source_dataset(source):
                if isinstance(row, dict):
                    yield from extract_texts(row, source)
            return
        except Exception as err:
            if not is_transient_error(err):
                raise
            attempt += 1
            if attempt > SOURCE_RETRY_LIMIT:
                raise RuntimeError(f"Source {source.get('name')} failed after {SOURCE_RETRY_LIMIT} retries") from err
            delay = SOURCE_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"[SourceRetry] transient error: {err}. Retrying in {delay:.0f}s ({attempt}/{SOURCE_RETRY_LIMIT})")
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

    buffer: List[int] = []
    produced = 0
    seen = 0
    next_report = report_every
    t0 = time.perf_counter()

    rng = random.Random(int(cfg["seed"]) + shard_id)
    stream = text_stream(sources, rng)
    pool = Pool(max(1, cpu_count() - 1), initializer=_init_worker, initargs=(cfg["tokenizer_path"],))

    try:
        for ids in pool.imap_unordered(_tokenize, stream, chunksize=8):
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
        "sources": [
            {
                "name": s.get("name", "json"),
                "split": s.get("split", "train"),
                "prob": float(s["prob"]),
                "config_name": s.get("config_name"),
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
