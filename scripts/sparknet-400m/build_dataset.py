#!/usr/bin/env python3
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
from transformers import LlamaTokenizer

os.environ["HF_DATASETS_DISABLE_CACHE"] = "1"
disable_caching()


DEFAULT_CONFIG_PATH = "configs/sparknet-400m/datasets.json"
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BLOCK_SIZE = 1024
DEFAULT_SHARD_TOKENS = 500_000_000
DEFAULT_WRITER_BATCH_SIZE = 50_000
DEFAULT_REPORT_TOKENS_EVERY = 10_000_000

FALLBACK_TEXT_FIELDS = ("text", "content", "body", "page_content")
SOURCE_RETRY_LIMIT = 8
SOURCE_RETRY_BASE_DELAY = 2.0


_WORKER_TOK = None
_WORKER_EOS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the dataset build config JSON.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, object]:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r") as handle:
        cfg = json.load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a JSON object: {path}")

    cfg.setdefault("seed", 42)
    cfg.setdefault("tokenizer_path", "./tokenizer-v6")
    cfg.setdefault("output_root", "datasets/sparknet-v6-pretrain")
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

    normalized: List[Dict[str, object]] = []
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
        if "text_fields" in item and not isinstance(item["text_fields"], list):
            raise ValueError(f"Source {idx} `text_fields` must be a list when provided.")

        normalized.append(item)

    cfg["tokenizer_path"] = str(resolve_repo_path(str(cfg["tokenizer_path"])))
    cfg["output_root"] = str(resolve_repo_path(str(cfg["output_root"])))
    cfg["sources"] = normalized
    for source in cfg["sources"]:
        data_files = source.get("data_files")
        if isinstance(data_files, dict):
            source["data_files"] = {
                split: str(resolve_repo_path(str(file_path)))
                for split, file_path in data_files.items()
            }
        elif isinstance(data_files, str):
            source["data_files"] = str(resolve_repo_path(data_files))
    return cfg


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _init_worker(tokenizer_path: str):
    global _WORKER_TOK, _WORKER_EOS
    path = Path(tokenizer_path).expanduser()
    model_path = path / "tokenizer.model" if path.is_dir() else path
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
    tok = LlamaTokenizer(vocab_file=str(model_path), legacy=True)
    _WORKER_TOK = tok
    _WORKER_EOS = tok.eos_token_id


def _tokenize(text: str) -> Optional[List[int]]:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    ids = _WORKER_TOK(text, add_special_tokens=False)["input_ids"]
    if not ids:
        return None
    ids.append(_WORKER_EOS)
    return ids


def extract_texts(row: Dict[str, object], source: Dict[str, object]) -> Iterator[str]:
    text_fields = source.get("text_fields")
    if isinstance(text_fields, list) and text_fields:
        fields = tuple(str(field) for field in text_fields)
    else:
        fields = FALLBACK_TEXT_FIELDS

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


def load_source_dataset(source: Dict[str, object]):
    split = str(source.get("split", "train"))
    streaming = bool(source.get("streaming", True))
    config_name = source.get("config_name")

    if source.get("data_files") is not None:
        return load_dataset(
            "json",
            data_files=source["data_files"],
            split=split,
            streaming=streaming,
        )

    return load_dataset(
        str(source["name"]),
        name=str(config_name) if config_name is not None else None,
        split=split,
        streaming=streaming,
    )


def is_transient_source_error(err: Exception) -> bool:
    transient_types = (HfHubHTTPError, RequestException, ConnectionError, TimeoutError, DatasetGenerationError)
    if isinstance(err, transient_types):
        return True

    text = str(err).lower()
    return any(
        needle in text
        for needle in (
            "502",
            "503",
            "504",
            "bad gateway",
            "gateway timeout",
            "read timed out",
            "temporarily unavailable",
            "connection aborted",
            "connection reset",
        )
    )


def source_name(source: Dict[str, object]) -> str:
    config_name = source.get("config_name")
    name = str(source.get("name", "json"))
    if config_name is not None:
        return f"{name}[{config_name}]"
    return name


def source_text_iterator(source: Dict[str, object]) -> Iterator[str]:
    attempt = 0
    while True:
        try:
            dataset = load_source_dataset(source)
            for row in dataset:
                if not isinstance(row, dict):
                    continue
                yield from extract_texts(row, source)
            return
        except Exception as err:
            if not is_transient_source_error(err):
                raise
            attempt += 1
            if attempt > SOURCE_RETRY_LIMIT:
                raise RuntimeError(
                    f"Source {source_name(source)} failed after {SOURCE_RETRY_LIMIT} retries"
                ) from err
            delay = SOURCE_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(
                f"[SourceRetry] {source_name(source)} failed with transient error: {err}. "
                f"Retrying in {delay:.0f}s ({attempt}/{SOURCE_RETRY_LIMIT})"
            )
            time.sleep(delay)


def init_source_iterators(sources: List[Dict[str, object]]) -> Dict[str, Iterator[str]]:
    out: Dict[str, Iterator[str]] = {}
    for idx, source in enumerate(sources):
        out[str(idx)] = iter(source_text_iterator(source))
    return out


def next_source_text(
    source_iters: Dict[str, Iterator[str]],
    sources: List[Dict[str, object]],
    source_keys: List[str],
    weights: List[float],
    rng: random.Random,
) -> str:
    while True:
        key = rng.choices(source_keys, weights=weights, k=1)[0]
        source = sources[int(key)]
        try:
            return next(source_iters[key])
        except StopIteration:
            source_iters[key] = iter(source_text_iterator(source))


def text_stream(sources: List[Dict[str, object]], rng: random.Random) -> Iterator[str]:
    source_keys = [str(idx) for idx in range(len(sources))]
    weights = [float(source["prob"]) for source in sources]
    source_iters = init_source_iterators(sources)

    while True:
        yield next_source_text(source_iters, sources, source_keys, weights, rng)


def is_complete_shard_dir(path: Path) -> bool:
    return (path / "dataset_info.json").exists() and (path / "state.json").exists()


def next_shard_dir(output_root: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    existing = sorted(path for path in root.glob("shard-*") if path.is_dir())
    incomplete = [path for path in existing if not is_complete_shard_dir(path)]
    if incomplete:
        bad = ", ".join(str(path) for path in incomplete)
        raise RuntimeError(
            "Found incomplete shard directories. Clean them up before continuing: "
            f"{bad}"
        )

    next_id = len(existing)
    shard_dir = root / f"shard-{next_id:03d}"
    shard_dir.mkdir(exist_ok=False)
    return shard_dir


def shard_generator(cfg: Dict[str, object], shard_id: int):
    block_size = int(cfg["block_size"])
    shard_tokens = int(cfg["shard_tokens"])
    report_tokens_every = int(cfg["report_tokens_every"])
    tokenizer_path = str(cfg["tokenizer_path"])
    sources = list(cfg["sources"])

    buffer: List[int] = []
    produced_tokens = 0
    total_seen = 0
    next_report = report_tokens_every
    start = time.perf_counter()

    rng = random.Random(int(cfg["seed"]) + shard_id)
    stream = text_stream(sources, rng)
    num_workers = max(1, cpu_count() - 1)
    pool = Pool(num_workers, initializer=_init_worker, initargs=(tokenizer_path,))

    try:
        for ids in pool.imap_unordered(_tokenize, stream, chunksize=8):
            if not ids:
                continue

            buffer.extend(ids)
            total_seen += len(ids)

            while len(buffer) >= block_size:
                chunk = buffer[:block_size]
                buffer = buffer[block_size:]

                produced_tokens += block_size
                yield {
                    "input_ids": chunk,
                    "labels": chunk.copy(),
                    "attention_mask": [1] * block_size,
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
                next_report += report_tokens_every
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

    print(f"Config: {args.config}")
    print(f"Building shard {shard_id:03d} -> {shard_dir}")
    print(f"Target tokens: {int(cfg['shard_tokens']):,}")

    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "labels": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
        }
    )

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
                "name": source.get("name", "json"),
                "split": source.get("split", "train"),
                "prob": float(source["prob"]),
                "streaming": bool(source.get("streaming", True)),
                "config_name": source.get("config_name"),
                "data_files": source.get("data_files"),
                "text_fields": source.get("text_fields"),
            }
            for source in cfg["sources"]
        ],
        "seed": int(cfg["seed"]),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
    }

    with open(shard_dir / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Shard {shard_id:03d} complete")


if __name__ == "__main__":
    main()
