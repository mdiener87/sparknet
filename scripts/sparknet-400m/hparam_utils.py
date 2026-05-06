"""
Shared utilities for SparkNet v3 hyperparameter experiments.
Factored out so each phase script stays focused on its own logic.
"""
import os
from pathlib import Path
from typing import Optional, List

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import LlamaConfig, LlamaTokenizer, AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]


def setup_env():
    os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / "cache"))
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_tf32(enable: bool = True):
    mode = "tf32" if enable else "ieee"
    try:
        torch.backends.cuda.matmul.fp32_precision = mode
    except AttributeError:
        pass
    try:
        torch.backends.cudnn.conv.fp32_precision = mode
    except AttributeError:
        pass


def resolve_repo_path(path: str) -> str:
    p = Path(path).expanduser()
    return str(p) if p.is_absolute() else str((REPO_ROOT / p).resolve())


def load_sparknet_tokenizer(tokenizer_path: str) -> LlamaTokenizer:
    path = Path(tokenizer_path).expanduser()
    model_file = path / "tokenizer.model" if path.is_dir() else path
    if not model_file.exists():
        raise FileNotFoundError(f"Tokenizer not found: {model_file}")
    tok = LlamaTokenizer(vocab_file=str(model_file), legacy=True)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def list_shards(train_root: str) -> List[str]:
    root = Path(train_root)
    all_dirs = sorted(p for p in root.glob("shard-*") if p.is_dir())
    # A valid HF Dataset shard contains dataset_info.json or state.json
    valid, skipped = [], []
    for p in all_dirs:
        if (p / "dataset_info.json").exists() or (p / "state.json").exists():
            valid.append(str(p))
        else:
            skipped.append(p.name)
    if skipped:
        print(f"[Data] Skipping {len(skipped)} empty/invalid shard(s): {skipped}")
    if not valid:
        raise FileNotFoundError(f"No valid shard-* dirs under: {train_root}")
    return valid


def load_prepacked(
    train_root: str,
    limit_shards: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> Dataset:
    shards = list_shards(train_root)
    if limit_shards:
        shards = shards[:limit_shards]
    print(f"[Data] Loading {len(shards)} shard(s) from {Path(train_root).name} ...")
    dsets = [Dataset.load_from_disk(s) for s in shards]
    ds = concatenate_datasets(dsets) if len(dsets) > 1 else dsets[0]
    if max_rows and len(ds) > max_rows:
        ds = ds.select(range(max_rows))
    ds.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
    print(f"[Data] {len(ds):,} rows ready")
    return ds


def build_eval_wikitext(tok: LlamaTokenizer, block_size: int) -> Optional[Dataset]:
    """Returns packed WikiText-2 validation blocks, or None if not cached."""
    eos_id = tok.eos_token_id
    try:
        eval_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    except Exception as e:
        print(f"[Eval] WikiText not available ({e}). Skipping eval dataset.")
        return None

    def tok_line(ex):
        ids = tok(ex["text"], add_special_tokens=False)["input_ids"]
        if not ids:
            return {"ids": []}
        return {"ids": ids + ([] if ids[-1] == eos_id else [eos_id])}

    eval_tok = eval_raw.map(tok_line, remove_columns=eval_raw.column_names)

    def pack(batch):
        flat = [tok_id for ids in batch["ids"] for tok_id in ids]
        blocks = [
            {
                "input_ids": flat[i : i + block_size],
                "labels": flat[i : i + block_size],
                "attention_mask": [1] * block_size,
            }
            for i in range(0, len(flat) - block_size, block_size)
        ]
        if not blocks:
            return {"input_ids": [], "labels": [], "attention_mask": []}
        return {k: [b[k] for b in blocks] for k in ("input_ids", "labels", "attention_mask")}

    packed = eval_tok.map(pack, batched=True, batch_size=1000, remove_columns=["ids"])
    packed.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
    print(f"[Eval] {len(packed):,} WikiText-2 blocks")
    return packed


def build_model(vocab_size: int = 32000, block_size: int = 1024) -> AutoModelForCausalLM:
    """Builds the SparkNet v3 400M Llama architecture from random weights.

    32 layers (v2 used 29 — changed for tensor-parallelism compatibility and
    conventional even-layer depth). intermediate_size=2816 (11×256) for full
    256-alignment on CUDA Tensor Core matmuls.
    """
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=8,
        max_position_embeddings=block_size,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.config.use_cache = False
    model.gradient_checkpointing_disable()
    model.config.attn_implementation = "sdpa"
    return model


def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
