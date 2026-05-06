"""
Shared utilities for SparkNet-410M v1 hyperparameter experiments.
Factored out so each phase script stays focused on its own logic.
"""
import os
from pathlib import Path
from typing import Optional, List

import torch
from datasets import Dataset, concatenate_datasets
from transformers import LlamaConfig, PreTrainedTokenizerFast, AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]


def setup_env():
    os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / "cache"))
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


def load_sparknet_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    path = Path(tokenizer_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    tok = PreTrainedTokenizerFast.from_pretrained(str(path))
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    validate_tokenizer(tok)
    return tok


def validate_tokenizer(tok: PreTrainedTokenizerFast):
    expected_ids = {
        "<|begin_of_text|>": 0,
        "<|end_of_text|>": 1,
        "<|im_start|>": 2,
        "<|im_end|>": 3,
    }
    if tok.vocab_size != 32000 or len(tok) != 32000:
        raise ValueError(f"Expected tokenizer vocab/len 32000, got vocab_size={tok.vocab_size}, len={len(tok)}")
    for token, expected_id in expected_ids.items():
        actual_id = tok.convert_tokens_to_ids(token)
        if actual_id != expected_id:
            raise ValueError(f"{token} id mismatch: expected {expected_id}, got {actual_id}")
    missing = [t for t in ("<|im_start|>", "<|im_end|>") if t not in tok.additional_special_tokens]
    if missing:
        raise ValueError(f"ChatML tokens missing from additional_special_tokens: {missing}")
    if not tok.chat_template:
        raise ValueError("Tokenizer missing chat_template")


def list_shards(train_root: str) -> List[str]:
    root = Path(train_root)
    all_dirs = sorted(p for p in root.glob("shard-*") if p.is_dir())
    # A complete HF Dataset shard contains both files.
    valid, skipped = [], []
    for p in all_dirs:
        if (p / "dataset_info.json").exists() and (p / "state.json").exists():
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


def load_eval_prepacked(eval_root: str, max_rows: Optional[int] = None) -> Dataset:
    return load_prepacked(eval_root, limit_shards=None, max_rows=max_rows)


def build_model(vocab_size: int = 32000, block_size: int = 1024) -> AutoModelForCausalLM:
    """Builds the SparkNet-410M v1 Llama architecture from random weights.

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
        rope_theta=500000.0,
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
