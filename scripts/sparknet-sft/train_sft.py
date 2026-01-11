#!/usr/bin/env python3
"""
train_sft_phase1.py

Phase 1 SFT trainer for SparkNet-400m v1 on packed SFT shards.

Improvements vs your prior script:
- WORLD_SIZE-aware token budgeting (correct max_steps under DDP).
- Proper SDPA backend selection logging (no misleading "flash enabled" claim).
- Optional gradient checkpointing.
- Optional tiny eval split + load_best_model_at_end.
- bf16 safety checks.
- Validates dataset seq_len == block_size.
- Passes tokenizer to Trainer (future-proof).
- Richer training_metadata.json (versions, world_size, effective batch/tokens).
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
    set_seed,
)

# -----------------------------
# Config
# -----------------------------
@dataclass
class RunConfig:
    run_name: str = "sparknet-400m-v1-instruct"
    seed: int = 42

    model_path: str = "checkpoints/sparknet-400m-v1"
    tokenizer_path: str = "./tokenizer-v6"
    train_root: str = "datasets/sft_chat_v1"
    block_size: int = 1024

    # Training
    bf16: bool = True
    gradient_checkpointing: bool = False

    per_device_train_batch_size: int = 16
    grad_accum: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"
    max_grad_norm: float = 1.0

    # Token budget
    target_tokens: int = 200_000_000

    # Eval (optional)
    eval_fraction: float = 0.0      # e.g. 0.005 (0.5%) to enable
    eval_steps: Optional[int] = None  # default: save_steps if eval enabled

    # I/O + cadence
    output_dir: str = "checkpoints/sparknet-400m-v1-instruct"
    logging_steps: int = 50
    save_steps: int = 1000
    save_total_limit: Optional[int] = 3

    dataloader_num_workers: int = 8
    dataloader_prefetch_factor: int = 4

    # Optional controls
    limit_shards: Optional[int] = None
    resume_from: Optional[str] = None  # path to checkpoint dir, or "latest"


# -----------------------------
# Utilities
# -----------------------------
def set_tf32(enable: bool = True):
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32" if enable else "ieee"
    except Exception:
        pass
    try:
        torch.backends.cudnn.conv.fp32_precision = "tf32" if enable else "ieee"
    except Exception:
        pass


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def list_shards(train_root: str) -> List[str]:
    root = Path(train_root)
    shards = sorted([str(p) for p in root.glob("shard-*") if p.is_dir()])
    if not shards:
        raise FileNotFoundError(f"No shard-* dirs found under: {train_root}")
    return shards


def load_prepacked(train_root: str, limit: Optional[int] = None) -> Dataset:
    shards = list_shards(train_root)
    if limit is not None:
        shards = shards[:limit]
    print(f"[Data] Loading {len(shards)} shard(s) from {train_root}")

    dsets = [Dataset.load_from_disk(sd) for sd in shards]
    train = concatenate_datasets(dsets) if len(dsets) > 1 else dsets[0]
    train.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
    print(f"[Data] rows={len(train):,}")
    return train


def validate_block_size(ds: Dataset, block_size: int, n: int = 3):
    """Fast sanity check: ensure prepacked sequences match expected length."""
    n = min(n, len(ds))
    for i in range(n):
        ex = ds[i]
        seq_len = len(ex["input_ids"])
        if seq_len != block_size:
            raise ValueError(f"Dataset seq_len mismatch at row {i}: {seq_len} != block_size {block_size}")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    out = Path(output_dir)
    if not out.exists():
        return None
    ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(ckpts[-1]) if ckpts else None


def check_bf16_or_die(enable_bf16: bool):
    if not enable_bf16:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("bf16 requested but CUDA is not available.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested but this GPU does not support bf16.")


def try_log_sdpa_backend():
    """
    Best-effort SDPA backend logging (no guarantees; PyTorch chooses per op).
    We avoid claiming flash is enabled without evidence.
    """
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend  # type: ignore
        # If import works, SDPA kernel API exists; backend selection is still dynamic.
        _ = (sdpa_kernel, SDPBackend)
        print("[Attn] SDPA available (backend chosen dynamically per call).")
    except Exception:
        print("[Attn] SDPA backend API not available; using default attention behavior.")


# -----------------------------
# Callbacks
# -----------------------------
class PerfCallback(TrainerCallback):
    def __init__(self, tokens_per_step: int):
        self.tokens_per_step = tokens_per_step
        self.last_t = None
        self.last_step = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_t = time.time()
        self.last_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.last_t is None or self.last_step is None:
            return
        if state.global_step <= self.last_step:
            return
        now = time.time()
        dt = now - self.last_t
        ds = state.global_step - self.last_step
        if dt <= 0 or ds <= 0:
            return
        tok_per_s = (ds / dt) * self.tokens_per_step
        print(f"[Perf] ~{tok_per_s:,.0f} tok/s | step={state.global_step:,}")
        self.last_t = now
        self.last_step = state.global_step


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Optional path to a json config override.")
    parser.add_argument("--resume", type=str, default=None, help='Checkpoint path or "latest". Overrides config.resume_from.')
    parser.add_argument("--run-name", type=str, default=None, help="Override cfg.run_name (overrides config file)")
    parser.add_argument("--eval-fraction", type=float, default=None, help="Override cfg.eval_fraction")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    args_cli = parser.parse_args()

    default_cfg = RunConfig()
    cfg = RunConfig()

    # Load JSON overrides
    if args_cli.config:
        with open(args_cli.config, "r") as f:
            overrides = json.load(f)
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(f"Unknown config field: {k}")
            setattr(cfg, k, v)

    # CLI overrides
    if args_cli.resume:
        cfg.resume_from = args_cli.resume
    if args_cli.run_name:
        cfg.run_name = args_cli.run_name
        if cfg.output_dir == default_cfg.output_dir:
            cfg.output_dir = f"checkpoints/{cfg.run_name}"
    if args_cli.eval_fraction is not None:
        cfg.eval_fraction = float(args_cli.eval_fraction)
    if args_cli.gradient_checkpointing:
        cfg.gradient_checkpointing = True

    # Environment
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    set_seed(cfg.seed)
    set_tf32(True)
    check_bf16_or_die(cfg.bf16)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Data
    full = load_prepacked(cfg.train_root, cfg.limit_shards)
    validate_block_size(full, cfg.block_size)

    train_ds = full
    eval_ds = None
    if cfg.eval_fraction and cfg.eval_fraction > 0:
        splits = full.train_test_split(test_size=cfg.eval_fraction, seed=cfg.seed)
        train_ds = splits["train"]
        eval_ds = splits["test"]
        print(f"[Data] train={len(train_ds):,} eval={len(eval_ds):,} (eval_fraction={cfg.eval_fraction})")

    # Tokenizer + model
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_path, padding_side="right", use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_path)
    model.config.use_cache = False

    # Prefer SDPA if available; no misleading “enabled” print.
    try:
        model.config.attn_implementation = "sdpa"
    except Exception:
        pass
    try_log_sdpa_backend()

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        print("[Train] Gradient checkpointing enabled")

    # Budgeting (WORLD_SIZE-aware)
    ws = world_size()
    tokens_per_step = cfg.block_size * cfg.per_device_train_batch_size * cfg.grad_accum * ws
    max_steps = math.ceil(cfg.target_tokens / tokens_per_step)
    eff_bsz = cfg.per_device_train_batch_size * cfg.grad_accum * ws
    print(
        f"[Budget] target_tokens={cfg.target_tokens:,} | tokens/step={tokens_per_step:,} | max_steps={max_steps:,} | "
        f"world_size={ws} | effective_batch={eff_bsz}"
    )
    warmup_steps = int(cfg.warmup_ratio * max_steps)
    print(f"[LR] warmup_ratio={cfg.warmup_ratio} -> warmup_steps={warmup_steps:,}")

    # Resume handling
    resume_from = cfg.resume_from
    if resume_from == "latest":
        resume_from = find_latest_checkpoint(cfg.output_dir)
        if resume_from:
            print(f"[Resume] latest checkpoint: {resume_from}")

    # Eval settings
    evaluation_strategy = "no"
    eval_steps = None
    load_best = False
    metric_for_best = None
    greater_is_better = None

    if eval_ds is not None:
        evaluation_strategy = "steps"
        eval_steps = cfg.eval_steps or cfg.save_steps
        if cfg.save_steps % eval_steps != 0:
            print(
                f"[Eval] save_steps ({cfg.save_steps}) is not a multiple of eval_steps ({eval_steps}); "
                f"setting eval_steps={cfg.save_steps} for load_best_model_at_end compatibility."
            )
            eval_steps = cfg.save_steps
        load_best = True
        metric_for_best = "eval_loss"
        greater_is_better = False

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        bf16=cfg.bf16,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.scheduler,
        max_steps=max_steps,

        logging_dir=f"logs/{cfg.run_name}",
        logging_steps=cfg.logging_steps,

        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,

        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=load_best,
        metric_for_best_model=metric_for_best,
        greater_is_better=greater_is_better,

        optim="adamw_torch_fused",
        report_to=["tensorboard"],
        remove_unused_columns=False,

        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=(cfg.dataloader_num_workers > 0),
        dataloader_prefetch_factor=cfg.dataloader_prefetch_factor,

        max_grad_norm=cfg.max_grad_norm,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=default_data_collator,
        callbacks=[PerfCallback(tokens_per_step)],
    )

    trainer.train(resume_from_checkpoint=resume_from)

    # Save final artifacts
    tok.save_pretrained(cfg.output_dir)
    trainer.save_model(cfg.output_dir)

    # Metadata for reproducibility
    metadata = {
        "run_name": cfg.run_name,
        "output_dir": cfg.output_dir,
        "model_path": cfg.model_path,
        "tokenizer_path": cfg.tokenizer_path,
        "train_root": cfg.train_root,
        "block_size": cfg.block_size,

        "target_tokens": cfg.target_tokens,
        "world_size": ws,
        "tokens_per_step": tokens_per_step,
        "effective_batch_size": eff_bsz,
        "max_steps": max_steps,

        "bf16": cfg.bf16,
        "gradient_checkpointing": cfg.gradient_checkpointing,

        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "grad_accum": cfg.grad_accum,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio,
        "warmup_steps": warmup_steps,
        "scheduler": cfg.scheduler,
        "max_grad_norm": cfg.max_grad_norm,

        "eval_fraction": cfg.eval_fraction,
        "eval_steps": eval_steps,

        "torch_version": torch.__version__,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(cfg.output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete")


if __name__ == "__main__":
    main()
