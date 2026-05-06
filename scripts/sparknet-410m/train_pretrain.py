#!/usr/bin/env python3
"""
SparkNet-410M v1 pretraining script.

Key changes from the sparknet-400m scripts:
  - PreTrainedTokenizerFast (ByteLevel BPE) replaces LlamaTokenizer (SentencePiece).
    Eliminates the HF/llama.cpp newline tokenization mismatch that broke v2 GGUF.
  - Corpus-sampled eval dataset replaces WikiText-2. The 261-row WikiText eval
    produced a noisy U-shaped loss curve in v2; a 9,765-row held-out corpus shard
    gives a stable signal throughout the full 16-day run.
  - SmartCheckpointCallback manages three retention buckets:
      1. Top-3 by eval loss (protected from pruning)
      2. Token-cardinal snapshots: 6B, 7B, 8B, 9B, 10B (copied to separate dirs)
      3. Last-3 regular step checkpoints (power-failure resume)
  - Cosine schedule with min_lr floor (cosine_with_min_lr, floor = 0.1 × peak LR).
    Plain cosine to zero over a 16-day run wastes the final ~10% of training steps.
  - Learning rate 1.83e-3 anchored to Phase 2 LR range test (ceiling / 3).
    Phase 3 grid runs at 96 steps were invalid due to cosine schedule collapse.
  - Architecture: 32 layers, intermediate_size=2816 → ~410M parameters.

Usage:
  python train_pretrain.py --config configs/sparknet-410m/pretrain_v1.json
  python train_pretrain.py --config configs/sparknet-410m/pretrain_v1.json --resume latest
"""

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
import argparse

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    run_name: str = "sparknet-410m-v1"
    seed: int = 42

    tokenizer_path: str = "./tokenizer-v7"
    train_root: str = "datasets/sparknet-v3-pretrain"
    eval_root: str = "datasets/sparknet-v3-pretrain-eval"
    block_size: int = 1024

    # Architecture (~410M)
    hidden_size: int = 1024
    num_layers: int = 32
    num_heads: int = 16
    num_kv_heads: int = 8
    intermediate_size: int = 2816
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    # Training
    bf16: bool = True
    per_device_train_batch_size: int = 32
    grad_accum: int = 32

    learning_rate: float = 1.83e-3
    weight_decay: float = 0.1
    warmup_ratio: float = 0.02
    cosine_min_lr_ratio: float = 0.1
    scheduler: str = "cosine_with_min_lr"
    max_grad_norm: float = 1.0

    # Budget
    target_tokens: int = 10_000_000_000
    checkpoint_tokens: List[int] = field(
        default_factory=lambda: [6_000_000_000, 7_000_000_000, 8_000_000_000,
                                  9_000_000_000, 10_000_000_000]
    )
    keep_best_checkpoints: int = 3
    keep_last_checkpoints: int = 3

    # Cadence
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500

    dataloader_num_workers: int = 16
    dataloader_prefetch_factor: int = 4

    # Optional
    limit_shards: Optional[int] = None
    resume_from: Optional[str] = None
    do_sample_generations: bool = True
    sample_gen_every_eval: bool = True
    sample_prompts_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_tf32(enable: bool = True):
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32" if enable else "ieee"
    except Exception:
        pass
    try:
        torch.backends.cudnn.conv.fp32_precision = "tf32" if enable else "ieee"
    except Exception:
        pass


def resolve_repo_path(path: str) -> str:
    p = Path(path).expanduser()
    return str(p) if p.is_absolute() else str((REPO_ROOT / p).resolve())


def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_prepacked(root: str, limit: Optional[int] = None) -> Dataset:
    shards = sorted(Path(root).glob("shard-*"), key=lambda p: p.name)
    if not shards:
        raise FileNotFoundError(f"No shard-* dirs found under: {root}")
    if limit is not None:
        shards = shards[:limit]
    print(f"[Data] Loading {len(shards)} shard(s) from {root}")
    dsets = [Dataset.load_from_disk(str(s)) for s in shards]
    ds = concatenate_datasets(dsets) if len(dsets) > 1 else dsets[0]
    ds.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
    print(f"[Data] rows={len(ds):,}")
    return ds


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    ckpts = sorted(Path(output_dir).glob("checkpoint-[0-9]*"),
                   key=lambda p: int(p.name.split("-")[-1]))
    return str(ckpts[-1]) if ckpts else None


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class PerfCallback(TrainerCallback):
    def __init__(self, tokens_per_step: int):
        self.tokens_per_step = tokens_per_step
        self.t0 = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or self.t0 is None or state.global_step <= 0:
            return
        dt = time.time() - self.t0
        if dt <= 0:
            return
        tok_per_s = (state.global_step / dt) * self.tokens_per_step
        if "loss" in logs:
            print(f"[Perf] ~{tok_per_s:,.0f} tok/s | step={state.global_step:,}")


class SampleGenCallback(TrainerCallback):
    def __init__(self, tok, prompts: List[str], every_eval: bool = True):
        self.tok = tok
        self.prompts = prompts
        self.every_eval = every_eval

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not self.every_eval or model is None:
            return
        model.eval()
        device = next(model.parameters()).device
        print("\n[SampleGen] =====")
        with torch.no_grad():
            for p in self.prompts:
                ids = self.tok(p, return_tensors="pt").input_ids.to(device)
                out = model.generate(
                    ids,
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    repetition_penalty=1.05,
                    pad_token_id=self.tok.eos_token_id,
                    eos_token_id=self.tok.eos_token_id,
                )
                text = self.tok.decode(out[0], skip_special_tokens=True)
                print(f"\nPROMPT: {p}\n{text}\n")
        print("[SampleGen] =====\n")


class SmartCheckpointCallback(TrainerCallback):
    """
    Three-bucket checkpoint retention policy:

      1. best-K    — top keep_best checkpoints by eval_loss, never pruned.
      2. cardinal  — token milestone snapshots (6B, 7B, … tokens), copied to
                     checkpoint-{N}B-tokens dirs and never touched by pruning.
      3. last-N    — keep_last most recent step checkpoints for power-failure
                     resume; rolling buffer, oldest pruned as new ones arrive.

    Eval loss is associated with each checkpoint by writing a .eval_loss marker
    file into the checkpoint directory at save time, using the most recent eval
    result. With eval_steps == save_steps the association is exact.
    """

    def __init__(self, output_dir: str, token_step_map: dict,
                 keep_best: int = 3, keep_last: int = 3):
        self.output_dir = Path(output_dir)
        self.token_step_map = dict(token_step_map)   # step -> token_count
        self.keep_best = keep_best
        self.keep_last = keep_last
        self._step_eval_loss: dict = {}
        self._last_eval_loss: Optional[float] = None
        self._triggered_cardinal: set = set()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            loss = float(metrics["eval_loss"])
            self._step_eval_loss[state.global_step] = loss
            self._last_eval_loss = loss

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step in self.token_step_map and step not in self._triggered_cardinal:
            self._triggered_cardinal.add(step)
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        ckpt = self.output_dir / f"checkpoint-{step}"

        # Annotate with the most recent eval loss
        loss = self._step_eval_loss.get(step, self._last_eval_loss)
        if loss is not None and ckpt.exists():
            (ckpt / ".eval_loss").write_text(str(loss))

        # Cardinal snapshot
        if step in self.token_step_map and step in self._triggered_cardinal:
            token_count = self.token_step_map[step]
            n_b = token_count // 1_000_000_000
            cardinal = self.output_dir / f"checkpoint-{n_b}B-tokens"
            if not cardinal.exists() and ckpt.exists():
                shutil.copytree(str(ckpt), str(cardinal))
                print(f"[Checkpoint] Cardinal snapshot → {cardinal.name}")

        self._prune()
        return control

    def _prune(self):
        # Only manage numbered step dirs; cardinal dirs are left alone
        ckpts = sorted(
            (p for p in self.output_dir.iterdir()
             if p.is_dir() and p.name.startswith("checkpoint-")
             and p.name[len("checkpoint-"):].isdigit()),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not ckpts:
            return

        def eval_loss(p: Path) -> float:
            marker = p / ".eval_loss"
            if marker.exists():
                try:
                    return float(marker.read_text().strip())
                except ValueError:
                    pass
            return float("inf")

        best_k = {p for p in sorted(ckpts, key=eval_loss)[:self.keep_best]}
        last_n = set(ckpts[-self.keep_last:])
        protected = best_k | last_n

        for ckpt in ckpts:
            if ckpt not in protected:
                shutil.rmtree(str(ckpt), ignore_errors=True)
                print(f"[Checkpoint] Pruned {ckpt.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help='Checkpoint path or "latest".')
    parser.add_argument("--run-name", type=str, default=None)
    args_cli = parser.parse_args()

    cfg = RunConfig()
    if args_cli.config:
        with open(args_cli.config, "r") as f:
            overrides = json.load(f)
        for k, v in overrides.items():
            if k.startswith("_"):
                continue
            if not hasattr(cfg, k):
                raise ValueError(f"Unknown config field: {k!r}")
            setattr(cfg, k, v)
    if args_cli.resume:
        cfg.resume_from = args_cli.resume
    if args_cli.run_name:
        cfg.run_name = args_cli.run_name

    cfg.tokenizer_path = resolve_repo_path(cfg.tokenizer_path)
    cfg.train_root = resolve_repo_path(cfg.train_root)
    cfg.eval_root = resolve_repo_path(cfg.eval_root)
    if cfg.sample_prompts_path:
        cfg.sample_prompts_path = resolve_repo_path(cfg.sample_prompts_path)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_tf32(True)
    set_seed(cfg.seed)

    run_dir = str((REPO_ROOT / "checkpoints" / cfg.run_name).resolve())
    log_dir = str((REPO_ROOT / "logs" / cfg.run_name).resolve())
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Tokenizer
    tok = load_tokenizer(cfg.tokenizer_path)

    # Data
    train_ds = load_prepacked(cfg.train_root, limit=cfg.limit_shards)
    eval_ds = load_prepacked(cfg.eval_root)

    # Model
    model_cfg = LlamaConfig(
        vocab_size=tok.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_kv_heads,
        max_position_embeddings=cfg.block_size,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(model_cfg)
    model.config.use_cache = False
    model.gradient_checkpointing_disable()
    model.config.attn_implementation = "sdpa"

    if hasattr(torch.backends.cuda, "sdp_kernel"):
        torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=True, enable_math=False
        )

    # Budget
    tokens_per_step = cfg.block_size * cfg.per_device_train_batch_size * cfg.grad_accum
    max_steps = math.ceil(cfg.target_tokens / tokens_per_step)
    checkpoint_steps = [math.ceil(t / tokens_per_step) for t in cfg.checkpoint_tokens]
    token_step_map = dict(zip(checkpoint_steps, cfg.checkpoint_tokens))

    print(f"[Budget] target={cfg.target_tokens:,} | tok/step={tokens_per_step:,} | steps={max_steps:,}")
    print(f"[Budget] cardinal steps: {checkpoint_steps}")

    # Scheduler: cosine_with_min_lr requires transformers >= 4.38.
    # If unavailable, falls back to plain cosine with a logged warning.
    import transformers as _tf
    from packaging.version import Version
    if cfg.scheduler == "cosine_with_min_lr":
        if Version(_tf.__version__) >= Version("4.38.0"):
            scheduler_type = "cosine_with_min_lr"
            scheduler_kwargs = {"min_lr_rate": cfg.cosine_min_lr_ratio}
        else:
            print(
                f"[Warning] cosine_with_min_lr requires transformers>=4.38 "
                f"(found {_tf.__version__}). Falling back to plain cosine."
            )
            scheduler_type = "cosine"
            scheduler_kwargs = {}
    else:
        scheduler_type = cfg.scheduler
        scheduler_kwargs = {}

    train_args = TrainingArguments(
        output_dir=run_dir,
        bf16=cfg.bf16,

        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,

        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=scheduler_type,
        lr_scheduler_kwargs=scheduler_kwargs,

        max_steps=max_steps,
        max_grad_norm=cfg.max_grad_norm,

        logging_dir=log_dir,
        logging_steps=cfg.logging_steps,

        eval_strategy="steps",
        eval_steps=cfg.eval_steps,

        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=None,          # SmartCheckpointCallback owns retention

        optim="adamw_torch_fused",

        report_to=["tensorboard"],
        remove_unused_columns=False,

        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=cfg.dataloader_prefetch_factor,
    )

    prompts = [
        "The meaning of life is",
        "In the future, AI will",
        "A good software engineer knows that",
        "Write a Python function that",
        "If I have 12 apples and give away 5,",
        "Explain the difference between supervised and unsupervised learning:",
        "The capital of France is",
    ]
    if cfg.sample_prompts_path:
        with open(cfg.sample_prompts_path, "r") as f:
            prompts = json.load(f)

    smart_ckpt = SmartCheckpointCallback(
        output_dir=run_dir,
        token_step_map=token_step_map,
        keep_best=cfg.keep_best_checkpoints,
        keep_last=cfg.keep_last_checkpoints,
    )
    callbacks = [PerfCallback(tokens_per_step), smart_ckpt]
    if cfg.do_sample_generations:
        callbacks.append(SampleGenCallback(tok, prompts, every_eval=cfg.sample_gen_every_eval))

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    # Resume
    resume_path = None
    if cfg.resume_from:
        resume_path = find_latest_checkpoint(run_dir) if cfg.resume_from == "latest" else cfg.resume_from
        print(f"[Resume] {resume_path}")

    # Persist config before training starts so a crash still leaves a record
    with open(f"{run_dir}/run_config.json", "w") as f:
        json.dump({
            "run_name": cfg.run_name,
            "timestamp": datetime.now().isoformat(),
            "config": asdict(cfg),
            "derived": {
                "tokens_per_step": tokens_per_step,
                "max_steps": max_steps,
                "checkpoint_steps": checkpoint_steps,
            },
        }, f, indent=2)

    trainer.train(resume_from_checkpoint=resume_path)

    # Save final artifacts
    tok.model_max_length = cfg.block_size
    tok.save_pretrained(run_dir)
    model.save_pretrained(run_dir)

    with open(f"{run_dir}/run_summary.json", "w") as f:
        json.dump({
            "finished_at": datetime.now().isoformat(),
            "global_step": trainer.state.global_step,
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
        }, f, indent=2)


if __name__ == "__main__":
    main()
