import os
import shutil
import json
import math
import time
import random
import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    LlamaConfig,
    LlamaTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# -----------------------------
# Config
# -----------------------------
@dataclass
class RunConfig:
    run_name: str = "sparknet-400m-v1"
    seed: int = 42

    tokenizer_path: str = "./tokenizer-v6"
    train_root: str = "datasets/sparknet-v6-pretrain"
    block_size: int = 1024

    # Model (~400M)
    hidden_size: int = 1024
    num_layers: int = 29
    num_heads: int = 16
    # GQA: fewer kv heads than q heads (recommended)
    num_kv_heads: int = 8
    intermediate_size: int = 2736
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    # Training
    bf16: bool = True
    per_device_train_batch_size: int = 32
    grad_accum: int = 16

    learning_rate: float = 1.5e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    scheduler: str = "cosine"
    max_grad_norm: float = 1.0

    # Token budget (production)
    target_tokens: int = 10_000_000_000  # total budget for the run
    checkpoint_tokens: List[int] = field(default_factory=lambda: [8_000_000_000])  # explicit token checkpoints

    # I/O + cadence
    logging_steps: int = 50
    eval_steps: int = 1000
    save_steps: int = 500
    save_total_limit: Optional[int] = 5

    dataloader_num_workers: int = 16
    dataloader_prefetch_factor: int = 4

    # Optional controls
    limit_shards: Optional[int] = None  # e.g. 1 for debugging; None for all shards
    resume_from: Optional[str] = None   # path to checkpoint dir, or "latest"
    do_sample_generations: bool = True
    sample_gen_every_eval: bool = True
    sample_prompts_path: Optional[str] = None  # JSON list of prompts

# -----------------------------
# Utilities
# -----------------------------
def set_tf32(enable: bool = True):
    # New API (PyTorch is deprecating allow_tf32 flags)
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32" if enable else "ieee"
    except Exception:
        pass

def resolve_repo_path(path: str) -> str:
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p)
    return str((REPO_ROOT / p).resolve())


def load_sparknet_tokenizer(tokenizer_path: str, padding_side: str = "right"):
    path = Path(tokenizer_path).expanduser()
    model_path = path / "tokenizer.model" if path.is_dir() else path
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")

    tok = LlamaTokenizer(vocab_file=str(model_path), legacy=True)
    tok.padding_side = padding_side
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok
    try:
        torch.backends.cudnn.conv.fp32_precision = "tf32" if enable else "ieee"
    except Exception:
        pass

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

    dsets = []
    for sd in shards:
        ds = Dataset.load_from_disk(sd)
        dsets.append(ds)

    train = concatenate_datasets(dsets) if len(dsets) > 1 else dsets[0]
    train.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
    print(f"[Data] train rows={len(train):,}")
    return train

def build_eval_wikitext(tok, block_size: int, eos_id: int) -> Dataset:
    eval_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    def tok_line(ex):
        ids = tok(ex["text"], add_special_tokens=False)["input_ids"]
        if not ids:
            return {"ids": []}
        if ids[-1] != eos_id:
            ids = ids + [eos_id]
        return {"ids": ids}

    eval_tok = eval_raw.map(tok_line, remove_columns=eval_raw.column_names)

    def pack(batch):
        flat = []
        for ids in batch["ids"]:
            flat.extend(ids)

        blocks = []
        for i in range(0, len(flat) - block_size, block_size):
            seg = flat[i : i + block_size]
            blocks.append(
                {"input_ids": seg, "labels": seg, "attention_mask": [1] * block_size}
            )

        if not blocks:
            return {"input_ids": [], "labels": [], "attention_mask": []}

        return {
            "input_ids": [b["input_ids"] for b in blocks],
            "labels": [b["labels"] for b in blocks],
            "attention_mask": [b["attention_mask"] for b in blocks],
        }

    packed = eval_tok.map(pack, batched=True, batch_size=1000, remove_columns=["ids"])
    packed.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
    print(f"[Eval] eval rows={len(packed):,}")
    return packed

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    out = Path(output_dir)
    if not out.exists():
        return None
    ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(ckpts[-1]) if ckpts else None

# -----------------------------
# Callbacks
# -----------------------------
class PerfCallback(TrainerCallback):
    def __init__(self, tokens_per_step: int):
        self.tokens_per_step = tokens_per_step
        self.t0 = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or self.t0 is None:
            return
        if state.global_step <= 0:
            return
        dt = time.time() - self.t0
        if dt <= 0:
            return
        steps_per_s = state.global_step / dt
        tok_per_s = steps_per_s * self.tokens_per_step
        logs = logs or {}
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

class TokenCheckpointCallback(TrainerCallback):
    def __init__(self, target_step_map: dict):
        self.target_step_map = dict(target_step_map)
        self._saved_steps = set()

    def on_step_end(self, args, state, control, **kwargs):
        if not self.target_step_map:
            return control
        step = state.global_step
        if step in self._saved_steps:
            return control
        if step in self.target_step_map:
            self._saved_steps.add(step)
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        if step not in self.target_step_map:
            return control
        token_target = self.target_step_map[step]
        src = os.path.join(args.output_dir, f"checkpoint-{step}")
        dst = os.path.join(args.output_dir, f"checkpoint-{token_target}-tokens")
        if os.path.exists(dst):
            return control
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        return control

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Optional path to a json config override.")
    parser.add_argument("--resume", type=str, default=None, help='Checkpoint path or "latest". Overrides config.resume_from.')
    parser.add_argument("--run-name", type=str, default=None, help="Override cfg.run_name (overrides config file)")
    args_cli = parser.parse_args()

    cfg = RunConfig()
    if args_cli.config:
        with open(args_cli.config, "r") as f:
            overrides = json.load(f)
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(f"Unknown config field: {k}")
            setattr(cfg, k, v)
    if args_cli.resume:
        cfg.resume_from = args_cli.resume
    if args_cli.run_name:
        cfg.run_name = args_cli.run_name

    cfg.tokenizer_path = resolve_repo_path(cfg.tokenizer_path)
    cfg.train_root = resolve_repo_path(cfg.train_root)
    if cfg.sample_prompts_path:
        cfg.sample_prompts_path = resolve_repo_path(cfg.sample_prompts_path)

    # Environment
    os.environ["HF_DATASETS_CACHE"] = str((REPO_ROOT / "cache").resolve())
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_tf32(True)
    set_seed(cfg.seed)

    run_dir = str((REPO_ROOT / "checkpoints" / cfg.run_name).resolve())
    log_dir = str((REPO_ROOT / "logs" / cfg.run_name).resolve())
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Tokenizer
    tok = load_sparknet_tokenizer(cfg.tokenizer_path, padding_side="right")
    eos_id = tok.eos_token_id

    # Data
    train_ds = load_prepacked(cfg.train_root, limit=cfg.limit_shards)
    eval_ds = build_eval_wikitext(tok, cfg.block_size, eos_id)

    # Model
    model_cfg = LlamaConfig(
        vocab_size=tok.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_kv_heads,  # GQA
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
        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
        print("Flash/SDPA enabled (if supported)")

    # Budget
    tokens_per_step = cfg.block_size * cfg.per_device_train_batch_size * cfg.grad_accum
    max_steps = math.ceil(cfg.target_tokens / tokens_per_step)
    print(f"[Budget] target_tokens={cfg.target_tokens:,} | tokens/step={tokens_per_step:,} | max_steps={max_steps:,}")
    checkpoint_tokens = cfg.checkpoint_tokens or []
    checkpoint_steps = [math.ceil(t / tokens_per_step) for t in checkpoint_tokens]
    checkpoint_step_map = dict(zip(checkpoint_steps, checkpoint_tokens))

    # TrainingArguments
    train_args = TrainingArguments(
        output_dir=run_dir,
        bf16=cfg.bf16,

        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,

        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.scheduler,

        max_steps=max_steps,
        max_grad_norm=cfg.max_grad_norm,

        logging_dir=log_dir,
        logging_steps=cfg.logging_steps,

        eval_strategy="steps",
        eval_steps=cfg.eval_steps,

        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        optim="adamw_torch_fused",

        report_to=["tensorboard"],
        remove_unused_columns=False,

        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=cfg.dataloader_prefetch_factor,
    )

    # Prompts for sample generation
    prompts = [
        "The meaning of life is",
        "In the future, AI will",
        "Dungeons and Dragons is a game where",
        "A good software engineer knows that",
    ]
    if cfg.sample_prompts_path:
        with open(cfg.sample_prompts_path, "r") as f:
            prompts = json.load(f)

    callbacks = [PerfCallback(tokens_per_step)]
    if checkpoint_step_map:
        callbacks.append(TokenCheckpointCallback(checkpoint_step_map))
    if cfg.do_sample_generations:
        callbacks.append(SampleGenCallback(tok, prompts, every_eval=cfg.sample_gen_every_eval))

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,  # prepacked dataset already has labels/attention_mask
        callbacks=callbacks,
    )

    # Resume logic
    resume_path = None
    if cfg.resume_from:
        if cfg.resume_from == "latest":
            resume_path = find_latest_checkpoint(run_dir)
        else:
            resume_path = cfg.resume_from
        print(f"[Resume] {resume_path}")

    # Persist run config + metadata up front (so a crash still leaves breadcrumbs)
    metadata = {
        "run_name": cfg.run_name,
        "timestamp": datetime.now().isoformat(),
        "config": asdict(cfg),
        "derived": {
            "tokens_per_step": tokens_per_step,
            "max_steps": max_steps,
            "checkpoint_tokens": checkpoint_tokens,
            "checkpoint_steps": checkpoint_steps,
        },
        "datasets": {
            "train_root": cfg.train_root,
            "limit_shards": cfg.limit_shards,
            "eval": "wikitext-2-raw-v1/validation packed to block_size",
        },
    }
    with open(f"{run_dir}/run_config.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Train
    trainer.train(resume_from_checkpoint=resume_path)

    # Save final artifacts
    tok.model_max_length = cfg.block_size
    tok.padding_side = "right"
    tok.save_pretrained(run_dir)
    model.save_pretrained(run_dir)

    # Final summary
    summary = {
        "finished_at": datetime.now().isoformat(),
        "global_step": trainer.state.global_step,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
    }
    with open(f"{run_dir}/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
