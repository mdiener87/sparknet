#!/usr/bin/env python3
"""
train_sft.py

SparkNet-400M SFT trainer for non-packed chat samples (one conversation per row).

The v4 regimen keeps the v3 data-format fixes and adds the missing recovery pieces:
- Held-out evaluation by default.
- Fixed prompt generation on every eval so checkpoint selection is behavioral.
- Stronger run metadata and dataset sanity summaries.
- Checkpoint cadence aligned with eval cadence.
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional

import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


ROLE_PREFIX = {
    "system": "### System:\n",
    "user": "### User:\n",
    "assistant": "### Assistant:\n",
}


# -----------------------------
# Config
# -----------------------------
@dataclass
class RunConfig:
    run_name: str = "sparknet-400m-v1-instruct-v4"
    seed: int = 42

    model_path: str = "checkpoints/sparknet-400m-v1"
    tokenizer_path: str = "./tokenizer-v6"
    train_root: str = "datasets/sft_chat_v3"
    block_size: int = 1024

    # Training
    bf16: bool = True
    gradient_checkpointing: bool = False

    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    grad_accum: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"
    max_grad_norm: float = 1.0

    # Token budget (approximate; uses block_size for budgeting)
    target_tokens: int = 300_000_000

    # Eval
    eval_fraction: float = 0.02
    eval_steps: int = 100

    # Sample generations
    sample_prompts_path: str = "configs/sparknet-400m/sft_eval_prompts_v4.json"
    sample_max_new_tokens: int = 120
    sample_temperature: float = 0.0
    sample_top_p: float = 0.9
    sample_top_k: int = 0
    sample_repetition_penalty: float = 1.1
    sample_no_repeat_ngram_size: int = 3

    # I/O + cadence
    output_dir: str = "checkpoints/sparknet-400m-v1-instruct-v4"
    logging_steps: int = 25
    save_steps: int = 100
    save_total_limit: Optional[int] = 4

    dataloader_num_workers: int = 8
    dataloader_prefetch_factor: int = 4

    # Optional controls
    limit_shards: Optional[int] = None
    resume_from: Optional[str] = None


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
    shards = sorted(str(p) for p in root.glob("shard-*") if p.is_dir())
    if not shards:
        raise FileNotFoundError(f"No shard-* dirs found under: {train_root}")
    return shards


def load_dataset_shards(train_root: str, limit: Optional[int] = None) -> Dataset:
    shards = list_shards(train_root)
    if limit is not None:
        shards = shards[:limit]
    print(f"[Data] Loading {len(shards)} shard(s) from {train_root}")
    dsets = [Dataset.load_from_disk(sd) for sd in shards]
    ds = concatenate_datasets(dsets) if len(dsets) > 1 else dsets[0]
    print(f"[Data] rows={len(ds):,}")
    return ds


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
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel  # type: ignore

        _ = (sdpa_kernel, SDPBackend)
        print("[Attn] SDPA available (backend chosen dynamically per call).")
    except Exception:
        print("[Attn] SDPA backend API not available; using default attention behavior.")


def quantile(values: List[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    idx = max(0, min(idx, len(ordered) - 1))
    return int(ordered[idx])


def summarize_rows(ds: Dataset, block_size: int, n: int = 1024) -> Dict[str, int]:
    n = min(n, len(ds))
    if n == 0:
        raise ValueError("Dataset is empty.")

    lengths: List[int] = []
    supervised_counts: List[int] = []
    first_supervised: List[int] = []

    for i in range(n):
        ex = ds[i]
        if "input_ids" not in ex or "labels" not in ex:
            raise ValueError(f"Row {i} missing input_ids/labels keys.")

        ids = ex["input_ids"]
        labels = ex["labels"]
        if len(ids) != len(labels):
            raise ValueError(f"Row {i} length mismatch: input_ids={len(ids)} labels={len(labels)}")
        if len(ids) > block_size:
            raise ValueError(f"Row {i} too long: {len(ids)} > block_size={block_size}")

        active = [idx for idx, label in enumerate(labels) if label != -100]
        if active:
            supervised_counts.append(len(active))
            first_supervised.append(active[0])
        lengths.append(len(ids))

    if not supervised_counts:
        raise ValueError(
            "No supervised labels found in the validation window. "
            "This usually means the builder masked everything."
        )

    stats = {
        "rows_checked": n,
        "supervised_rows": len(supervised_counts),
        "seq_len_p50": int(median(lengths)),
        "seq_len_p90": quantile(lengths, 0.90),
        "supervised_tokens_p50": int(median(supervised_counts)),
        "supervised_tokens_p90": quantile(supervised_counts, 0.90),
        "first_supervised_p50": int(median(first_supervised)),
        "first_supervised_p90": quantile(first_supervised, 0.90),
    }
    return stats


def build_prompt(messages: List[Dict[str, str]]) -> str:
    chunks: List[str] = []
    for msg in messages:
        role = msg["role"]
        if role not in ROLE_PREFIX:
            raise ValueError(f"Unsupported role in prompt suite: {role}")
        chunks.append(ROLE_PREFIX[role] + msg["content"].strip() + "\n")
    chunks.append(ROLE_PREFIX["assistant"])
    return "".join(chunks)


def stop_at_next_role_header(text: str) -> str:
    cut = None
    for marker in ("\n### User:", "\n### System:", "\n### Assistant:"):
        idx = text.find(marker)
        if idx != -1:
            cut = idx if cut is None else min(cut, idx)
    if cut is not None:
        text = text[:cut]
    return text.strip()


def load_prompt_suite(path: str) -> List[Dict[str, object]]:
    with open(path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Prompt suite must be a JSON list: {path}")

    prompts: List[Dict[str, object]] = []
    for idx, item in enumerate(raw):
        prompt_id = f"prompt_{idx + 1:02d}"
        if isinstance(item, str):
            prompts.append(
                {
                    "id": prompt_id,
                    "messages": [{"role": "user", "content": item.strip()}],
                }
            )
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Prompt entry {idx} must be a string or object.")

        messages = item.get("messages")
        if messages is None:
            user_text = str(item.get("user", "")).strip()
            if not user_text:
                raise ValueError(f"Prompt entry {idx} is missing `messages` or `user`.")
            messages = []
            system_text = str(item.get("system", "")).strip()
            if system_text:
                messages.append({"role": "system", "content": system_text})
            messages.append({"role": "user", "content": user_text})

        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Prompt entry {idx} has no usable messages.")

        norm_messages: List[Dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(f"Prompt entry {idx} contains a non-dict message.")
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role not in ROLE_PREFIX or not content:
                raise ValueError(f"Prompt entry {idx} has an invalid message: {message}")
            norm_messages.append({"role": role, "content": content})

        prompts.append({"id": str(item.get("id") or prompt_id), "messages": norm_messages})
    return prompts


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


# -----------------------------
# Collator
# -----------------------------
class SFTPadCollator:
    def __init__(self, tokenizer, block_size: int):
        self.tok = tokenizer
        self.block_size = int(block_size)
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        bs = len(features)
        max_len = max(len(f["input_ids"]) for f in features)
        max_len = min(max_len, self.block_size)

        input_ids = torch.full((bs, max_len), fill_value=self.tok.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
        labels = torch.full((bs, max_len), fill_value=-100, dtype=torch.long)

        for i, f in enumerate(features):
            ids = f["input_ids"][:max_len]
            lbs = f["labels"][:max_len]
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :n] = 1
            labels[i, :n] = torch.tensor(lbs, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# -----------------------------
# Callbacks
# -----------------------------
class PerfCallback(TrainerCallback):
    def __init__(self, approx_tokens_per_step: int):
        self.tokens_per_step = approx_tokens_per_step
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
        print(f"[Perf] ~{tok_per_s:,.0f} tok/s (approx) | step={state.global_step:,}")
        self.last_t = now
        self.last_step = state.global_step


class SampleGenCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        prompts: List[Dict[str, object]],
        output_dir: str,
        max_context: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
    ):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.output_path = os.path.join(output_dir, "sample_generations.jsonl")
        self.max_context = max_context
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.last_logged_step = None

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        if self.last_logged_step == state.global_step:
            return

        device = next(model.parameters()).device
        model.eval()
        do_sample = self.temperature > 0

        print("\n[SampleGen] =====")
        records: List[Dict[str, object]] = []
        with torch.no_grad():
            for prompt in self.prompts:
                prompt_text = build_prompt(prompt["messages"])  # type: ignore[index]
                inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=self.max_context - self.max_new_tokens).to(device)
                input_len = inputs["input_ids"].shape[1]

                gen_kwargs = dict(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=do_sample,
                    repetition_penalty=self.repetition_penalty,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                if do_sample:
                    gen_kwargs["temperature"] = self.temperature
                    gen_kwargs["top_p"] = self.top_p
                    gen_kwargs["top_k"] = self.top_k

                output_ids = model.generate(**gen_kwargs)[0]
                new_tokens = output_ids[input_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                text = stop_at_next_role_header(text)

                prompt_id = str(prompt["id"])
                print(f"[SampleGen] {prompt_id}\n{text}\n")
                records.append(
                    {
                        "step": state.global_step,
                        "prompt_id": prompt_id,
                        "response": text,
                    }
                )
        print("[SampleGen] =====\n")

        with open(self.output_path, "a") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        self.last_logged_step = state.global_step


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Optional path to a JSON config override.")
    parser.add_argument("--resume", type=str, default=None, help='Checkpoint path or "latest". Overrides cfg.resume_from.')
    parser.add_argument("--run-name", type=str, default=None, help="Override cfg.run_name")
    parser.add_argument("--eval-fraction", type=float, default=None, help="Override cfg.eval_fraction")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    args_cli = parser.parse_args()

    default_cfg = RunConfig()
    cfg = RunConfig()

    if args_cli.config:
        with open(args_cli.config, "r") as f:
            overrides = json.load(f)
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown config field: {key}")
            setattr(cfg, key, value)

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

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    set_seed(cfg.seed)
    set_tf32(True)
    check_bf16_or_die(cfg.bf16)

    os.makedirs(cfg.output_dir, exist_ok=True)

    full = load_dataset_shards(cfg.train_root, cfg.limit_shards)
    full_stats = summarize_rows(full, cfg.block_size)
    print(f"[Data] summary={json.dumps(full_stats, sort_keys=True)}")

    train_ds = full
    eval_ds = None
    if cfg.eval_fraction and cfg.eval_fraction > 0:
        splits = full.train_test_split(test_size=cfg.eval_fraction, seed=cfg.seed)
        train_ds = splits["train"]
        eval_ds = splits["test"]
        train_stats = summarize_rows(train_ds, cfg.block_size)
        eval_stats = summarize_rows(eval_ds, cfg.block_size)
        print(f"[Data] train={len(train_ds):,} eval={len(eval_ds):,} (eval_fraction={cfg.eval_fraction})")
        print(f"[Data] train_summary={json.dumps(train_stats, sort_keys=True)}")
        print(f"[Data] eval_summary={json.dumps(eval_stats, sort_keys=True)}")

    tok = load_sparknet_tokenizer(cfg.tokenizer_path, padding_side="right")

    model = AutoModelForCausalLM.from_pretrained(cfg.model_path)

    def die(msg: str):
        raise RuntimeError(f"[Harmony] {msg}")

    emb_vocab = model.get_input_embeddings().weight.shape[0]
    tok_vocab = len(tok)
    if emb_vocab != tok_vocab:
        die(
            f"Embedding vocab ({emb_vocab}) != tokenizer vocab ({tok_vocab}). "
            "Did you resize embeddings or save the wrong tokenizer?"
        )
    if tok.eos_token_id is None or tok.pad_token_id is None:
        die("Tokenizer missing eos/pad ids.")

    model.config.eos_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id
    if tok.bos_token_id is not None:
        model.config.bos_token_id = tok.bos_token_id

    print(
        f"[Harmony] tok_vocab={tok_vocab} emb_vocab={emb_vocab} "
        f"eos={tok.eos_token_id} pad={tok.pad_token_id} bos={tok.bos_token_id}"
    )

    try:
        model.config.attn_implementation = "sdpa"
    except Exception:
        pass
    try_log_sdpa_backend()

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        print("[Train] Gradient checkpointing enabled")

    ws = world_size()
    approx_tokens_per_step = cfg.block_size * cfg.per_device_train_batch_size * cfg.grad_accum * ws
    max_steps = math.ceil(cfg.target_tokens / approx_tokens_per_step)
    effective_batch = cfg.per_device_train_batch_size * cfg.grad_accum * ws
    steps_per_epoch = math.ceil(len(train_ds) / max(1, effective_batch))
    planned_epochs = max_steps / max(1, steps_per_epoch)
    print(
        f"[Budget] target_tokens={cfg.target_tokens:,} | approx tokens/step={approx_tokens_per_step:,} | "
        f"max_steps={max_steps:,} | world_size={ws} | effective_batch={effective_batch} | "
        f"planned_epochs={planned_epochs:.2f}"
    )
    warmup_steps = int(cfg.warmup_ratio * max_steps)
    print(f"[LR] warmup_ratio={cfg.warmup_ratio} -> warmup_steps={warmup_steps:,}")

    resume_from = cfg.resume_from
    if resume_from == "latest":
        resume_from = find_latest_checkpoint(cfg.output_dir)
        if resume_from:
            print(f"[Resume] latest checkpoint: {resume_from}")

    evaluation_strategy = "no"
    eval_steps = None
    load_best = False
    metric_for_best = None
    greater_is_better = None
    prompts: List[Dict[str, object]] = []

    if eval_ds is not None:
        evaluation_strategy = "steps"
        eval_steps = cfg.eval_steps or cfg.save_steps
        if cfg.save_steps % eval_steps != 0:
            print(
                f"[Eval] save_steps ({cfg.save_steps}) not multiple of eval_steps ({eval_steps}); "
                "setting eval_steps=save_steps."
            )
            eval_steps = cfg.save_steps
        load_best = True
        metric_for_best = "eval_loss"
        greater_is_better = False
        prompts = load_prompt_suite(cfg.sample_prompts_path)
        print(f"[Eval] loaded {len(prompts)} prompt(s) from {cfg.sample_prompts_path}")

    collator = SFTPadCollator(tok, block_size=cfg.block_size)

    train_args = TrainingArguments(
        output_dir=cfg.output_dir,
        bf16=cfg.bf16,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.scheduler,
        max_steps=max_steps,
        logging_dir=f"logs/{cfg.run_name}",
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy=evaluation_strategy,
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
        save_safetensors=True,
    )

    callbacks: List[TrainerCallback] = [PerfCallback(approx_tokens_per_step)]
    if prompts:
        callbacks.append(
            SampleGenCallback(
                tokenizer=tok,
                prompts=prompts,
                output_dir=cfg.output_dir,
                max_context=cfg.block_size,
                max_new_tokens=cfg.sample_max_new_tokens,
                temperature=cfg.sample_temperature,
                top_p=cfg.sample_top_p,
                top_k=cfg.sample_top_k,
                repetition_penalty=cfg.sample_repetition_penalty,
                no_repeat_ngram_size=cfg.sample_no_repeat_ngram_size,
            )
        )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=resume_from)

    tok.save_pretrained(cfg.output_dir)
    trainer.save_model(cfg.output_dir)

    metadata = {
        "run_name": cfg.run_name,
        "output_dir": cfg.output_dir,
        "model_path": cfg.model_path,
        "tokenizer_path": cfg.tokenizer_path,
        "train_root": cfg.train_root,
        "block_size": cfg.block_size,
        "target_tokens": cfg.target_tokens,
        "world_size": ws,
        "approx_tokens_per_step": approx_tokens_per_step,
        "effective_batch_size": effective_batch,
        "steps_per_epoch": steps_per_epoch,
        "planned_epochs": planned_epochs,
        "max_steps": max_steps,
        "bf16": cfg.bf16,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "grad_accum": cfg.grad_accum,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio,
        "warmup_steps": warmup_steps,
        "scheduler": cfg.scheduler,
        "max_grad_norm": cfg.max_grad_norm,
        "eval_fraction": cfg.eval_fraction,
        "eval_steps": eval_steps,
        "save_steps": cfg.save_steps,
        "sample_prompts_path": cfg.sample_prompts_path if prompts else None,
        "sample_max_new_tokens": cfg.sample_max_new_tokens,
        "sample_temperature": cfg.sample_temperature,
        "sample_repetition_penalty": cfg.sample_repetition_penalty,
        "dataset_summary": full_stats,
        "torch_version": torch.__version__,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(cfg.output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete")


if __name__ == "__main__":
    main()
