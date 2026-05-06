#!/usr/bin/env python3
"""
Run a 250M-token canary at the selected SparkNet-410M v1 learning rate.

This is the final gate before the full 10B run: same architecture, tokenizer,
corpus eval shard, optimizer, grad_accum=32, and production cosine_with_min_lr
schedule. It is long enough to observe warmup exit and early post-warmup loss
shape without spending full-run compute.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from packaging.version import Version
from transformers import Trainer, TrainerCallback, TrainingArguments, default_data_collator, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hparam_utils import (
    REPO_ROOT,
    build_model,
    load_eval_prepacked,
    load_prepacked,
    load_sparknet_tokenizer,
    param_count,
    resolve_repo_path,
    set_tf32,
    setup_env,
)

TARGET_TOKENS = 250_000_000
BLOCK_SIZE = 1024
PER_DEVICE_BATCH = 32
GRAD_ACCUM = 32
WARMUP_RATIO = 0.02
COSINE_MIN_LR_RATIO = 0.1
LOG_EVERY_STEPS = 10
EVAL_EVERY_STEPS = 50


class CanaryRecorder(TrainerCallback):
    def __init__(self):
        self.train_records = []
        self.eval_records = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_records.append({
                "step": state.global_step,
                "loss": logs["loss"],
                "grad_norm": logs.get("grad_norm"),
                "lr": logs.get("learning_rate"),
            })

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.eval_records.append({
                "step": state.global_step,
                "eval_loss": metrics.get("eval_loss"),
            })


def has_post_warmup_spike(records: list, warmup_steps: int) -> bool:
    pre = [r for r in records if r["step"] <= warmup_steps]
    post = [r for r in records if r["step"] > warmup_steps]
    if not pre or not post:
        return False
    baseline = min(r["loss"] for r in pre if r.get("loss") is not None)
    check_window = post[:max(1, len(post) // 3)]
    return any(r.get("loss") is not None and r["loss"] > baseline * 1.2 for r in check_window)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--train-root", default="datasets/sparknet-410m-v1-pretrain")
    parser.add_argument("--eval-root", default="datasets/sparknet-410m-v1-pretrain-eval")
    parser.add_argument("--tokenizer-path", default="./tokenizer-v8")
    parser.add_argument("--limit-shards", type=int, default=None)
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    args = parser.parse_args()

    setup_env()
    set_tf32(True)
    set_seed(42)

    import transformers as _tf
    if Version(_tf.__version__) < Version("4.38.0"):
        raise RuntimeError(f"cosine_with_min_lr requires transformers>=4.38, found {_tf.__version__}")

    tok = load_sparknet_tokenizer(resolve_repo_path(args.tokenizer_path))
    train_ds = load_prepacked(resolve_repo_path(args.train_root), limit_shards=args.limit_shards)
    eval_ds = load_eval_prepacked(resolve_repo_path(args.eval_root))

    model = build_model(vocab_size=len(tok), block_size=BLOCK_SIZE)
    print(f"Parameters: {param_count(model) / 1e6:.1f}M")

    tokens_per_step = BLOCK_SIZE * PER_DEVICE_BATCH * GRAD_ACCUM
    max_steps = math.ceil(args.target_tokens / tokens_per_step)
    warmup_steps = math.ceil(max_steps * WARMUP_RATIO)
    lr_tag = f"{args.lr:.0e}".replace("-0", "-").replace("+0", "")
    run_name = f"hparam-410m-canary-lr{lr_tag}"
    run_dir = str(REPO_ROOT / "checkpoints" / run_name)
    log_dir = str(REPO_ROOT / "logs" / run_name)

    print(f"Canary LR: {args.lr:.2e}")
    print(f"Tokens/step: {tokens_per_step:,} | steps={max_steps} | warmup_steps={warmup_steps}")

    train_args = TrainingArguments(
        output_dir=run_dir,
        bf16=True,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=args.lr,
        weight_decay=0.1,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": COSINE_MIN_LR_RATIO},
        max_grad_norm=1.0,
        max_steps=max_steps,
        logging_dir=log_dir,
        logging_steps=LOG_EVERY_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_EVERY_STEPS,
        save_strategy="no",
        optim="adamw_torch_fused",
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
    )

    recorder = CanaryRecorder()
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds.shuffle(seed=42),
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        callbacks=[recorder],
    )
    trainer.train()
    final_metrics = trainer.evaluate()

    spike = has_post_warmup_spike(recorder.train_records, warmup_steps)
    final_train_loss = recorder.train_records[-1]["loss"] if recorder.train_records else None
    final_eval_loss = final_metrics.get("eval_loss")
    out_dir = REPO_ROOT / "logs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "canary_summary.json"
    with open(out_path, "w") as f:
        json.dump({
            "lr": args.lr,
            "target_tokens": args.target_tokens,
            "tokens_per_step": tokens_per_step,
            "max_steps": max_steps,
            "warmup_ratio": WARMUP_RATIO,
            "warmup_steps": warmup_steps,
            "cosine_min_lr_ratio": COSINE_MIN_LR_RATIO,
            "final_train_loss": final_train_loss,
            "final_eval_loss": final_eval_loss,
            "post_warmup_spike": spike,
            "train_records": recorder.train_records,
            "eval_records": recorder.eval_records,
        }, f, indent=2)

    print(f"Final train loss: {final_train_loss}")
    print(f"Final eval loss : {final_eval_loss}")
    print(f"Post-warmup spike: {'YES' if spike else 'no'}")
    print(f"Summary: {out_path}")


if __name__ == "__main__":
    main()
