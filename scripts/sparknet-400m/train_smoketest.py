import os
import json
import math
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
)

# --------------------------------------------------------------------------------------
# Environment / Repro
# --------------------------------------------------------------------------------------
os.environ["HF_DATASETS_CACHE"] = os.path.expanduser("~/projects/sparknet/cache")
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------------------
# Run / Mode
# --------------------------------------------------------------------------------------
RUN_NAME = "sparknet-400m-smoketest"

# Flip this to False when you’re ready to bake for real
SMOKE_TEST = True

# If SMOKE_TEST=True, we’ll train only a small number of steps (overnight-safe).
SMOKE_MAX_STEPS = 300

# Optionally limit to first N shards for smoke tests (set None for all).
SMOKE_SHARD_LIMIT = 1  # e.g. 1 shard for fast validation; set None to use all shards even in smoke test.

# --------------------------------------------------------------------------------------
# Seed / Block size
# --------------------------------------------------------------------------------------
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

block_size = 1024  # matches your packed dataset blocks

# --------------------------------------------------------------------------------------
# Tokenizer (v6 SentencePiece)
# --------------------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained("./tokenizer-v6", padding_side="right", use_fast=False)
# LLaMA tokenizers often have no pad token by default — set it safely to EOS for training batches.
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
eos_id = tok.eos_token_id

# --------------------------------------------------------------------------------------
# Load Packed Sharded Dataset
# Expecting each shard to already contain: input_ids, labels, attention_mask (packed 1024)
# --------------------------------------------------------------------------------------
def list_shard_dirs(root: str) -> List[str]:
    root_p = Path(root)
    shard_dirs = sorted([str(p) for p in root_p.glob("shard-*") if p.is_dir()])
    if not shard_dirs:
        raise FileNotFoundError(f"No shard-* dirs found under: {root}")
    return shard_dirs

def load_prepacked_shards(root: str, limit: int | None = None) -> Dataset:
    shard_dirs = list_shard_dirs(root)
    if limit is not None:
        shard_dirs = shard_dirs[:limit]

    print(f"[Data] Loading {len(shard_dirs)} shard(s) from {root}")
    dsets = []
    for sd in shard_dirs:
        ds = Dataset.load_from_disk(sd)
        dsets.append(ds)

    # Concatenate into one logical training dataset (Arrow-backed / memory-mapped)
    train = concatenate_datasets(dsets) if len(dsets) > 1 else dsets[0]

    # Ensure torch formatting
    train.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
    print(f"[Data] train rows={len(train):,}")
    return train

train_root = "datasets/sparknet-v6-pretrain"
train_ds = load_prepacked_shards(
    train_root,
    limit=(SMOKE_SHARD_LIMIT if SMOKE_TEST else None),
)

# --------------------------------------------------------------------------------------
# Eval Dataset (Wikitext-2 validation), tokenized + packed to 1024
# --------------------------------------------------------------------------------------
def build_eval_dataset(block: int) -> Dataset:
    eval_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    def tok_line(ex):
        ids = tok(ex["text"], add_special_tokens=False)["input_ids"]
        if not ids:
            return {"ids": []}
        if ids[-1] != eos_id:
            ids = ids + [eos_id]
        return {"ids": ids}

    eval_tok = eval_raw.map(tok_line, remove_columns=eval_raw.column_names)

    def pack_examples(batch):
        flat = []
        for ids in batch["ids"]:
            flat.extend(ids)

        blocks = []
        for i in range(0, len(flat) - block, block):
            seg = flat[i : i + block]
            blocks.append(
                {
                    "input_ids": seg,
                    "labels": seg,
                    "attention_mask": [1] * block,
                }
            )

        if not blocks:
            return {"input_ids": [], "labels": [], "attention_mask": []}

        return {
            "input_ids": [b["input_ids"] for b in blocks],
            "labels": [b["labels"] for b in blocks],
            "attention_mask": [b["attention_mask"] for b in blocks],
        }

    eval_packed = eval_tok.map(
        pack_examples,
        batched=True,
        batch_size=1000,
        remove_columns=["ids"],
    )
    eval_packed.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
    print(f"[Eval] eval rows={len(eval_packed):,}")
    return eval_packed

eval_ds = build_eval_dataset(block_size)

# --------------------------------------------------------------------------------------
# Model (LLaMA-style ~398M params)
#  hidden_size=1024, layers=29, heads=16, intermediate=2736
#  RoPE, RMSNorm, SwiGLU come "for free" via LlamaConfig
# --------------------------------------------------------------------------------------
model_cfg = LlamaConfig(
    vocab_size=tok.vocab_size,
    hidden_size=1024,
    intermediate_size=2736,
    num_hidden_layers=29,
    num_attention_heads=16,
    num_key_value_heads=16,   # keep simple for v1 (no GQA). You can set 8 later.
    max_position_embeddings=block_size,  # with RoPE this is mostly a training-time nominal limit
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    attention_bias=False,
    mlp_bias=False,
    tie_word_embeddings=True,
)

model = AutoModelForCausalLM.from_config(model_cfg)

# memory/perf toggles
model.config.use_cache = False

# Gradient checkpointing is usually worth it at 400M
model.gradient_checkpointing_enable()

# Use PyTorch SDPA (Flash if available)
model.config.attn_implementation = "sdpa"
if hasattr(torch.backends.cuda, "sdp_kernel"):
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
    print("Flash/SDPA enabled (if supported)")

# --------------------------------------------------------------------------------------
# TrainingArguments
# --------------------------------------------------------------------------------------
# You will likely need to tune these based on Spark memory; these are safe-ish defaults.
per_device_train_batch_size = 8
gradient_accumulation_steps = 64

tokens_per_step = block_size * per_device_train_batch_size * gradient_accumulation_steps
print(f"[Budget] tokens/step={tokens_per_step:,}")

if SMOKE_TEST:
    max_steps = SMOKE_MAX_STEPS
    target_tokens = max_steps * tokens_per_step
    print(f"[Smoke] max_steps={max_steps:,} => ~{target_tokens:,} tokens")
else:
    # Example: 4B tokens target
    target_tokens = 4_000_000_000
    max_steps = math.ceil(target_tokens / tokens_per_step)
    print(f"[Bake] target_tokens={target_tokens:,} | max_steps={max_steps:,}")

args = TrainingArguments(
    output_dir=f"checkpoints/{RUN_NAME}",
    bf16=True,

    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,

    # Smoke-safe LR; for full bake you may raise a bit (e.g. 2e-4 to 3e-4) after stability confirmed.
    learning_rate=3e-4 if SMOKE_TEST else 2e-4,
    weight_decay=0.1,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",

    max_steps=max_steps,
    max_grad_norm=1.0,

    logging_dir=f"logs/{RUN_NAME}",
    logging_steps=25 if SMOKE_TEST else 250,

    eval_strategy="steps",
    eval_steps=50 if SMOKE_TEST else 2000,

    save_strategy="steps",
    save_steps=100 if SMOKE_TEST else 2000,
    save_total_limit=3,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    optim="adamw_torch_fused",

    report_to=["tensorboard"],
    remove_unused_columns=False,

    dataloader_num_workers=16,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    dataloader_prefetch_factor=4,
)

# --------------------------------------------------------------------------------------
# Optional: simple callback to print throughput-ish info
# (Keep your existing v4/v5 callbacks if you like)
# --------------------------------------------------------------------------------------
class StepTimingCallback(TrainerCallback):
    def __init__(self):
        self._t0 = None

    def on_step_begin(self, args, state, control, **kwargs):
        if self._t0 is None:
            self._t0 = datetime.now()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        # lightweight timing (not perfect, but useful)
        dt = (datetime.now() - self._t0).total_seconds()
        if state.global_step > 0 and dt > 0:
            steps_per_s = state.global_step / dt
            tok_per_s = steps_per_s * tokens_per_step
            print(f"[Perf] ~{tok_per_s:,.0f} tok/s | step={state.global_step:,}")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=default_data_collator,  # dataset already has labels/attention_mask
    callbacks=[StepTimingCallback()],
)

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
os.makedirs(f"checkpoints/{RUN_NAME}", exist_ok=True)
os.makedirs(f"logs/{RUN_NAME}", exist_ok=True)

if __name__ == "__main__":
    trainer.train()

    # Save tokenizer + model
    tok.model_max_length = block_size
    tok.padding_side = "right"
    tok.save_pretrained(f"checkpoints/{RUN_NAME}")
    model.save_pretrained(f"checkpoints/{RUN_NAME}")

    metadata = {
        "run_name": RUN_NAME,
        "timestamp": datetime.now().isoformat(),
        "smoke_test": SMOKE_TEST,
        "params": {
            "arch": "llama",
            "hidden_size": model_cfg.hidden_size,
            "num_hidden_layers": model_cfg.num_hidden_layers,
            "num_attention_heads": model_cfg.num_attention_heads,
            "intermediate_size": model_cfg.intermediate_size,
            "context_length": block_size,
            "token_budget": int(target_tokens),
        },
        "datasets": {
            "train_root": train_root,
            "smoke_shard_limit": SMOKE_SHARD_LIMIT if SMOKE_TEST else None,
        },
        "notes": "SparkNet-400M v1 | LLaMA-style (RoPE, RMSNorm, SwiGLU), SDPA attention.",
    }
    with open(f"checkpoints/{RUN_NAME}/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
