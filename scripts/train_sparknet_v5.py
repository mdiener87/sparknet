# train_sparknet_v5.py
import os
import json
import math
import random
import warnings
from typing import Dict, Any, Iterable

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    GPT2Config,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
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

# RUN NAME
### --- UPDATED FOR v5 ---
RUN_NAME = "sparknet-70m-v5"

# --------------------------------------------------------------------------------------
# Load Config
# --------------------------------------------------------------------------------------
with open("configs/datasets_v5.json") as f:
    cfg = json.load(f)

seed = int(cfg.get("seed", 42))
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

block_size = int(cfg.get("context_length", 1024))

# --------------------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------------------
### --- UPDATED FOR v5 ---
# Use custom tokenizer (trained from finepdfs/dclm/fineweb)
tok = AutoTokenizer.from_pretrained("./tokenizer-v5", padding_side="right")
tok.pad_token = tok.eos_token
eos_id = tok.eos_token_id

# --------------------------------------------------------------------------------------
# Load Static 1B-Token Dataset
# --------------------------------------------------------------------------------------
### --- UPDATED FOR v5 ---
# No streaming or interleave. Use prepacked 1024-block dataset.
train_ds = Dataset.load_from_disk("datasets/sparknet-v5-1b")

# --------------------------------------------------------------------------------------
# Eval Dataset (unchanged)
# --------------------------------------------------------------------------------------
def build_eval_dataset(block: int) -> Dataset:
    eval_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    def tok_line(ex):
        ids = tok(ex["text"], add_special_tokens=False)["input_ids"]
        if not ids or ids[-1] != eos_id:
            ids = ids + [eos_id]
        return {"ids": ids}

    eval_tok = eval_raw.map(tok_line, remove_columns=eval_raw.column_names)

    def pack_examples(batch):
        flat = []
        for ids in batch["ids"]:
            flat.extend(ids)
        blocks = []
        for i in range(0, len(flat) - block, block):
            seg = flat[i:i+block]
            blocks.append({
                "input_ids": seg,
                "labels": seg,
                "attention_mask": [1]*block
            })
        if not blocks:
            return {"input_ids": [], "labels": [], "attention_mask": []}
        return {
            "input_ids": [b["input_ids"] for b in blocks],
            "labels": [b["labels"] for b in blocks],
            "attention_mask": [b["attention_mask"] for b in blocks],
        }

    eval_packed = eval_tok.map(pack_examples, batched=True, batch_size=1000, remove_columns=["ids"])
    eval_packed.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
    return eval_packed

eval_ds = build_eval_dataset(block_size)

# --------------------------------------------------------------------------------------
# Model (~70M GPT-2-style)
# --------------------------------------------------------------------------------------
### --- UPDATED FOR v5 ---
model_cfg = GPT2Config(
    vocab_size=tok.vocab_size,     # use custom tokenizer vocab
    n_positions=block_size,
    n_embd=512,
    n_layer=12,
    n_head=8,
    resid_pdrop=0.1,               # dropout ON
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    tie_word_embeddings=True
)

model = AutoModelForCausalLM.from_config(model_cfg)

model.gradient_checkpointing_disable()

model.config.use_cache = False
model.config.attn_implementation = "sdpa"

if hasattr(torch.backends.cuda, "sdp_kernel"):
    torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_mem_efficient=True, enable_math=False
    )
    print("Flash Attention enabled")

# --------------------------------------------------------------------------------------
# TrainingArguments
# --------------------------------------------------------------------------------------
per_device_train_batch_size = 32
gradient_accumulation_steps = 2

tokens_per_step = block_size * per_device_train_batch_size * gradient_accumulation_steps

### --- UPDATED FOR v5 ---
# 1 billion tokens
target_tokens = 1_000_000_000
max_steps = math.ceil(target_tokens / tokens_per_step)

print(f"[Budget] target_tokens={target_tokens:,} | tokens/step={tokens_per_step:,} | max_steps={max_steps:,}")

collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

args = TrainingArguments(
    output_dir=f"checkpoints/{RUN_NAME}",
    bf16=True,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,

    ### --- UPDATED FOR v5 ---
    learning_rate=1e-4,
    weight_decay=0.1,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",

    max_steps=max_steps,
    max_grad_norm=1.0,

    logging_dir=f"logs/{RUN_NAME}",
    logging_steps=250,
    eval_strategy="steps",
    eval_steps=2000,
    save_steps=2000,
    save_strategy="steps",
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
# Trainer + Callbacks (unchanged)
# --------------------------------------------------------------------------------------
from transformers import TrainerCallback
import time, os
from torch.utils.tensorboard import SummaryWriter

# (callbacks unchanged, omitted for brevity — keep them exactly as in your v4)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
)

# ... callbacks here unchanged ...

# --------------------------------------------------------------------------------------
# Training Main Init
# --------------------------------------------------------------------------------------
os.makedirs(f"checkpoints/{RUN_NAME}", exist_ok=True)
os.makedirs(f"logs/{RUN_NAME}", exist_ok=True)

from datetime import datetime

if __name__ == "__main__":
    trainer.train()

    tok.model_max_length = block_size
    tok.padding_side = "right"
    tok.save_pretrained(f"checkpoints/{RUN_NAME}")

    model.save_pretrained(f"checkpoints/{RUN_NAME}")

    metadata = {
        "run_name": RUN_NAME,
        "timestamp": datetime.now().isoformat(),
        "params": {
            "n_embd": model_cfg.n_embd, "n_layer": model_cfg.n_layer, "n_head": model_cfg.n_head,
            "context_length": block_size, "token_budget": target_tokens
        },
        "datasets": ["sparknet-v5-1b"],
        "notes": "V5 | Custom tokenizer, dropout, cosine LR, static 1B token dataset."
    }
    with open(f"checkpoints/{RUN_NAME}/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
