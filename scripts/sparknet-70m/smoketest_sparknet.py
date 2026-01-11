import os
import math, json
from datasets import load_dataset, interleave_datasets
from transformers import (
    AutoTokenizer, GPT2Config, AutoModelForCausalLM,
    Trainer, TrainingArguments
)

import datasets
os.environ["HF_DATASETS_CACHE"] = os.path.expanduser("~/projects/sparknet/cache")
os.environ["HF_DATASETS_OFFLINE"] = "0"  # optional, ensure it can hit hub if needed




# Load config
with open("configs/datasets_smoketest.json") as f:
    cfg = json.load(f)

# Load and mix datasets
from datasets import load_dataset, interleave_datasets

datasets = []
for m in cfg["mix"]:
    name = m["name"]
    config = m.get("config", None)
    try:
        ds = load_dataset(name, config, split="train")
        datasets.append(ds)
        print(f"Loaded {name} ({len(ds)} samples)")
    except Exception as e:
        print(f"⚠️ Could not load {name}: {e}")
        continue

if not datasets:
    raise RuntimeError("No datasets loaded — please check dataset names/configs.")

mixture = interleave_datasets(
    datasets, probabilities=[m["prob"] for m in cfg["mix"][:len(datasets)]], seed=cfg["seed"]
)


# === Tokenization Setup ===
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token

def find_text_column(ds):
    """Finds the main text column in a dataset."""
    for key in ("text", "content", "paragraph", "data"):
        if key in ds.column_names:
            return key
    # Fallback heuristic: first string-like column
    for k in ds.column_names:
        sample = ds[k][0]
        if isinstance(sample, str) and len(sample.strip()) > 0:
            return k
    raise ValueError(f"No suitable text column found in {ds}")

text_cols = [find_text_column(ds) for ds in datasets]

def tokenize(batch, text_col):
    """Tokenizes text safely, skipping None/empty entries and adds labels for Trainer."""
    texts = []
    for t in batch[text_col]:
        if isinstance(t, str):
            stripped = t.strip()
            if stripped:
                texts.append(stripped)

    # If this batch has no valid text, return empty fields
    if not texts:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    tokens = tok(
        texts,
        truncation=True,
        padding="max_length",
        max_length=cfg["context_length"],
    )

    # Hugging Face Trainer expects labels for loss computation
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


# === Apply tokenization and cleanup ===
tokenized_datasets = []
for ds, col in zip(datasets, text_cols):
    tokenized = ds.map(
        lambda batch: tokenize(batch, col),
        batched=True,
        num_proc=20,                     # optional parallelism
        load_from_cache_file=True,      # reuse cache if it exists
        keep_in_memory=False,           # force write to disk
        remove_columns=ds.column_names,
    )
    # filter out any leftover empty or None samples
    tokenized = tokenized.filter(
        lambda x: x["input_ids"] is not None and len(x["input_ids"]) > 0
    )
    tokenized_datasets.append(tokenized)

# Interleave or concatenate as before
mixture = interleave_datasets(tokenized_datasets, probabilities=[m["prob"] for m in cfg["mix"]])



# Model config
model_cfg = GPT2Config(
    vocab_size=len(tok),
    n_positions=cfg["context_length"],
    n_embd=768, n_layer=12, n_head=12  # GPT-2 small (124M)
)
model = AutoModelForCausalLM.from_config(model_cfg)

# Training setup
args = TrainingArguments(
    output_dir="checkpoints/sparknet-70m",
    bf16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    num_train_epochs=1,
    logging_dir="logs/tensorboard",
    logging_steps=50,
    eval_strategy="no",
    eval_steps=500,
    save_steps=1000,
    report_to=["tensorboard"],
    save_total_limit=2,
    load_best_model_at_end=False,
    remove_unused_columns=False,
)

trainer = Trainer(model=model, args=args, train_dataset=mixture)
trainer.train()
