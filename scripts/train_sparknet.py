# train_sparknet.py
import os
import json
import math
import random
from typing import Dict, Any, Iterable

import torch
from datasets import load_dataset, interleave_datasets, IterableDataset, Dataset
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
os.environ["HF_DATASETS_OFFLINE"] = "0"  # allow hub access
torch.backends.cuda.matmul.allow_tf32 = True  # safe perf boost
torch.backends.cudnn.allow_tf32 = True

# --------------------------------------------------------------------------------------
# Load Config
# --------------------------------------------------------------------------------------
with open("configs/datasets_v1.json") as f:
    cfg = json.load(f)

seed = int(cfg.get("seed", 42))
random.seed(seed)
torch.manual_seed(seed)

block_size = int(cfg.get("context_length", 1024))
target_tokens = int(cfg.get("target_tokens", 200_000_000))  # training budget

# --------------------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
eos_id = tok.eos_token_id

# --------------------------------------------------------------------------------------
# Build streaming sources and interleave by probability
# --------------------------------------------------------------------------------------
sources = []
probs = []

for m in cfg["mix"]:
    # Accept either hub datasets or local JSONL via datasets "json" builder
    name = m["name"]
    split = m.get("split", "train")

    kwargs: Dict[str, Any] = {"split": split, "streaming": True}
    if name == "json":
        # Expected: {"name":"json", "data_files":{"train":"data/diener_blog.jsonl"}, "prob": X}
        kwargs.update(m)
    else:
        kwargs["path"] = name

    try:
        ds = load_dataset(**kwargs)
        sources.append(ds)
        probs.append(float(m["prob"]))
        print(f"✓ streaming source: {name} [{split}] (prob={m['prob']})")
    except Exception as e:
        print(f"⚠️ skipping {name}: {e}")

if not sources:
    raise RuntimeError("No datasets could be loaded in streaming mode. Check your config/datasets and network.")

# normalize probabilities
psum = sum(probs)
probs = [p / psum for p in probs]

# Interleaved mixed stream
mixture_stream = interleave_datasets(
    sources,
    probabilities=probs,
    seed=seed,
    stopping_strategy="first_exhausted",  # keep going while all have data available
)

# --------------------------------------------------------------------------------------
# Tokenize lazily then pack to fixed-size blocks
# --------------------------------------------------------------------------------------
def extract_text(example: Dict[str, Any]) -> str:
    # Try common keys, then any single string field
    for k in ("text", "content", "paragraph", "data"):
        if k in example and isinstance(example[k], str) and example[k].strip():
            return example[k].strip()
    # Fallback: scan for a string value
    for v in example.values():
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def tokenize_map(example: Dict[str, Any]) -> Dict[str, Any]:
    txt = extract_text(example)
    if not txt:
        return {"input_ids": []}
    ids = tok.encode(txt, add_special_tokens=False)
    # ensure EOS boundary
    if len(ids) == 0 or ids[-1] != eos_id:
        ids = ids + [eos_id]
    return {"input_ids": ids}

tokenized_stream = mixture_stream.map(tokenize_map)

def packing_generator(stream: Iterable[Dict[str, Any]], block: int):
    """Concatenate incoming token ids and emit fixed-size blocks."""
    buffer = []
    total_emitted = 0
    for ex in stream:
        ids = ex.get("input_ids", [])
        if not ids:
            continue
        buffer.extend(ids)
        while len(buffer) >= block:
            chunk = buffer[:block]
            del buffer[:block]
            total_emitted += block
            yield {
                "input_ids": chunk,
                "attention_mask": [1] * block,
                "labels": chunk.copy()
            }

# Wrap the generator so Trainer can consume it as a regular Dataset
def gen_train():
    for item in packing_generator(tokenized_stream, block_size):
        yield item

train_ds = Dataset.from_generator(gen_train)

# --------------------------------------------------------------------------------------
# Eval dataset (for loss curve)
# Using wikitext-2 raw validation; packed to block_size
# --------------------------------------------------------------------------------------
def build_eval_dataset(block: int) -> Dataset:
    eval_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    def tok_line(ex):
        ids = tok(ex["text"], add_special_tokens=False)["input_ids"]
        # append EOS to each line for safe boundaries
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
# Model (~70M parameters, GPT-2-style)
# --------------------------------------------------------------------------------------
model_cfg = GPT2Config(
    vocab_size=len(tok),
    n_positions=block_size,
    n_embd=512,   # 70M-ish class
    n_layer=8,
    n_head=8,
)
model = AutoModelForCausalLM.from_config(model_cfg)

# Memory/perf niceties
model.gradient_checkpointing_enable()
if hasattr(torch, "compile"):
    try:
        model = torch.compile(model)
    except Exception as _:
        pass

# --------------------------------------------------------------------------------------
# TrainingArguments
# - Compute max_steps from token budget so mixture probabilities map to token shares
# --------------------------------------------------------------------------------------
per_device_train_batch_size = 4
gradient_accumulation_steps = 1

tokens_per_step = block_size * per_device_train_batch_size * gradient_accumulation_steps
max_steps = math.ceil(target_tokens / tokens_per_step)

print(f"[Budget] target_tokens={target_tokens:,} | tokens/step={tokens_per_step:,} | max_steps={max_steps:,}")

collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

args = TrainingArguments(
    output_dir="checkpoints/sparknet-70m-v1",
    bf16=True,  # Grace-Blackwell: yes
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=2e-4,          # slightly gentler for 70M
    weight_decay=0.01,
    warmup_ratio=0.01,
    lr_scheduler_type="cosine",
    max_steps=max_steps,
    logging_dir="logs/tensorboard",   # fresh run dir
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=500,              # chart eval loss every 500 steps
    save_steps=1000,
    save_strategy="steps",
    save_total_limit=3,
    report_to=["tensorboard"],
    remove_unused_columns=False,
    dataloader_num_workers=2,
)

# --------------------------------------------------------------------------------------
# Training Callbacks
# --------------------------------------------------------------------------------------

from transformers import TrainerCallback
import time

class ThroughputCallback(TrainerCallback):
    """Logs examples/sec, tokens/sec, and step time to TensorBoard."""
    def __init__(self):
        self.start_time = None
        self.total_tokens = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.total_tokens = 0

    def on_step_end(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        # approximate tokens processed in this step
        step_tokens = block_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self.total_tokens += step_tokens
        elapsed = time.time() - self.start_time
        toks_per_sec = self.total_tokens / elapsed if elapsed > 0 else 0.0
        logs["throughput/tokens_per_sec"] = toks_per_sec
        logs["throughput/elapsed_min"] = elapsed / 60.0
        return control

trainer.add_callback(ThroughputCallback())


class GradNormCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # logs already contains 'loss', etc.
        grad_norm = logs.get("grad_norm")
        if grad_norm is not None:
            logs["grad/grad_norm"] = grad_norm
        return control

trainer.add_callback(GradNormCallback())  # optional


class SampleGenCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, model=None, tokenizer=None, **kwargs):
        prompt = "In the quiet heart of Mechanus,"
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**input_ids, max_length=100, do_sample=True, temperature=0.8)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        log_file = os.path.join(args.output_dir, f"sample_step{state.global_step}.txt")
        with open(log_file, "w") as f:
            f.write(text)
        print(f"[SampleGen] Wrote sample_step{state.global_step}.txt")
        return control

trainer.add_callback(SampleGenCallback())  # optional

import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
def gpu_stats():
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"[GPU] {info.used/1e9:.1f} GB / {info.total/1e9:.1f} GB  |  {util.gpu}% util")

trainer.add_callback(gpu_stats())  # optional


# --------------------------------------------------------------------------------------
# Training Main Init
# --------------------------------------------------------------------------------------

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
)


from datetime import datetime

if __name__ == "__main__":
    trainer.train(resume_from_checkpoint=True)
    # Save final artifacts
    tok.save_pretrained("checkpoints/sparknet-70m-v1")
    model.save_pretrained("checkpoints/sparknet-70m-v1")

    # Save model metadata
    metadata = {
        "run_name": "sparknet-70m-v1",
        "timestamp": datetime.now().isoformat(),
        "params": {
            "n_embd": 512, "n_layer": 8, "n_head": 8,
            "context_length": block_size, "token_budget": target_tokens
        },
        "datasets": [m["name"] for m in cfg["mix"]],
        "notes": "First full v1 run; matches Codelion 70M recipe with small blog inclusion."
    }
    with open("checkpoints/sparknet-70m-v1/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


