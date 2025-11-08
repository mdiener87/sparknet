# train_sparknet.py
import os
import json
import math
import random
from typing import Dict, Any, Iterable

import torch
from datasets import load_dataset, interleave_datasets, IterableDataset, Dataset, Features, Value
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

# Run Configuration
RUN_NAME = "sparknet-70m-v1-1bil"

# --------------------------------------------------------------------------------------
# Load Config
# --------------------------------------------------------------------------------------
with open("configs/datasets_v1.json") as f:
    cfg = json.load(f)

seed = int(cfg.get("seed", 42))
random.seed(seed)
torch.manual_seed(seed)

block_size = int(cfg.get("context_length", 1024))
target_tokens = int(cfg.get("target_tokens", 1_000_000_000))  # training budget

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
    name = m["name"]
    split = m.get("split", "train")

    # --- Proper load_dataset call ---
    if name == "json":
        # Local JSONL (blog posts)
        ds = load_dataset(
            path="json",
            data_files=m["data_files"],
            split=split,
            streaming=True
        )
    else:
        ds = load_dataset(
            path=name,
            split=split,
            streaming=True
        )

    # --- Normalize schema if static features are present ---
    if getattr(ds, "features", None) is not None:
        features = ds.features
        updated_cols = []
        for key, val in features.items():
            if hasattr(val, "dtype") and val.dtype == "float64":
                ds = ds.cast_column(key, Value("float32"))
                updated_cols.append(key)
        if updated_cols:
            # rebuild proper Features object (ensures interleave_datasets compatibility)
            new_features = Features({
                k: (Value("float32") if k in updated_cols else v)
                for k, v in features.items()
            })
            ds._features = new_features
            print(f"↻ Casting {name} columns {updated_cols} → Value('float32') for schema alignment")
    else:
        print(f"ℹ️  {name} has no static features (streaming parquet) — skipping dtype cast")

    # ✅ Append after processing each dataset
    sources.append(ds)
    probs.append(float(m["prob"]))
    print(f"✓ streaming source: {name} [{split}] (prob={m['prob']})")

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
    n_embd=512,
    n_layer=12,
    n_head=8,
)
model = AutoModelForCausalLM.from_config(model_cfg)

# Memory/perf niceties
model.gradient_checkpointing_disable()
# if hasattr(torch, "compile"):
#     try:
#         model = torch.compile(model)
#     except Exception as _:
#         pass

# --------------------------------------------------------------------------------------
# TrainingArguments
# - Compute max_steps from token budget so mixture probabilities map to token shares
# --------------------------------------------------------------------------------------
per_device_train_batch_size = 16
gradient_accumulation_steps = 2

tokens_per_step = block_size * per_device_train_batch_size * gradient_accumulation_steps
max_steps = math.ceil(target_tokens / tokens_per_step)

print(f"[Budget] target_tokens={target_tokens:,} | tokens/step={tokens_per_step:,} | max_steps={max_steps:,}")

collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

args = TrainingArguments(
    output_dir=f"checkpoints/{RUN_NAME}",
    bf16=True, 
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=1.5e-4,          
    weight_decay=0.01,
    warmup_ratio=0.02,
    lr_scheduler_type="cosine",
    max_steps=max_steps,
    logging_dir=f"logs/{RUN_NAME}",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=5000,              
    save_steps=10000,
    save_strategy="steps",
    optim="adamw_torch_fused",
    save_total_limit=3,
    report_to=["tensorboard"],
    remove_unused_columns=False,
    dataloader_num_workers=8,
)

# --------------------------------------------------------------------------------------
# Training Callbacks
# --------------------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
)



from transformers import TrainerCallback
import time, os
from torch.utils.tensorboard import SummaryWriter

class ThroughputCallback(TrainerCallback):
    """
    Logs throughput metrics (tokens/sec, examples/sec, elapsed minutes)
    to both console and TensorBoard without interfering with Trainer control flow.
    """

    def __init__(self, block_size, log_dir=None):
        self.block_size = block_size
        self.start_time = None
        self.last_time = None
        self.last_step = 0
        self.total_tokens = 0
        self.tb_writer = None
        self.log_dir = log_dir or os.path.join("logs", "tensorboard")

    def on_train_begin(self, args, state, control, **kwargs):
        # initialize timers and tensorboard writer
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_step = 0
        self.total_tokens = 0
        if state.is_local_process_zero:
            os.makedirs(self.log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=self.log_dir)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        elapsed = now - self.last_time
        step_diff = state.global_step - self.last_step
        self.last_time = now
        self.last_step = state.global_step

        # compute throughput
        step_tokens = (
            self.block_size
            * args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * step_diff
        )
        self.total_tokens += step_tokens

        toks_per_sec = step_tokens / elapsed if elapsed > 0 else 0.0
        ex_per_sec = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * step_diff
        ) / elapsed if elapsed > 0 else 0.0
        elapsed_min = (now - self.start_time) / 60.0

        # console output every logging interval
        if state.is_local_process_zero and state.global_step % args.logging_steps == 0:
            print(
                f"[Step {state.global_step:>6}] "
                f"{toks_per_sec:>9.0f} tok/s | {ex_per_sec:>6.1f} ex/s | "
                f"elapsed {elapsed_min:5.1f} min"
            )

        # tensorboard logging
        if self.tb_writer and state.is_local_process_zero:
            self.tb_writer.add_scalar("throughput/tokens_per_sec", toks_per_sec, state.global_step)
            self.tb_writer.add_scalar("throughput/examples_per_sec", ex_per_sec, state.global_step)
            self.tb_writer.add_scalar("throughput/elapsed_min", elapsed_min, state.global_step)

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()
        return control


trainer.add_callback(ThroughputCallback(cfg["context_length"], log_dir=f"logs/{RUN_NAME}"))


class GradNormCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # logs already contains 'loss', etc.
        grad_norm = logs.get("grad_norm")
        if grad_norm is not None:
            logs["grad/grad_norm"] = grad_norm
        return control

trainer.add_callback(GradNormCallback())  # optional


class SampleGenCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        prompt = "In the quiet heart of Mechanus,"
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**input_ids, max_length=100, do_sample=True, temperature=0.8)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        log_file = os.path.join(args.output_dir, f"sample_step{state.global_step}.txt")
        with open(log_file, "w") as f:
            f.write(text)
        print(f"[SampleGen] wrote sample_step{state.global_step}.txt")
        return control


trainer.add_callback(SampleGenCallback(tok))  # optional

# --------------------------------------------------------------------------------------
# Training Main Init
# --------------------------------------------------------------------------------------

# Validate output dirs exist
os.makedirs(f"checkpoints/{RUN_NAME}", exist_ok=True)
os.makedirs(f"logs/{RUN_NAME}", exist_ok=True)


from datetime import datetime

if __name__ == "__main__":
    trainer.train()
    # Save final artifacts
    tok.save_pretrained(f"checkpoints/{RUN_NAME}")
    model.save_pretrained(f"checkpoints/{RUN_NAME}")

    # Save model metadata
    metadata = {
        "run_name": RUN_NAME,
        "timestamp": datetime.now().isoformat(),
        "params": {
            "n_embd": model_cfg.n_embd, "n_layer": model_cfg.n_layer, "n_head": model_cfg.n_head,
            "context_length": block_size, "token_budget": target_tokens
        },
        "datasets": [m["name"] for m in cfg["mix"]],
        "notes": "First full v1 run; matches Codelion 70M recipe with small blog inclusion."
    }
    with open(f"checkpoints/{RUN_NAME}/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


