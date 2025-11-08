import os, math, random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments
)

# Optional but recommended for fast/cheap SFT
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ----------------------------
# Config
# ----------------------------
BASE_MODEL_DIR = "checkpoints/sparknet-70m-v1-final"   # your pretrained 70M
RUN_NAME       = "sparknet-70m-instruct-v1"
OUT_DIR        = f"checkpoints/{RUN_NAME}"
LOG_DIR        = f"logs/{RUN_NAME}"

DATASET_ID     = "yahma/alpaca-cleaned"
VAL_FRACTION   = 0.02           # small validation slice
CTX_LEN        = 1024
SEED           = 42

# Hyperparams for LoRA SFT (safe defaults for 70M)
LR             = 5e-5
EPOCHS         = 2
BATCH_PER_DEV  = 8
GRAD_ACCUM     = 2
WARMUP_RATIO   = 0.03
WEIGHT_DECAY   = 0.0

USE_LORA       = True
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
LORA_TARGETS   = ["c_attn", "c_proj", "c_fc"]  # GPT-2 module names

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------
# Load tokenizer/model
# ----------------------------
tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
tok.pad_token = tok.eos_token
eos_id = tok.eos_token_id

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR)
model.config.n_positions = CTX_LEN  # keep consistent

if USE_LORA:
    # (Optional) if using 4/8-bit, call prepare_model_for_kbit_training(model)
    peft_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_cfg)

# ----------------------------
# Prompt formatting + label masking
# ----------------------------
INSTR_TPL = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

NO_INPUT_TPL = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)

def format_example(ex: Dict) -> Dict:
    instruction = ex.get("instruction", "").strip()
    input_      = ex.get("input", "").strip()
    output      = ex.get("output", "").strip()

    if input_:
        text = INSTR_TPL.format(instruction=instruction, input=input_, output=output)
        prefix = INSTR_TPL.format(instruction=instruction, input=input_, output="")
    else:
        text = NO_INPUT_TPL.format(instruction=instruction, output=output)
        prefix = NO_INPUT_TPL.format(instruction=instruction, output="")

    # tokenize
    ids = tok(text, truncation=True, max_length=CTX_LEN, add_special_tokens=False)["input_ids"]
    pref_ids = tok(prefix, truncation=True, max_length=CTX_LEN, add_special_tokens=False)["input_ids"]

    # labels: ignore prompt tokens, train only on response tokens
    labels = [-100]*len(pref_ids) + ids[len(pref_ids):]
    labels = labels[:CTX_LEN]
    ids    = ids[:CTX_LEN]

    # ensure we end with EOS if room
    if len(ids) < CTX_LEN and (len(ids) == 0 or ids[-1] != eos_id):
        ids.append(eos_id)
        labels.append(eos_id if len(labels) == len(ids) else -100)

    attn = [1]*len(ids)
    return {"input_ids": ids, "labels": labels, "attention_mask": attn}

# ----------------------------
# Load data
# ----------------------------
raw = load_dataset(DATASET_ID)
if "train" in raw and "test" in raw:
    train_ds = raw["train"]
    val_ds   = raw["test"]
else:
    # split from train
    split = raw["train"].train_test_split(test_size=VAL_FRACTION, seed=SEED)
    train_ds, val_ds = split["train"], split["test"]

train_tok = train_ds.map(format_example, remove_columns=train_ds.column_names, num_proc=4)
val_tok   = val_ds.map(format_example,   remove_columns=val_ds.column_names,   num_proc=4)

# filter out empty/too-short
def keep_ok(ex): return len(ex["input_ids"]) > 8 and len(ex["labels"]) > 8
train_tok = train_tok.filter(keep_ok)
val_tok   = val_tok.filter(keep_ok)

# set pt format
cols = ["input_ids","labels","attention_mask"]
train_tok.set_format("torch", columns=cols)
val_tok.set_format("torch", columns=cols)

# ----------------------------
# Data collator (pad to max in batch)
# ----------------------------
@dataclass
class Collator:
    pad_id: int
    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(
                torch.cat([
                    b["input_ids"],
                    torch.full((pad,), self.pad_id, dtype=torch.long)
                ])
            )
            labels.append(
                torch.cat([
                    b["labels"],
                    torch.full((pad,), -100, dtype=torch.long)
                ])
            )
            attn.append(
                torch.cat([
                    b["attention_mask"],
                    torch.zeros(pad, dtype=torch.long)
                ])
            )
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn),
        }


collator = Collator(pad_id=tok.pad_token_id)

# ----------------------------
# Training args
# ----------------------------
total_train_tokens = len(train_tok) * CTX_LEN
print(f"[SFT] samples={len(train_tok):,}  val={len(val_tok):,}  approx_tokens={total_train_tokens:,}")

args = TrainingArguments(
    output_dir=OUT_DIR,
    logging_dir=LOG_DIR,
    report_to=["tensorboard"],
    run_name=RUN_NAME,

    per_device_train_batch_size=BATCH_PER_DEV,
    gradient_accumulation_steps=GRAD_ACCUM,
    per_device_eval_batch_size=BATCH_PER_DEV,
    bf16=True,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    num_train_epochs=EPOCHS,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    dataloader_num_workers=2,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    train_dataset=train_tok,
    eval_dataset=val_tok,
)

trainer.train()

# Save LoRA (or full) artifacts separately
trainer.save_model(OUT_DIR)
tok.save_pretrained(OUT_DIR)

print("\n[SFT] Done. To run generation, load from:", OUT_DIR)
