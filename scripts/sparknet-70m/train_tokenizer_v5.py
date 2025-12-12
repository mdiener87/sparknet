import random
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers
from transformers import PreTrainedTokenizerFast
import os

# Determinism
random.seed(42)

# ---- CONFIG ----
MAX_LINES = 2_000_000      # number of training text samples
VOCAB_SIZE = 50257         # GPT-2 vocab size
SAVE_DIR = "tokenizer-v5"

# ---- DATA SOURCES ----
sources = [
    "codelion/finepdfs-1B",
    "codelion/dclm-baseline-1B",
    "codelion/fineweb-edu-1B",
]

# ---- BUILD BYTE-LEVEL BPE BASE ----
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    show_progress=True,
    special_tokens=[
        "",          # EOS
        "<|pad|>",    # PAD
    ]
)

# ---- RESERVOIR SAMPLING ----
def sample_text():
    """Uniform sampling across all datasets with streaming mode."""
    while True:
        src = random.choice(sources)
        ds = load_dataset(src, split="train", streaming=True)

        for row in ds:
            # Find the "text-like" field
            if isinstance(row, dict):
                if "text" in row:
                    yield row["text"]
                else:
                    # fallback scanning for a string
                    for v in row.values():
                        if isinstance(v, str):
                            yield v

# ---- COLLECT TRAINING LINES ----
print(f"Collecting {MAX_LINES:,} lines...")
reservoir = []

for text in sample_text():
    reservoir.append(text)
    if len(reservoir) >= MAX_LINES:
        break

print(f"Collected {len(reservoir):,} lines.")

# ---- TRAIN TOKENIZER ----
print("Training tokenizer…")
tokenizer.train_from_iterator(reservoir, trainer)

# ---- POST-PROCESSING (GPT-2 behavior) ----
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
tokenizer.decoder = decoders.ByteLevel()

# ---- SAVE RAW TOKENIZER ----
os.makedirs(SAVE_DIR, exist_ok=True)
tokenizer.save(os.path.join(SAVE_DIR, "tokenizer-v5.json"))

# ---- CONVERT TO HF FORMAT ----
hf_tok = PreTrainedTokenizerFast(
    tokenizer_file=os.path.join(SAVE_DIR, "tokenizer-v5.json"),
    eos_token="",
    pad_token="<|pad|>",
)

hf_tok.save_pretrained(SAVE_DIR)

print("Tokenizer training complete!")
print(f"Saved in: {SAVE_DIR}/")
