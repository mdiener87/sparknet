import os
import json
import random
import re
from datasets import load_dataset
import sentencepiece as spm
from tqdm import tqdm

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
OUTPUT_DIR = "tokenizer-v6"
SPM_MODEL_PREFIX = "sparknet_v6"
VOCAB_SIZE = 32000
CHAR_COVERAGE = 0.9995

# How much raw text to feed SentencePiece (enough, but not insane)
MAX_TEXT_SAMPLES = 1_000_000   # ~several GB depending on avg length

# Dataset mix (roughly mirrors pretraining distribution)
DATA_SOURCES = [
    ("codelion/fineweb-edu-1B", 0.52),
    ("codelion/dclm-baseline-1B", 0.22),
    ("codelion/finepdfs-1B", 0.12),
    ("eli5", 0.13),
    ("data/diener_blog.jsonl", 0.01),
]

TMP_TEXT_FILE = "spm_training_text.txt"
SEED = 42

random.seed(SEED)

# ---------------------------------------------------------------------
# Light normalization (VERY important)
# ---------------------------------------------------------------------
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Normalize newlines
    text = text.replace("\r\n", "\n")

    # Collapse long runs of spaces/tabs (keep up to 2)
    text = re.sub(r"[ \t]{3,}", "  ", text)

    # Collapse pathological newlines
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------
# Stream text from datasets
# ---------------------------------------------------------------------
def stream_text():
    while True:
        name = random.choices(
            [s for s, _ in DATA_SOURCES],
            weights=[w for _, w in DATA_SOURCES],
        )[0]

        if name.endswith(".jsonl"):
            with open(name, "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        for v in obj.values():
                            if isinstance(v, str):
                                yield v
                    except Exception:
                        continue
        else:
            ds = load_dataset(name, split="train", streaming=True)
            for row in ds:
                if "text" in row and isinstance(row["text"], str):
                    yield row["text"]
                else:
                    for v in row.values():
                        if isinstance(v, str):
                            yield v


# ---------------------------------------------------------------------
# Collect text
# ---------------------------------------------------------------------
def build_training_corpus():
    print("Building tokenizer training corpus...")
    count = 0

    with open(TMP_TEXT_FILE, "w", encoding="utf-8") as f:
        for text in tqdm(stream_text(), total=MAX_TEXT_SAMPLES):
            text = normalize_text(text)
            if not text:
                continue

            f.write(text)
            f.write("\n")

            count += 1
            if count >= MAX_TEXT_SAMPLES:
                break

    print(f"Wrote {count:,} samples to {TMP_TEXT_FILE}")


# ---------------------------------------------------------------------
# Train SentencePiece
# ---------------------------------------------------------------------
def train_sentencepiece():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Training SentencePiece tokenizer...")
    spm.SentencePieceTrainer.train(
        input=TMP_TEXT_FILE,
        model_prefix=os.path.join(OUTPUT_DIR, SPM_MODEL_PREFIX),
        vocab_size=VOCAB_SIZE,
        model_type="unigram",
        character_coverage=CHAR_COVERAGE,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
        byte_fallback=True,
        shuffle_input_sentence=True,
        input_sentence_size=1_000_000,
        seed_sentencepiece_size=1000000,
        num_threads=os.cpu_count(),
    )

    print("Tokenizer training complete.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    build_training_corpus()
    train_sentencepiece()

    print("\nTokenizer files:")
    print(f"  {OUTPUT_DIR}/{SPM_MODEL_PREFIX}.model")
    print(f"  {OUTPUT_DIR}/{SPM_MODEL_PREFIX}.vocab")
