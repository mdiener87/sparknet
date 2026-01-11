from transformers import AutoTokenizer, LlamaTokenizer
from datasets import load_dataset
import statistics
from tqdm import tqdm

# ---------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------
TOKENIZER_GPT2 = "gpt2"
TOKENIZER_V5 = "./tokenizer-v5"
TOKENIZER_V6 = "./tokenizer-v6"

SAMPLES = 10_000
DATASET = "codelion/fineweb-edu-100M"

print("Loading tokenizers...")

tok_gpt2 = AutoTokenizer.from_pretrained(TOKENIZER_GPT2)
tok_v5 = AutoTokenizer.from_pretrained(TOKENIZER_V5)

tok_v6 = LlamaTokenizer.from_pretrained(
    TOKENIZER_V6,
    model_max_length=1024,
)
tok_v6.pad_token = tok_v6.eos_token

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
ds = load_dataset(DATASET, split="train", streaming=True)

lengths = {
    "gpt2": [],
    "v5": [],
    "v6": [],
}

examples = []

for row in tqdm(ds, total=SAMPLES):
    if "text" not in row or not isinstance(row["text"], str):
        continue

    text = row["text"]

    ids_gpt2 = tok_gpt2(text, add_special_tokens=False)["input_ids"]
    ids_v5 = tok_v5(text, add_special_tokens=False)["input_ids"]
    ids_v6 = tok_v6(text, add_special_tokens=False)["input_ids"]

    lengths["gpt2"].append(len(ids_gpt2))
    lengths["v5"].append(len(ids_v5))
    lengths["v6"].append(len(ids_v6))

    if len(examples) < 5:
        examples.append(text[:500])

    if len(lengths["gpt2"]) >= SAMPLES:
        break


def summarize(name, vals):
    print(f"\n{name}:")
    print(f"  mean tokens:   {statistics.mean(vals):.2f}")
    print(f"  median:        {statistics.median(vals)}")
    print(f"  p90:           {statistics.quantiles(vals, n=10)[8]}")


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------
summarize("GPT-2 (stock)", lengths["gpt2"])
summarize("SparkNet v5 (custom GPT-2 BPE)", lengths["v5"])
summarize("SparkNet v6 (SentencePiece Unigram)", lengths["v6"])

print("\nCompression ratios:")
print(f"  v5 / GPT-2: {statistics.mean(v5 / gpt2 for v5, gpt2 in zip(lengths['v5'], lengths['gpt2']) if gpt2 > 0):.3f}")
print(f"  v6 / GPT-2: {statistics.mean(v6 / gpt2 for v6, gpt2 in zip(lengths['v6'], lengths['gpt2']) if gpt2 > 0):.3f}")
print(f"  v6 / v5:    {statistics.mean(v6 / v5 for v6, v5 in zip(lengths['v6'], lengths['v5']) if v5 > 0):.3f}")
