from datasets import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./tokenizer-v6")
ds = Dataset.load_from_disk("datasets/sft_chat_v1/shard-000")

def show(idx):
    row = ds[idx]
    text = tokenizer.decode(row["input_ids"], skip_special_tokens=False)
    labels = row["labels"]

    print("=" * 80)
    print(text)
    print("\n[labels active at start?]", any(l != -100 for l in labels[:50]))

for i in range(5):
    show(i)
