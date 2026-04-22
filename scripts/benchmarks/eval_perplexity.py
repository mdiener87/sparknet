import math
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
MODEL_CONFIGS = [
    # {"label": "SparkNet 70M v5", "path": "checkpoints/sparknet-70m-v5"},
    # {"label": "GPT-2", "path": "gpt2"},
    # {"label": "CodeLion GPT-2 70M", "path": "codelion/gpt-2-70m"},
    {"label": "DienerTech Sparknet-400m v2 22889", "path": "checkpoints/sparknet-400m-v2-12b/checkpoint-22889"},
]

DATASET_CONFIGS = [
    {
        "label": "WikiText-2 Raw v1 (validation)",
        "dataset_name": "wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "split": "validation",
        "text_field": "text",
    },
]

VALIDATION_SAMPLES = 2000
MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 8
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("eval")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    # Ensure pad token exists (GPT2 normally doesn't have one)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  

    model = AutoModelForCausalLM.from_pretrained(path).to(DEVICE)
    model.eval()
    return tok, model


def prepare_dataset_samples(cfg, sample_limit, seed):
    load_args = [cfg["dataset_name"]]
    if cfg.get("dataset_config"):
        load_args.append(cfg["dataset_config"])

    dataset = load_dataset(*load_args, split=cfg.get("split", "validation"))
    dataset = dataset.shuffle(seed=seed)

    text_field = cfg.get("text_field", "text")
    collected = []

    for record in dataset:
        text = record.get(text_field, "")
        text = text.strip()
        if text:
            collected.append(text)
        if len(collected) >= sample_limit:
            break

    return collected


def evaluate_model_on_texts(model, tokenizer, texts):
    losses = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
        )

        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        # Mask out padding tokens for labels
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        batch_loss = out.loss.detach().cpu().item()
        losses.append(batch_loss)

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


# ---------------------------------------------------------
# Main run
# ---------------------------------------------------------
def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_samples = {}
    for cfg in DATASET_CONFIGS:
        samples = prepare_dataset_samples(cfg, VALIDATION_SAMPLES, SEED)
        dataset_samples[cfg["label"]] = samples
        print(f"Loaded dataset '{cfg['label']}' with {len(samples)} usable samples.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"validation_{timestamp}.txt"

    results_lines = [f"Validation run: {datetime.now().isoformat()}"]

    for model_cfg in MODEL_CONFIGS:
        print(f"\n=== Evaluating {model_cfg['label']} ===")
        tokenizer, model = load_model(model_cfg["path"])

        for dataset_label, samples in dataset_samples.items():
            avg_loss, perplexity = evaluate_model_on_texts(model, tokenizer, samples)
            line = (
                f"Model: {model_cfg['label']} | Dataset: {dataset_label} | "
                f"Samples: {len(samples)} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}"
            )
            print(line)
            results_lines.append(line)

    with open(output_file, "w") as f:
        f.write("\n".join(results_lines))

    print(f"\nDetailed results saved to {output_file}")


if __name__ == "__main__":
    main()
