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
    {"label": "SparkNet 70M v2", "path": "checkpoints/sparknet-70m-v4"},
    {"label": "GPT-2", "path": "gpt2"},
    {"label": "CodeLion GPT-2 70M", "path": "codelion/gpt-2-70m"},
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

VALIDATION_SAMPLES = 2000  # number of samples per dataset
MAX_SEQUENCE_LENGTH = 256  # truncate longer texts for efficiency
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("eval")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
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
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)

    text_field = cfg.get("text_field", "text")
    collected = []
    for record in dataset:
        text = record.get(text_field)
        if not text:
            continue
        stripped = text.strip()
        if not stripped:
            continue
        collected.append(stripped)
        if sample_limit and len(collected) >= sample_limit:
            break
    return collected


def evaluate_model_on_texts(model, tokenizer, texts):
    losses = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs, labels=inputs["input_ids"])
        losses.append(output.loss.item())

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
        print(
            f"Loaded dataset '{cfg['label']}' with {len(samples)} usable samples (limit {VALIDATION_SAMPLES})."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"validation_{timestamp}.txt"

    header_lines = [
        f"Validation run: {datetime.now().isoformat()}",
        f"Device: {DEVICE}",
        f"Max sequence length: {MAX_SEQUENCE_LENGTH}",
        "Models:" + "".join(
            f"\n  - {cfg['label']}: {cfg['path']}" for cfg in MODEL_CONFIGS
        ),
        "Datasets:" + "".join(
            f"\n  - {cfg['label']} ({cfg['dataset_name']} / {cfg.get('dataset_config','default')} split={cfg.get('split','validation')})"
            for cfg in DATASET_CONFIGS
        ),
        f"Samples per dataset: {VALIDATION_SAMPLES}",
        "",
    ]

    results_lines = header_lines.copy()

    for model_cfg in MODEL_CONFIGS:
        print(f"\n=== Evaluating {model_cfg['label']} ({model_cfg['path']}) ===")
        tokenizer, model = load_model(model_cfg["path"])

        for dataset_label, samples in dataset_samples.items():
            if not samples:
                msg = f"No samples available for dataset '{dataset_label}'. Skipping."
                print(msg)
                results_lines.append(msg)
                continue

            avg_loss, perplexity = evaluate_model_on_texts(model, tokenizer, samples)
            line = (
                f"Model: {model_cfg['label']} | Dataset: {dataset_label} | "
                f"Samples: {len(samples)} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}"
            )
            print(line)
            results_lines.append(line)

    with output_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(results_lines))

    print(f"\nDetailed validation results saved to {output_file}")


if __name__ == "__main__":
    main()
