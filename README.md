# SparkNet

<img src="sparknet.png" width="400"/>


SparkNet is a from-scratch, GPT-2–style language model project that focuses on training compact (≈70M parameter) causal decoders with modern Hugging Face tooling. The repo contains everything needed to:

- stream and mix public Hugging Face datasets or consume a pre-packed 1B-token corpus
- train custom byte-level tokenizers
- launch full runs with instrumentation (TensorBoard throughput, gradient norm, periodic sample generation)
- export checkpoints that can be reused with standard 🤗 `AutoModelForCausalLM` APIs

---

## Requirements

- Linux or WSL2 with Python 3.10+
- NVIDIA GPU with at least 24 GB VRAM (32×1024 tokens with GradAccum=2 fits comfortably)
- CUDA 12.9 runtime and driver (matching the pinned `torch==2.9.0+cu129`)
- Hugging Face Hub credentials (datasets such as `codelion/finepdfs-1B` require auth)
- ~200 GB of free disk for cache + packed dataset + checkpoints

> The training scripts enable TF32 kernels and Flash Attention automatically when available.

---

## Installation

```bash
git clone https://github.com/<you>/sparknet.git
cd sparknet

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Recommended environment variables (already set inside the scripts, but exporting them keeps CLI tools consistent):

```bash
export HF_DATASETS_CACHE=~/projects/sparknet/cache
export TOKENIZERS_PARALLELISM=false
```

Login to the Hugging Face hub once so the streaming scripts can read gated datasets:

```bash
huggingface-cli login
```

---

## Dataset & Config Basics

Training is driven by JSON configs under `configs/`. Each config defines:

```json
{
  "seed": 42,
  "context_length": 1024,
  "target_tokens": 1000000000,
  "mix": [
    {"name": "codelion/finepdfs-1B", "split": "train", "prob": 0.5},
    {"name": "json", "data_files": {"train": "data/diener_blog.jsonl"}, "prob": 0.002}
  ]
}
```

- `mix`: list of streaming datasets with sampling probabilities. Use `"json"` for local JSONL files.
- `context_length`: block size fed into the model.
- `target_tokens`: total training budget (used to derive max steps).

Update or duplicate a config (e.g., `configs/datasets_v5.json`) and point the training script to it.

---

## Training Workflow

You can either (A) stream data on-the-fly (`scripts/train_sparknet.py`) or (B) build a static 1B-token dataset once and train on it (`scripts/train_sparknet_v5.py`). Both pipelines share the same callbacks/logging behavior.

### 1. (Optional) Train a Custom Tokenizer

```bash
python scripts/train_tokenizer_v5.py \
  --maybe_edit_script_for_sources
```

The script uniformly samples lines from the configured datasets, trains a byte-level BPE (GPT-2 compatible), and stores it under `tokenizer-v5/`. Edit `sources`, `MAX_LINES`, or `VOCAB_SIZE` inside the script if you want a different mix.

### 2. (Optional) Build a Static Packed Dataset

`train_sparknet_v5.py` expects a packed dataset created by `scripts/build_dataset_v5.py`:

```bash
python scripts/build_dataset_v5.py
```

This script:

- loads the tokenizer from `tokenizer-v5/`
- samples text using the weighted mix in `DATA_SOURCES`
- tokenizes until `TARGET_TOKENS` (default 1B) are accumulated
- packs sequences into 1024-token blocks
- writes the result to `datasets/sparknet-v5-1b/`

Adjust `DATA_SOURCES`, `TARGET_TOKENS`, or `BLOCK_SIZE` in the script to suit your run.

### 3. Launch Training

#### Streaming (no pre-built dataset)

```bash
python scripts/train_sparknet.py \
  --edit RUN_NAME/config path inside script if needed
```

Key characteristics:

- Loads the mix defined in `configs/datasets_v1.json` (change the filename near the top).
- Streams datasets with `datasets.interleave_datasets`, lazily tokenizes, and packs batches on the fly.
- Useful when experimenting with proportions or when disk is limited.

#### Static dataset (recommended for v5)

```bash
python scripts/train_sparknet_v5.py
```

- Consumes the packed dataset from `datasets/sparknet-v5-1b/`.
- Uses the custom tokenizer in `tokenizer-v5/`.
- Enables dropout, cosine LR, AdamW fused optimizer, `load_best_model_at_end`, etc.

Both scripts will:

- write checkpoints to `checkpoints/<RUN_NAME>/` (model weights, tokenizer, metadata)
- log metrics and throughput to `logs/<RUN_NAME>/` (view with `tensorboard --logdir logs`)
- emit periodic text samples as `sample_stepXXXX.txt` inside the checkpoint directory

You can resume training by re-running the same script pointing at the existing `output_dir`.

---

## Evaluating a Trained Model

- `scripts/eval_perplexity.py`: compute perplexity on validation sets.
- `scripts/eval_generation.py`: run qualitative generations from prompts.
- `scripts/eval_benchmarks.py`: plug into downstream benchmark harnesses.

Each script accepts the checkpoint path (e.g., `checkpoints/sparknet-70m-v5`) and will reuse the saved tokenizer/model.

---

## Tips & Troubleshooting

- If you hit dataloader bottlenecks, reduce `dataloader_num_workers` inside the training script.
- Hugging Face streaming benefits from a warm cache; keep `HF_DATASETS_CACHE` on fast SSD.
- When experimenting with batch sizes, keep `block_size * per_device_train_batch_size * gradient_accumulation_steps` within your GPU memory budget.
- For offline clusters without internet, set `HF_DATASETS_OFFLINE=1` and mirror the needed datasets beforehand.

Happy training!
