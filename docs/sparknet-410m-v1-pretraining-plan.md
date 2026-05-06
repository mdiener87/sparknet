# SparkNet-410M v1 Pretraining Plan

Status: run-readiness update in progress 2026-05-06

## Context

This document records all architectural, tokenizer, hyperparameter, data, and tooling decisions for the SparkNet-410M v1 pretraining run. It supersedes the earlier `sparknet-v3-hyperparameter-strategy.md` planning document.

**Naming note.** What was internally referred to as the "v3" training run is now called `sparknet-410m-v1`. The parameter count increased from ~368M (v2, 29 layers) to ~410M (v1, 32 layers), making the 400M label inaccurate. A clean directory — `scripts/sparknet-410m/`, `configs/sparknet-410m/` — was created rather than adding a third generation of scripts to the already-crowded `sparknet-400m/` directory. The `v1` suffix reflects that this is the first run at the 410M scale, not the first run of the project.

---

## What changes from v2

| Area | v2 | v1 (this run) |
|---|---|---|
| Architecture | 29 layers, intermediate 2736, ~368M params | 32 layers, intermediate 2816, ~410M params |
| Tokenizer | SentencePiece v6 (32k, `LlamaTokenizer`) | ByteLevel BPE v8 (32k, `PreTrainedTokenizerFast`, full byte alphabet) |
| ChatML tokens | Added at SFT time (embedding resize) | Baked into base vocab, registered as additional specials, plus 0.5% ChatML pretraining slice |
| LR | 2e-4 (inherited from v1, untested) | Pending 410m LR screen + 250M canary |
| Warmup | 2% | 2% default; confirmed by canary before full run |
| Grad accum | 16 → 524k tok/step | 32 → 1M tok/step (Phase 1 throughput test) |
| Cosine floor | None (decays to zero) | 10% of peak LR |
| Eval dataset | WikiText-2 (261 rows, wrong distribution) | Hash-disjoint held-out corpus shard (9,765 rows, same distribution) |
| Checkpoint retention | 5 rolling checkpoints | Top-3 by loss + 5 cardinal token snapshots + last-3 rolling |
| Data mix | 5 sources, no code, no math | 7 sources, +7% python-edu, +5% open-web-math |
| Token budget | 12B | 10B (Chinchilla-optimal; 12B is 20% overtraining) |

---

## Architecture

SparkNet-410M v1 is a LLaMA-style decoder-only transformer trained from scratch in bf16 with Flash Attention (SDPA).

| Parameter | Value | Notes |
|---|---|---|
| `hidden_size` | 1024 | Unchanged from v2 |
| `num_layers` | 32 | v2 was 29; 32 is a conventional even depth, better for potential future multi-GPU |
| `num_heads` | 16 | |
| `num_kv_heads` | 8 | GQA; unchanged |
| `intermediate_size` | 2816 | 11 × 256; full 256-alignment for CUDA Tensor Cores (v2 was 2736 = off-aligned) |
| `block_size` | 1024 | |
| `rope_theta` | 500000.0 | LLaMA 3-style; low-frequency RoPE period extends to ~3M tokens, enabling clean inference beyond training context without post-hoc extension tricks |
| `tie_word_embeddings` | True | Saves ~32M parameters |
| **Total parameters** | **~410M** | |

---

## Tokenizer — the primary deployment fix

### Why SentencePiece was replaced

SparkNet-400M v2's GGUF deployment failed completely: llama-server produced symbol salad while the identical HF model ran fine. Root cause: `LlamaTokenizer` (SentencePiece, `legacy=True`, `add_prefix_space=True`) absorbs `\n` as generic whitespace. llama.cpp's pre-tokenizer splits on newline boundaries first and emits an explicit `<0x0A>` byte token (id=14) at every `\n`. Every role prefix in the training format contained a `\n`, so every single turn boundary was tokenized differently at training time vs. inference time. The model's learned attention patterns for turn structure were entirely off-distribution from the first generated token.

The ChatML v7 fix (registering `<|im_start|>` and `<|im_end|>` as `additional_special_tokens`) resolved role boundaries, but left `\n` inside content still diverging. Full resolution requires eliminating SentencePiece.

Full writeup: `tokenization-gguf-findings.md`.

### ByteLevel BPE

ByteLevel BPE processes every byte position independently. There is no whitespace-absorption step. Both HF's `PreTrainedTokenizerFast` and llama.cpp's BPE pre-tokenizer produce identical token sequences for all inputs — newlines included, in all positions. The mismatch is eliminated structurally.

| Decision | Value | Rationale |
|---|---|---|
| Algorithm | ByteLevel BPE | Identical HF/llama.cpp tokenization |
| Vocabulary size | 32,000 | Sufficient for English-centric demo |
| Training corpus | Sampled from v3 data mix | Tokenizer reflects actual data distribution |
| `add_prefix_space` | False | Required for tiktoken-style byte independence |
| BOS token | `<|begin_of_text|>` | LLaMA 3 convention; well-tested by llama.cpp |
| EOS/PAD token | `<|end_of_text|>` | |
| `<|im_start|>` | In base vocabulary + `additional_special_tokens` | Receives base pretraining gradient from the 0.5% ChatML slice; no SFT resize |
| `<|im_end|>` | In base vocabulary + `additional_special_tokens` | Same |
| Chat template | ChatML (Jinja2, embedded in `tokenizer_config.json`) | Baked into GGUF automatically; no `--chat-template-file` at serve time |

### Why ChatML tokens belong in the base vocabulary

In v2, `<|im_start|>` and `<|im_end|>` did not exist at pretraining time. At SFT they were added as `additional_special_tokens`, growing the embedding table from 32000 to 32002. The two new rows were initialized to the mean embedding — a generic initialization with no learned signal. A single 5-hour SFT pass is not enough to build reliable semantics for a brand-new vocabulary entry.

In v1, both tokens are in the base vocabulary from the start and registered as `additional_special_tokens` in tokenizer metadata. A deterministic 0.5% pretraining slice wraps accepted training documents as ChatML continuations, so these embeddings receive real pretraining gradients before SFT. The SFT pass then refines semantics already grounded in context.

---

## Hyperparameters

The old 400m hparam logs remain historical context. Final 410m LR selection is gated on the ported `scripts/sparknet-410m/hparam_*.py` harness using tokenizer-v8 and the hash-disjoint corpus eval shard.

### Batch size (Phase 1)

`grad_accum = 32` selected based on throughput testing. This gives:

```
tokens per step = block_size × per_device_batch × grad_accum
               = 1024 × 32 × 32
               = 1,048,576 (~1M tokens/step)
```

Peak VRAM: 78.54 GB of 128 GB. No OOM risk.

### Learning rate gate

The 400m Phase 2 LR range test found a high ceiling, but the old Phase 3 grid was too short and used an eval setup that has since been replaced. The 410m run therefore does not treat 1.83e-3 as final until the following gate passes:

1. Run four 100M-token LR screens at `5.5e-4`, `9e-4`, `1.2e-3`, and `1.83e-3`.
2. Prefer the lower LR when eval/train losses are within 0.02 nats.
3. Run one 250M-token production-scheduler canary at the selected LR.
4. Reject any LR with NaN/inf, post-warmup loss jump, sustained gradient-norm instability, or clearly worse eval loss.

### Final hyperparameter table

| Hyperparameter | Value | Source |
|---|---|---|
| `learning_rate` | TBD from 410m LR screen + canary | Candidate set: 5.5e-4, 9e-4, 1.2e-3, 1.83e-3 |
| `warmup_ratio` | 0.02 | Production default; canary gate |
| `grad_accum` | 32 | Phase 1 |
| `weight_decay` | 0.1 | Unchanged from v2 |
| `max_grad_norm` | 1.0 | Unchanged |
| `scheduler` | `cosine_with_min_lr` | See below |
| `cosine_min_lr_ratio` | 0.1 | LR floor = 1.83e-4 at end of run |
| `target_tokens` | 10B | Chinchilla-optimal for ~410M |
| `tokens_per_step` | 1,048,576 | Phase 1 |
| `max_steps` | ~9,538 | target_tokens / tokens_per_step |

### Cosine schedule floor

The v2 run used a plain cosine schedule that decays LR to zero. With a 16-day run, the final ~10% of training steps (roughly the last 1.5 days) operate at near-zero LR and contribute almost nothing. A non-zero floor prevents over-decay without adding a hyperparameter to tune — 10% of peak LR (`min_lr = 1.83e-4`) is standard practice for models of this scale.

Implementation: `lr_scheduler_type = "cosine_with_min_lr"` with `lr_scheduler_kwargs = {"min_lr_rate": 0.1}`. Requires `transformers >= 4.38`; the training script falls back to plain cosine with a warning if the installed version is older.

---

## Data mix

### Source composition

| Source | HF dataset | v2 weight | v1 weight | Role |
|---|---|---|---|---|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | 50% | 44% | Educational/reference web text |
| DCLM | `mlfoundations/dclm-baseline-1.0` | 15% | 13% | Diverse web crawl |
| FinePDFs | `HuggingFaceFW/finepdfs` (eng_Latn) | 15% | 11% | Long-form documents |
| FineWiki | `HuggingFaceFW/finewiki` (en) | 10% | 10% | English Wikipedia |
| Cosmopedia-v2 | `HuggingFaceTB/smollm-corpus` | 10% | 10% | Synthetic educational |
| python-edu | `HuggingFaceTB/smollm-corpus` | — | **7%** | Code (Python-focused) |
| OpenWebMath | `open-web-math/open-web-math` | — | **5%** | Mathematical text |

The v2 base model had no code or math in its pretraining corpus. Post-training analysis showed that SFT could not recover arithmetic or structured-reasoning capability that was never present in the base. Adding python-edu and OpenWebMath at moderate percentages seeds these capabilities in the base model rather than asking SFT to create them from nothing.

FineWeb-Edu is reduced from 50% to 44% to make room. The educational register it provides remains the dominant signal.

### Document packing and partitioning

Each document is packed as `[BOS] token_ids [EOS]`, then documents are concatenated and sliced into 1024-token blocks. BOS/EOS boundaries teach the model document structure at pretraining time. This is a change from v2 where only EOS was appended (no BOS prepended), consistent with LLaMA 3 convention.

Train/eval membership is determined by a stable hash over `source_name`, stripped text, and `holdout_salt="sparknet-410m-v1"`. The eval shard uses the 2% holdout bucket; training shards use the complement. This prevents overlap even though both builders stream the same upstream HF splits.

For train only, 0.5% of accepted text chunks are deterministically wrapped as ChatML continuations. This gives `<|im_start|>` and `<|im_end|>` meaningful pretraining updates without changing the main corpus character.

### Token budget

10B tokens at ~7,300 tok/s ≈ **16 days** continuous on the DGX Spark GB10.

20 training shards × 500M tokens/shard = 10B tokens. Chinchilla-optimal for a 410M model is ~8.2B tokens (20 × N); 10B is a modest ~22% overtrain that improves inference quality at minimal additional compute cost. The v2 run at 12B was 46% overtrain at the 368M scale; 10B is a more principled budget for 410M.

---

## Evaluation dataset

### v2 failure mode

The v2-12b pretraining used WikiText-2 validation as the eval set, yielding 261 packed blocks after tokenization. The loss curve showed a U-shape: minimum at epoch 0.31, rising to 4.20 at epoch 1.0. This was not a genuine overfitting signal — it was an artifact of 261 Wikipedia-domain samples diverging from a training distribution that increasingly specializes beyond Wikipedia content. The eval set was measuring how well the model predicted Wikipedia, not how well it was learning the pretraining corpus.

### v1 fix

The eval dataset (`datasets/sparknet-410m-v1-pretrain-eval`) is a single hash-held-out shard built from the **same source mix** as the training data. It contains:

```
10M tokens / 1024 tokens per block = 9,765 rows
```

This gives a loss signal that tracks the training distribution throughout the full run. Statistical precision: standard error ≈ 0.004 loss units — sufficient to reliably distinguish checkpoints separated by a small fraction of a nat.

The eval shard is hash-disjoint from train regardless of build order, but it should still be built first so LR screens and canaries can run before the full dataset build finishes.

---

## Checkpoint management

### v2 problem

The v2 run used `save_total_limit=5` (5 rolling checkpoints). This meant that good intermediate checkpoints were silently deleted as training progressed. Post-training analysis showed that the best SFT checkpoint was consistently 200–400 steps before the final step — if pretraining had a similar pattern, the best pretraining checkpoint may have been deleted before it could be used.

### v1 SmartCheckpointCallback

`SmartCheckpointCallback` in `train_pretrain.py` manages three named buckets:

**1. Best-3 by eval loss.** The top 3 checkpoints by eval loss are protected from pruning. Updated on every evaluation. Identified by a `.eval_loss` marker file written into each checkpoint directory at save time.

**2. Cardinal token snapshots.** A permanent copy is made at each token milestone: 6B, 7B, 8B, 9B, 10B. Copied to `checkpoint-{N}B-tokens/` directories which are never touched by the pruner. These provide fixed reference points for downstream analysis regardless of which checkpoints survive the rolling buffer.

**3. Last-3 rolling buffer.** The 3 most recent step-numbered checkpoints are always kept for power-failure resume via `--resume latest`. Older regular checkpoints are pruned once they are no longer in the best-3 or last-3 sets.

`save_total_limit` is set to `None` — the HF Trainer is explicitly prevented from managing checkpoint retention independently.

### Expected steady-state storage

At ~2GB per checkpoint:

| Bucket | Count | Storage |
|---|---|---|
| Best-3 (worst case all distinct) | 3 | ~6 GB |
| Cardinal snapshots (6B–10B) | 5 | ~10 GB |
| Last-3 rolling | 3 | ~6 GB |
| **Total (worst case)** | **11** | **~22 GB** |

Best-3 and last-3 will frequently overlap in practice, so typical steady-state is closer to 8–9 checkpoints (~16–18 GB).

---

## GGUF deployment checklist

Carried forward from `sparknet-400m-v2-post-training-analysis.md` and updated for v1:

1. Verify `tokenizer_config.json` contains `chat_template` (written automatically by `tok.save_pretrained()`).
2. Run `convert_hf_to_gguf.py` — the ChatML template is embedded automatically.
3. Quantize to Q4_K_M.
4. Launch `llama-server` with **no** `--chat-template-file` and **no** `--chat-template` flag. The template is embedded in the GGUF.
5. Smoke test with a multi-turn chat prompt via `curl`. Verify coherent output.
6. If the template is somehow not detected: `--chat-template chatml` (built-in llama.cpp alias).

With the BPE tokenizer, step 5 is expected to pass on the first try. If it does not, check token IDs for `<|im_start|>` and `<|im_end|>` between the HF checkpoint and the GGUF — they should be identical (not `additional_special_tokens` mismatches).

---

## Run sequence

```bash
# 1. Build tokenizer-v8 (once)
python scripts/sparknet-410m/build_tokenizer.py

# 2. Build eval shard (once, before training shards)
python scripts/sparknet-410m/build_dataset.py \
    --config configs/sparknet-410m/datasets_v1_eval.json

# 3. Build training shards (×20, can run sequentially)
for i in $(seq 1 20); do
    python scripts/sparknet-410m/build_dataset.py \
        --config configs/sparknet-410m/datasets_v1.json
done

# 4. Run LR screens + 250M canary before full launch
python scripts/sparknet-410m/hparam_run_all.py --grad-accum 32

# 5. Launch training (in tmux) after LR config is updated from hparam results
tmux new -s sparknet-410m-v1
./scripts/sparknet-410m/run_pretrain_v1.sh

# 6. Resume after interruption
python scripts/sparknet-410m/train_pretrain.py \
    --config configs/sparknet-410m/pretrain_v1.json \
    --resume latest
```

---

## Expected timeline

| Milestone | Approx. date |
|---|---|
| Tokenizer built | 2026-05-06 |
| Eval + training shards built | 2026-05-07 |
| Training launch | 2026-05-07 |
| 6B token checkpoint | 2026-05-16 |
| 10B token checkpoint (end) | 2026-05-23 |
