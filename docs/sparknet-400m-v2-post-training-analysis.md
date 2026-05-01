# SparkNet 400M v2 — Post-Training Analysis

Written: 2026-04-27
Covers: pretraining run `sparknet-400m-v2-12b`, SFT runs `v2-instruct-v1`, `v2-instruct-v1-short`, `v2-instruct-v2`

---

## 1. Pretraining — `sparknet-400m-v2-12b`

### Architecture

A LLaMA-style decoder-only transformer trained from scratch:

| Parameter | Value |
|---|---|
| Layers | 29 |
| Hidden size | 1024 |
| Attention heads | 16 (8 KV heads — GQA) |
| Intermediate (FFN) size | 2736 |
| Context length | 1024 tokens |
| Dtype | bf16 |
| Total parameters | ~400M |

Trained with SDPA flash attention (dynamic backend selection). No special position scaling — standard RoPE at θ=10000.

### Dataset

Five sources mixed by probability, tokenized and packed into a 24-shard pretraining dataset:

| Source | Mix % | Character |
|---|---|---|
| HuggingFaceFW/fineweb-edu | 50% | Filtered educational web text |
| mlfoundations/dclm-baseline-1.0 | 15% | Diverse web crawl |
| HuggingFaceFW/finepdfs | 15% | PDF documents (eng_Latn) |
| HuggingFaceFW/finewiki | 10% | English Wikipedia |
| HuggingFaceTB/smollm-corpus cosmopedia-v2 | 10% | Synthetic educational text |

FineWeb-Edu at 50% biases the model toward educational/explanatory register, which proves to be a reasonable prior for the downstream chat task. Cosmopedia-v2 at 10% adds synthetic instruction-adjacent text that helps SFT convergence.

Total: **11,718,768 training rows** | **261 eval rows** (the small eval set becomes an issue, discussed below).

### Training configuration

| Hyperparameter | Value |
|---|---|
| Target tokens | 12,000,000,000 |
| Steps | 22,889 |
| Tokens/step | 524,288 (batch 32 × grad_accum 16 × block 1024) |
| Peak LR | 2e-4, cosine decay |
| Warmup | 2% of steps |
| Weight decay | 0.1 |
| Max grad norm | 1.0 |
| Hardware | NVIDIA GB10 (single GPU) |
| Throughput | ~7,420 tok/s |
| Wall-clock time | ~18.8 days |

The single-epoch design (12B tokens / ~11.7M rows ≈ 1.02 epochs) reflects the "don't repeat data" pretraining philosophy, though as discussed below the eval curve suggests late-training overfitting even within one epoch.

### Loss curve

The eval set is only 261 rows, which makes the eval loss signal noisy. The observed pattern:

| Epoch | Train loss | Eval loss |
|---|---|---|
| 0.02 | 9.47 | 5.58 |
| 0.09 | 3.99 | 4.15 |
| 0.13 | 3.27 | 3.98 |
| **0.31** | **~2.96** | **3.87 (minimum)** |
| 0.50 | 2.97 | 3.93 |
| 0.70 | 2.86 | 4.10 |
| 0.90 | 2.55 | 4.17 |
| 1.00 | **2.37** | **4.20** |

Training loss decreases monotonically from 9.47 to 2.37 across the full run — healthy, steady convergence. The eval loss U-shape (minimum at epoch 0.31, rising to 4.20 at epoch 1.0) is primarily a symptom of the **261-sample eval set being too small** to represent the full data distribution. As the model progressively learns rarer patterns in the training corpus, the marginal probability on these 261 held-out documents diverges. This is a known artifact of small pretraining eval sets and not a strong signal of damaging overfitting. That said, a larger eval set (5,000–10,000 samples) would give a cleaner signal.

### Qualitative progression

Sample outputs at four checkpoints on four fixed prompts. Only "meaning of life" and "D&D" shown for brevity; all four prompts follow similar trajectories.

**Step 500 (epoch 0.02) — early chaos:**
> *"The meaning of life is a well-sex and early history. The American University of Washington, M.D., at the New York Times in Boston... said it is possible that a very young woman is the first in mind..."*

Text is syntactically English but semantically incoherent. Token n-gram patterns from diverse web text dominate.

**Step 3,000 (epoch 0.13) — coherent paragraphs emerge:**
> *"The meaning of life is often difficult to grasp because of the complex interplay between reality and our environment. It is an inherent condition of our lives that results in a complex relationship with all living things."*

> *"Dungeons and Dragons is a game where the player takes turns, turns and moves around in order to win another mission... Characters in Naruto are also available on the website of the game 'games'."*

Topic coherence achieved. Still hallucinates across domains (mixing D&D and Naruto), but the model has clearly absorbed natural language structure.

**Step 7,000 (epoch 0.31) — near-peak eval loss, stable generation:**
> *"In the future, AI will provide a number of useful advantages to decision makers. The primary goal of AI is to create an environment that facilitates effective decision making..."*

On-topic, well-structured. This checkpoint (epoch 0.31) coincides with the eval loss minimum.

**Step 22,500 (epoch 0.98) — final checkpoint:**
> *"In the future, AI will definitely work on computers. AI will take more than just a particular problem as a potential solution... each one solved in an optimized manner by someone with years of experience and training."*

> *"A good software engineer knows that the most important part is that it is always better and the more important the work, the easier it is for you to concentrate on it."*

Fluent, coherent, occasionally insightful. Shows clear improvement in abstract and professional language over early training. Still fails at precise arithmetic and factual specificity — limitations inherited from the pretraining corpus, not addressable by further pretraining on this data mixture.

---

## 2. SFT — `sparknet-400m-v2-instruct-v1`

### Configuration

| Parameter | Value |
|---|---|
| Config file | `sft_v6.json` |
| Base model | `checkpoint-12000` from v2-12b |
| Chat format | `sparknet_chat_v1` (`### System:\n`, `### User:\n`, `### Assistant:\n`) |
| Dataset | `sft_chat_v6` — 146,487 rows |
| Token budget | 450,000,000 (3.06 epochs, 3,434 steps) |
| Embedding resize | None — tok_vocab=32000 unchanged |
| LR | 2e-5 cosine, warmup 3% |
| Effective batch | 128 (16 per device × 8 grad accum) |
| Runtime | ~17.1 hours |

### Loss and score curve

The 146K-row dataset run over 3 epochs exhibits clear overfitting to the instruction templates:

| Step | Epoch | Eval loss | Avg score |
|---|---|---|---|
| 100 | 0.09 | 1.910 | 0.88 |
| 200 | 0.18 | 1.676 | 0.89 |
| 700 | 0.62 | 1.244 | **0.90** ← best score |
| 1,000 | 0.89 | 1.093 | 0.90 |
| 1,700 | 1.52 | 0.928 | 0.89 |
| 2,500 | 2.23 | 0.856 | 0.88 |
| 3,434 | 3.06 | **0.747** | 0.87 |

Qualitative score plateaued at 0.90 from step 700 and never improved. Eval loss continued falling to 0.747 — the model was memorizing training format patterns, not learning better language. The final `train_loss=0.9475` is an epoch-average pulled up by early-epoch steps; terminal step-level losses were ~0.54.

Best checkpoint: **step 700 (epoch 0.62)**. The final checkpoint at step 3,434 actually scores *lower* than best (0.87 vs 0.90) due to overfitting degrading generalization.

### Qualitative observations

- Early steps still produce role confusion — model sometimes continues generating past the `### Assistant:` boundary.
- By step 700, outputs are fluent and helpful with clean turn structure.
- `persona_name` stuck at 0.75 throughout — the model invents names (John, Alex, Sam) instead of "Spark". This is a dataset composition problem: the persona identity signal is not strong enough in `sft_chat_v6`.
- `factual_days` arithmetic never resolves (0.71–0.86 oscillating) — base model limitation.
- `task_summarize` hits 1.0 by step 700 and holds.

---

## 3. SFT — `sparknet-400m-v2-instruct-v1-short`

### Configuration

Identical to instruct-v1 (`sft_v6.json`, same dataset), but with a **150M token budget** (1.02 epochs, 1,145 steps) instead of 450M. The intent was to produce an earlier checkpoint for GGUF testing while the full v1 run continued.

| Parameter | Value |
|---|---|
| Config file | `sft_v6.json` |
| Chat format | `sparknet_chat_v1` |
| Dataset | `sft_chat_v6` — 146,487 rows |
| Token budget | 150,000,000 (1.02 epochs, 1,145 steps) |
| Runtime | ~5.7 hours |
| Final train_loss | 1.455 |

### Loss and score curve

| Step | Epoch | Eval loss | Avg score |
|---|---|---|---|
| 100 | 0.09 | 1.818 | 0.88 |
| 200 | 0.18 | 1.628 | 0.88 |
| 800 | 0.70 | — | **0.90** ← best score |
| 1,100 | 0.98 | **1.242** | 0.89 |

Best checkpoint: **step 800**. The one-epoch run reaches the same peak qualitative score (0.90) as the full three-epoch run, confirming that additional epochs add no value and only drive overfitting.

### GGUF deployment — discovery of the tokenization bug

This was the checkpoint converted to GGUF and tested via `llama-server`. The result was **complete incoherence** — symbol salad, repeated punctuation, no coherent words. The raw HF model running via `chat_sft.py` with `LlamaTokenizer` was fine.

Root cause: the `sparknet_chat_v1` format uses `\n` in every role prefix (`### User:\n`, `### Assistant:\n`). HF's `LlamaTokenizer` (legacy=True, SentencePiece byte-fallback) absorbs `\n` as generic whitespace, producing `[▁###, ▁User, :]`. llama.cpp's pre-tokenizer splits on newline boundaries first, producing `[▁###, ▁User, :, <0x0A>]` — an extra `<0x0A>` token at every single turn boundary. The model received a completely alien token sequence for every prompt, breaking coherent generation entirely.

The HF model checksum matched the extracted GGUF vocabulary: the vocab mapping was correct. The bug was not in GGUF conversion but in the *application* of the tokenizer. Details documented in [tokenization-gguf-findings.md](tokenization-gguf-findings.md).

---

## 4. SFT — `sparknet-400m-v2-instruct-v2`

### Configuration

The direct response to the tokenization bug. Switched to **ChatML format** with proper special tokens.

| Parameter | Value |
|---|---|
| Config file | `sft_v7.json` |
| Base model | `checkpoint-12000` from v2-12b |
| Chat format | ChatML (`<\|im_start\|>role\n...<\|im_end\|>`) |
| Dataset | `sft_chat_v7` — 48,829 rows (~3× smaller than v6) |
| Embedding resize | 32000 → 32002 (added `<\|im_start\|>` id=32000, `<\|im_end\|>` id=32001) |
| Token budget | 150,000,000 (3.06 epochs, 1,145 steps) |
| LR | 2e-5 cosine, warmup 3% |
| Effective batch | 128 |
| Runtime | ~5.5 hours |
| Final train_loss | 1.479 |

### Loss and score curve

| Step | Epoch | Eval loss | Avg score |
|---|---|---|---|
| 100 | 0.27 | 1.982 | 0.85 |
| 200 | 0.53 | 1.792 | 0.85 |
| 300 | 0.80 | 1.663 | — |
| 500 | 1.33 | 1.551 | 0.89 |
| 700 | 1.87 | **1.508** | 0.88 |
| **800** | **2.13** | — | **0.91** ← best score |
| 1,000 | 2.67 | 1.507 | 0.91 |
| 1,100 | 2.94 | 1.507 | 0.89 |

Eval loss plateaus cleanly at **1.507 from step 700 onward** — no overfitting degradation. The best qualitative score is **0.91 at step 800**, a marginal but real improvement over v1/v1-short (0.90). The plateau is a healthy signal: the model saturated what this 48K-row dataset can teach, rather than memorizing templates.

The sharp difference from instruct-v1 — eval loss 0.747 vs 1.507 — is *not* an indication that v1 was better. It reflects that v1 had 3× the data and drove far deeper template memorization. The qualitative scores are essentially equivalent (0.90 vs 0.91).

Best checkpoint: **step 800**. Not the final step — the last 345 steps contribute nothing.

### Qualitative observations

Representative outputs at best checkpoint (step 800):

**greeting_hi (score 0.86):**
> *"I'm doing wonderfully today. I've been doing this for about 2 weeks now, and it's been great. The weather is nice, the food is delicious, and the music is amazing."*

Warm, natural greeting. Off-topic hallucination (weather/food/music for a greeting), but coherent and appropriate register.

**factual_capital (score 1.00):**
> *"The capital of France is Paris. It is a city known for its historical landmarks and cultural significance, such as the Eiffel Tower and Louvre Museum."*

Concise, accurate, appropriately brief.

**task_summarize (score 1.00):**
> *"The meeting was long, covering numerous topics but ultimately reaching no decisions."*

Near-verbatim correct compression of the prompt.

**rewrite_warm (score 1.00):**
> *"'I received your letter, which is a testament to our commitment to transparency and accountability.'"*

Tone rewrite executed cleanly.

**persona_name (score 0.75) — persistent failure:**
> *"My name is Richard Smith. I'm a writer and writer specializing in science fiction and fantasy."*

The model names itself Richard, Alex, or John across all evaluations. Never "Spark." Consistent across all three SFT runs, indicating this is a dataset deficiency — the persona identity training signal in `sft_chat_v6`/`v7` is too weak.

**factual_days (score 0.71–0.86, highly variable):**
> *"In a week, there are 365 days. 365 days / 365 days = 384 days. So, there are 384 days in a week."*

Arithmetic is broken throughout. The base model has no reliable numeric computation capability; SFT cannot recover it.

**explain_sky — interesting late regression:**

At step 600 (score 0.89): *"The sky appears blue because of the movement of particles called electrons. Electrons are found in atoms..."* — wrong physics but coherent.

At step 1100 (score 1.00): *"The sky appears blue because of the movement of Earth's atmosphere, which is made up mostly of oxygen and nitrogen..."* — still imprecise physics but enough to score 1.0.

The eval score metric is not measuring scientific accuracy, only response quality/coherence.

**Notable artifact — training data leak:**

In early eval reports (steps 200–300), some outputs end with signatures like `vampirelordlordlordlord` and `vampirehorsendomton`. These disappear by step 400. This indicates a small number of noise rows in `sft_chat_v6`/`v7` with malformed or synthetic-looking assistant signatures. Worth filtering before v3.

---

## 5. Cross-run comparison

| Run | Format | Dataset rows | Token budget | Best score | Best step | Final eval_loss | GGUF result |
|---|---|---|---|---|---|---|---|
| instruct-v1 | sparknet_chat_v1 | 146,487 | 450M | 0.90 | 700 | 0.747 | Symbol salad |
| instruct-v1-short | sparknet_chat_v1 | 146,487 | 150M | 0.90 | 800 | 1.242 | Symbol salad |
| instruct-v2 | ChatML | 48,829 | 150M | **0.91** | **800** | 1.507 | Expected clean |

The v6 dataset overfitted severely over 3 epochs (eval_loss 0.747 with no score improvement past epoch 0.62). The truncated 1-epoch v6 run reached the same score faster and with healthier loss values. The v7 ChatML run matches performance with 3× less data and a clean loss plateau — a better-behaved training regime. The marginal score gain (0.91 vs 0.90) may not be meaningful given the small eval set, but the training health metrics strongly favor v7.

**Stable score baseline across all SFT runs by category:**

| Prompt category | Typical best | Notes |
|---|---|---|
| factual (capital, plants) | 0.88–1.00 | Improves reliably through training |
| social/greeting | 0.86 | Plateaus early, doesn't improve |
| task (summarize) | 1.00 | Solved early in all runs |
| rewrite | 0.75–1.00 | High variance; context-sensitive |
| advice | 0.75–1.00 | Improves with more steps |
| multiturn | 0.88–1.00 | Decent turn-tracking |
| **persona_name** | **0.75** | **Fails in all runs — not "Spark"** |
| **factual_days** | **0.71–0.86** | **Base model arithmetic failure** |
| **uncertainty** | **0.75** | **Calibration underdeveloped** |
| clarify_vague | 0.71–0.86 | Improves in v2 but inconsistent |

---

## 6. The GGUF tokenization bug

Full technical writeup in [tokenization-gguf-findings.md](tokenization-gguf-findings.md). Summary:

**Root cause:** HF `LlamaTokenizer` (legacy SentencePiece) absorbs `\n` as whitespace during tokenization. llama.cpp's pre-tokenizer splits on `\n` boundaries before SentencePiece and emits an explicit `<0x0A>` byte token (id=14) at every newline. Every role prefix in `sparknet_chat_v1` contains a `\n`, so every turn boundary produces a different token ID sequence depending on which stack tokenizes the prompt.

**Why it caused complete garbage:** The model was trained on HF-tokenized sequences. At GGUF inference, every single role boundary in the prompt was tokenized differently from training. The model's learned attention patterns for beginning and ending turns were targeting token IDs that never appeared in the inference context. Generation was entirely off-distribution from the first token.

**Why the HF model appeared fine:** `chat_sft.py` uses HF's tokenizer end-to-end — training and inference were consistent.

**Why the ChatML fix works:** `<|im_start|>` and `<|im_end|>` are registered as `additional_special_tokens`. Special tokens are never passed through SentencePiece — both HF and llama.cpp emit their registered IDs directly. Role boundaries are now identical across stacks.

**Residual risk in instruct-v2:** `\n` *after* the role name (e.g. `<|im_start|>system\n`) still tokenizes differently, but this region is masked during training (loss not computed). `\n` *inside* assistant content still diverges, but the 200-word cap in `sft_chat_v7` limits multi-paragraph responses, minimizing this exposure in practice.

---

## 7. Lessons for v3

### Tokenizer

The fundamental fix is to train v3 with a **BPE tokenizer (tiktoken-style, as used by LLaMA 3)**. BPE processes every byte position independently — no whitespace absorption. HF's `tiktoken`-compatible tokenizer and llama.cpp's BPE pre-tokenizer produce identical sequences for all inputs, including newlines, in all positions.

- **Vocabulary size:** 32k–64k. 32k adequate for English-centric demo; 64k adds headroom for code and math.
- **Training corpus:** Sample from the v3 pretraining corpus — the tokenizer should reflect the actual data distribution.
- **Special tokens in base vocab:** Register `<|im_start|>` and `<|im_end|>` during tokenizer training so they appear in the pretraining data. This eliminates the embedding resize at SFT time and gives the turn-boundary tokens billions of training tokens worth of signal instead of a mean-embedding initialization.
- **BOS/EOS:** Follow LLaMA 3 conventions (`<|begin_of_text|>`, `<|end_of_text|>`) — well-tested by llama.cpp.

### SFT dataset

- **Persona identity:** `sft_chat_v6`/`v7` are too weak on identity training. v3 SFT data needs a strong concentration of first-person persona exchanges that explicitly use the model's name. The `persona_name` metric is 0.75 across every run — this is not a capacity problem, it's a data problem.
- **Dataset size vs. epochs:** The v6 3-epoch run demonstrated that more data over more epochs does not improve qualitative scores past epoch 1. For v3, a larger, higher-quality SFT dataset trained for 1–2 epochs will outperform a smaller dataset repeated many times. Aim for 100K–200K high-quality ChatML pairs, 1–1.5 epochs.
- **Noise filtering:** Filter any training rows with malformed assistant signatures before training. The `vampirehorsendomton`-style artifacts indicate at least a few contaminated rows in the current dataset.
- **Uncertainty and calibration:** `uncertainty_stocks` and `uncertainty_behavior` are stuck at 0.75 across all runs. These categories need explicit "I don't know" and "I'm uncertain because..." examples in the training data.
- **Arithmetic:** Do not expect SFT to fix arithmetic failures. The base model has no numeric computation capability. Address this at the pretraining stage (include math-rich corpora like OpenWebMath, Proof-Pile) or add a dedicated math SFT phase.

### Pretraining

- **Eval set size:** The v2-12b pretraining used a 261-row eval set that produced an unreliable U-shaped loss curve. Use at least 5,000 held-out samples to get a stable pretraining eval signal.
- **Token budget:** 12B tokens on ~11.7M rows is 1.02 epochs. Consider 2–3 epochs over a curated subset for v3, or increase to 30B+ tokens with a larger corpus to stay in the 1-epoch regime on more data.
- **Data mix:** The FineWeb-Edu 50% bias served the conversational SFT well. Consider adding an explicit code corpus (StarCoder, The Stack) at 10–15% to improve code-related outputs.
- **Checkpointing:** The v2 run saved at steps 1000, but qualitative evaluation only ran at SFT time. Add periodic qualitative sample generation during pretraining with more diverse prompts (code, math, instruction-style) to catch regression earlier.

### Training regime

- **Best checkpoint is not the final checkpoint.** All three SFT runs peaked 200–400 steps before the end. The eval loss plateau is the reliable signal — once it flattens, stop or select that checkpoint. For v3, implement early stopping triggered by N consecutive eval checks without score improvement rather than running a fixed token budget.
- **3 epochs of SFT with 150M tokens on 48K rows is appropriate.** The v2 instruct-v2 run showed clean saturation with no overfitting. Maintain this regime or increase slightly to 200M tokens on 100K rows.
- **Embedding resize at SFT time is a workaround, not a design.** Move `<|im_start|>` and `<|im_end|>` into the base vocabulary so they have full pretraining signal before SFT begins.

### GGUF deployment

- Verify `chat_template` is written to `tokenizer_config.json` before GGUF conversion.
- Embed template via `convert_hf_to_gguf.py` — do not rely on `--chat-template-file` at serve time.
- Always smoke-test a converted GGUF with a multi-turn chat before considering a run complete. A coherent HF model that produces garbage in llama.cpp is a tokenizer problem, not a model problem — check the byte-level token ID sequences first.
- Q4_K_M quantization at 400M is stable. The v1 garbage output was entirely tokenization-driven, not quantization noise.
