# SparkNet-400M v2 Pretraining Plan

Status: draft created on 2026-03-29

## Purpose

This document turns the high-level strategy from `sparknet-400m-v2-lessons-and-strategy.md` into a concrete pretraining plan for SparkNet-400M v2.

The target is not a coding model. The target is a helpful, lightweight chatbot base model that:

- explains clearly
- answers simple factual questions reasonably well
- has better language discipline than v1
- can later be instruction-tuned into a clean live website demo

The plan below is optimized for a single DGX Spark and is organized around three flights:

1. `120M` token smoke test
2. `12B` token main run
3. optional continuation from `12B` to `24B`

## Executive decision

My recommendation is to keep the architecture close to the current SparkNet-400M v1 model and move the big changes into:

- corpus design
- schedule design
- evaluation discipline

I do **not** recommend spending this cycle on a major architecture redesign. The v1 lessons point much more strongly at data and evaluation problems than at a fundamental parameter-count problem.

## What changes from v1

Keep:

- the current 400M-class decoder shape
- GQA
- tied embeddings
- bf16 training
- roughly `512k` tokens per optimizer step if the hardware allows it

Change:

- remove `ELI5`
- remove personal blog data from base pretraining
- add a factual reference corpus
- add a modest synthetic educational corpus
- reduce reliance on noisy or style-drifting sources
- stop treating `1024` vs `2048` context as a goal in itself
- design the learning-rate schedule so the `12B` checkpoint can continue cleanly to `24B`

## Architecture recommendation

Primary recommendation:

- keep the v1 architecture for v2 pretraining

Suggested v2 base config:

- parameters: current `~400M` class
- `hidden_size=1024`
- `num_layers=29`
- `num_heads=16`
- `num_kv_heads=8`
- `intermediate_size=2736`
- tied embeddings
- RMSNorm + RoPE + GQA

Why keep it:

- it already trains stably in this repo
- it already fits the intended product identity: SparkNet-400M
- changing the corpus while keeping the model fixed gives cleaner experimental readouts
- the expected gain from better data is larger than the expected gain from shifting to `350M`, `384M`, or `450M`

If you later want a more aggressive v2.5 run, that is the right time to test a `384M` or `450M` variant. It is not the highest-value variable for this cycle.

## Context window recommendation

Primary recommendation:

- pretrain at `1024`

Why:

- with your compute budget, more clean tokens matter more than doubling context length
- the v1 stack and throughput numbers are already proven at `1024`
- the product target is a lightweight chatbot, not long-document retrieval or coding

What I would not do:

- I would not halve training efficiency just to say the base model was pretrained at `2048`

If longer context becomes important for the demo:

- do a later context-extension step
- or fine-tune / continue-train a checkpoint with long-context recipes after the main base model is already good

## Data-source analysis

### Current v1 sources

v1 effectively used this mix in the packed pretraining dataset:

- `55%` `codelion/fineweb-edu-1B`
- `25%` `codelion/dclm-baseline-1B`
- `15%` `codelion/finepdfs-1B`
- `4%` `eli5`
- `1%` local blog data

That mix was a decent first pass, but it is not the best mix for a helpful chatbot base.

### Recommended v2 sources

### 1. FineWeb-Edu

Recommendation:

- keep as the anchor corpus
- make it the single largest source

Why:

- educational web text is a strong match for explanation, QA, and clear prose
- Hugging Face's small-model work uses FineWeb-Edu heavily in SmolLM
- it is much better aligned with "helpful chatbot" than noisy conversational data

Risk:

- if used alone, the model can become too narrow and too "educational-web-shaped"

Decision:

- **keep**
- **majority source**

### 2. DCLM-Baseline

Recommendation:

- keep, but reduce it to a support role

Why:

- it broadens the language distribution beyond purely educational text
- it adds natural web diversity and prevents the base model from becoming too synthetic or too textbook-like
- DCLM reports strong generalization benefits from diverse high-quality web data

Risk:

- the dataset card explicitly frames it as a research baseline, not a production-ready corpus
- it is still Common Crawl-derived and can carry web noise

Decision:

- **keep**
- **minority source**

### 3. FinePDFs

Recommendation:

- keep, but cap it

Why:

- it provides structured, expository, reference-like writing
- codelion's small-scale mixing work found that high-quality textbook/PDF-style data can strongly improve language discipline

Risk:

- PDF extraction noise is real
- too much PDF-style data can make the model dry, over-structured, or brittle
- codelion also found that overly synthetic / overly polished corpora can hurt generalization if they dominate

Decision:

- **keep**
- **moderate share only**

### 4. FineWiki

Recommendation:

- add it

Why:

- Wikipedia-style reference text is valuable for factual density, entity grounding, and compact explanations
- it is a much better factual anchor for a helpful chatbot than `ELI5`
- it complements FineWeb-Edu well because it is more reference-like and less article/blog-like

Risk:

- too much reference prose can make the model dry or overly encyclopedic

Decision:

- **add**
- **moderate share**

### 5. Cosmopedia

Recommendation:

- add a modest amount, not a dominant amount

Why:

- Cosmopedia gives synthetic textbooks, blog posts, and explanatory material across many topics
- SmolLM and related small-model work show that small models benefit from carefully curated educational synthetic data
- this source is directly aligned with the goal of producing clearer explanations and instructional prose

Risk:

- synthetic style homogenization
- repetitive textbook framing
- possible "assistant-like" or tutorial-ish tone bleeding into the base model if the share is too high

Decision:

- **add**
- **small to moderate share**

### 6. ELI5

Recommendation:

- remove it

Why:

- it is noisy
- it injects forum / Reddit style that is not the target product voice
- v1 already showed signs of style drift, markup junk, and weak answer discipline
- we now have better alternatives for explanatory prose

Decision:

- **drop**

### 7. Personal blog data

Recommendation:

- remove it from base pretraining

Why:

- it is too small to help materially
- it can still over-imprint tone or niche facts
- personal-brand alignment belongs in a later adapter or SFT stage if you want it

Decision:

- **drop**

### 8. Code / code-adjacent corpora

Recommendation:

- exclude them from this run

Why:

- the target is not a coding model
- the token budget is better spent on explanatory, factual, and reference text
- code would dilute the core chatbot objective

Decision:

- **drop for v2 pretraining**

## Final recommended v2 mixture

Primary v2 mixture:

- `50%` FineWeb-Edu
- `15%` DCLM-Baseline
- `15%` FinePDFs
- `10%` FineWiki
- `10%` Cosmopedia

Why this mix:

- it keeps educational web as the dominant backbone
- it keeps natural-web diversity, but no longer lets it dominate
- it adds a clean factual corpus
- it adds a small explanatory synthetic corpus without letting synthetic style run the model
- it removes the two lowest-value v1 components: `ELI5` and personal blog data

What I explicitly do **not** recommend:

- copying codelion's `50/30/20` mix directly

Why not:

- that result was found on a much smaller `70M` GPT-2-scale setup at `1B` tokens
- it optimized for generalization and perplexity, not specifically for a helpful chatbot base
- it does not use the stronger small-model educational-synthetic pattern now visible in SmolLM

What I **do** take from codelion's work:

- static mixing beats hard curricula
- polished reference text is useful
- diversity still matters
- hard distribution shifts are risky

## Mixing strategy

Recommendation:

- use a static mix for the whole main run

Do not:

- do a hard curriculum like "all educational first, then all diverse web"
- do a hard `12B` finish with full LR decay if you think you may continue to `24B`

Static mixing is the safer and simpler choice here.

## Training arguments

### Core settings

Recommended v2 core arguments:

- precision: `bf16=true`
- context length: `1024`
- effective tokens per step target: `524,288`
- optimizer: AdamW
- peak learning rate: `2e-4`
- weight decay: `0.1`
- max grad norm: `1.0`
- warmup: `2%` of the **full planned arc**
- scheduler: **trapezoidal / flat-top with cooldown**

Rationale:

- `2e-4` already trained stably in v1
- `weight_decay=0.1` is standard and already proven here
- `1024` preserves tokens-per-day efficiency
- the schedule should be chosen for optional continuation, not just the `12B` stop

### Learning-rate schedule recommendation

This is the most important training-argument decision in the whole plan.

If `24B` is even a plausible continuation target, design the run as a **single 24B-capable arc from the start**.

Recommended schedule:

- warmup: `2%`
- flat / high-LR phase: `78%`
- cooldown: `20%`

Applied to a `24B` plan:

- warmup over first `480M` tokens
- stay at or near peak LR through most of the run
- only start serious cooldown late in the `24B` arc

Implication:

- the `12B` checkpoint becomes a midpoint candidate, not a fully decayed endpoint
- you can stop cleanly at `12B`
- if the checkpoint looks strong, you can continue to `24B` without an awkward LR restart

What I would avoid:

- a cosine schedule that already decays to near-zero by `12B`

That would make later continuation possible, but clumsy. You would be forced into a second training stage with a new low LR and less clarity about whether the gain came from more tokens or a new schedule.

### Batch and throughput settings

Starting point:

- `per_device_train_batch_size=32`
- `grad_accum=16`
- `tokens_per_step=524,288`

This matches the proven v1 run geometry.

If DGX Spark does not like that exact microbatch:

- keep the **effective** token batch near `512k`
- change microbatch and accumulation, not the total target

Recommendation:

- do not enable gradient checkpointing unless you need it
- prefer throughput over squeezing every last token of microbatch size

### Evaluation settings

Pretraining eval should be better than v1.

Keep:

- Wikitext-2 validation

Add:

- a held-out mixed-corpus validation slice built from the same source families
- a small FineWiki held-out slice for factual/reference behavior
- a fixed generation prompt set with short factual, explanatory, and conversational prompts

Checkpoint cadence:

- eval every `250M-500M` tokens
- save every `500M` tokens
- retain explicit token checkpoints at `120M`, `1B`, `3B`, `6B`, `12B`, and optional `24B`

## Flight plan

### Flight 1: `120M` smoke test

Purpose:

- verify that the new data mix, tokenizer, and training arguments are stable
- confirm loss is falling normally
- confirm no dataloader or packing bottlenecks
- compare `1024` throughput against expectations

Recommended settings:

- architecture: same as planned main run
- context: `1024`
- target tokens: `120,000,000`
- LR schedule: same peak LR, shortened schedule
- steps at `524,288` tokens/step: about `229`

What to watch:

- loss curve shape
- gradient norm stability
- actual tokens/sec
- qualitative generations at steps `50`, `100`, `200`

Success criteria:

- stable loss decline
- no NaNs or optimizer instability
- no obvious packing / data corruption issues
- generations look at least as coherent as the v1 base did at comparable early training

### Flight 2: `12B` main run

Purpose:

- produce the real SparkNet-400M v2 base checkpoint

Recommended settings:

- same architecture
- same `1024` context
- static dataset mix from this document
- LR schedule planned as a `24B` arc
- target checkpoint for deployment candidate: `12B`

Step count at `524,288` tokens/step:

- about `22,889` steps

What to review at `12B`:

- validation loss trends
- generation quality on the fixed prompt set
- factual compactness
- verbosity discipline
- hallucination rate vs v1 base

Checkpoint decision at `12B`:

- if clearly strong, freeze and move to tokenizer / SFT work
- if still improving and compute tolerance remains acceptable, continue to Flight 3

### Flight 3: optional continuation to `24B`

Purpose:

- find out whether SparkNet-400M still benefits meaningfully from more clean tokens

Recommendation:

- continue the same run
- do **not** rebuild the data mix
- do **not** restart with a fresh high LR

Step count for full `24B`:

- about `45,777` steps

When continuation makes sense:

- the `12B` checkpoint is clearly better than v1
- eval and prompt-review curves have not flattened completely
- wall clock and hardware patience are still acceptable

When not to continue:

- the gains from `6B` to `12B` are already marginal
- the model is still failing for reasons that look like SFT or eval problems rather than base capability problems
- the run is compute-prohibitive for the next iteration cycle

## Estimated wall clock

Very rough estimate using existing local throughput and published DGX Spark numbers:

- local SparkNet benchmark: about `7.9k` tokens/sec
- NVIDIA DGX Spark fine-tuning reference: roughly `13.5k` tokens/sec on one Llama `3.2 3B` recipe

Inference:

- `12B` tokens is likely on the order of roughly `10-18` days
- `24B` tokens is likely on the order of roughly `21-35` days

These are planning numbers, not guarantees. The `120M` smoke test should be used to replace them with your own measured estimate.

## Recommended final decision

If I were locking the plan today, I would use:

- model: current SparkNet `~400M` architecture
- tokenizer: retrained on the final v2 corpus
- context: `1024`
- peak LR: `2e-4`
- optimizer: AdamW
- weight decay: `0.1`
- batch target: `~524k` tokens/step
- schedule: single `24B`-capable trapezoidal arc
- v2 corpus:
  - `50%` FineWeb-Edu
  - `15%` DCLM-Baseline
  - `15%` FinePDFs
  - `10%` FineWiki
  - `10%` Cosmopedia

That is the cleanest version of "SparkNet-400M v2 pretraining" I can justify from:

- the local v1 lessons
- current open small-model practice
- your stated product goal
- the realities of a single DGX Spark

## Sources

- Existing project lessons: `docs/sparknet-400m-v2-lessons-and-strategy.md`
- Existing SparkNet v1 run config: `checkpoints/sparknet-400m-v1/run_config.json`
- Existing SparkNet pretraining dataset metadata: `datasets/sparknet-v6-pretrain/shard-000/metadata.json`
- FineWeb-Edu dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- DCLM-Baseline dataset card: https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
- FineWiki sample dataset: https://huggingface.co/datasets/codelion/finewiki-1B
- Cosmopedia dataset card: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia
- SmolLM blog: https://huggingface.co/blog/smollm
- codelion dataset mixing article: https://huggingface.co/blog/codelion/optimal-dataset-mixing
- Chinchilla paper summary: https://huggingface.co/papers/2203.15556
- DGX Spark hardware overview: https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- DGX Spark performance blog: https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/
