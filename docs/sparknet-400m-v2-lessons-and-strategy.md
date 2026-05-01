# SparkNet-400M v2: Lessons Learned and Training Strategy

Status: draft created on 2026-03-25

## Why this document exists

SparkNet-400M v1 proved that the repo can train and ship a from-scratch base model plus multiple SFT variants. It did not yet produce a small assistant that feels reliably useful. This document turns the current evidence into a reusable plan for v2.

The goal for v2 is pragmatic:

- train the best small language model we can on a single DGX Spark
- optimize for a live website demo, not benchmark-chasing for its own sake
- improve actual answer quality, not just loss curves or format compliance

## Executive summary

The main issue with v1 was not that the model was "too small" in the abstract. The bigger problems were:

- weak behavioral evaluation
- SFT datasets that were too narrow and too easy to game
- post-training that tried to repair missing capabilities that were never strongly learned in pretraining
- overconfidence in loss and heuristic scores that did not match qualitative outputs

My current recommendation for v2 is:

- keep the model in the `~360M-400M` range
- retrain from scratch with a cleaner, more deliberate data curriculum
- target roughly `8B-12B` pretraining tokens as the primary budget
- treat SFT as a narrow polish stage, not the main source of capability
- replace the current heuristic-only eval with a stronger human-judged prompt set and checkpoint review process

If we do that well, I expect a better result than simply making the model 10-20% larger.

## What v1 actually achieved

### Base model

Local evidence:

- `sparknet-400m-v1` used a 400M-class decoder with `hidden_size=1024`, `29` layers, `16` attention heads, `8` KV heads, and `block_size=1024`
- pretraining mix: `codelion/fineweb-edu-1B` `52%`, `codelion/dclm-baseline-1B` `22%`, `codelion/finepdfs-1B` `12%`, `eli5` `13%`, local blog data `1%`
- actual run budget was `6B` tokens, `11,445` steps, `4,394,538` packed rows
- best base checkpoint by eval loss was `checkpoint-6500`

Interpretation:

- the base model is real and reusable
- the pretraining stack is stable enough to build on
- the model learned language patterns, but not strong answer discipline or factual reliability

The base eval outputs are especially important. On 2026-03-25, the stored base eval summary reported an average heuristic score of `0.8479`, yet the actual generations were clearly poor: markup soup, code fragments, malformed answers, and incoherent task handling. That means the current evaluator is not trustworthy as a primary success metric.

### SFT arc

| Run | Dataset / budget | What changed | Result |
| --- | --- | --- | --- |
| `instruct` | `sft_chat_v1`, `200M` tokens | first full SFT pass | loss improved, but there is no strong evidence that behavior improved enough |
| `instruct-v2` | `sft_chat_v2`, `200M` tokens | second pass over similar scale | slightly lower final loss than v1, still no reliable quality signal |
| `instruct-v3` | `sft_chat_v3`, `50M` tokens | saner chat formatting | undertrained at only `382` steps, about `0.25` epoch |
| `instruct-v4` | `sft_chat_v4`, `300M` tokens | added held-out eval split and prompt-suite sampling | better process, but outputs remained repetitive, hallucinatory, and weakly grounded |
| `instruct-v5` | `sft_chat_v5`, `180M` train budget | narrower dataset, shorter answers, stronger filtering, checkpoint report summaries | cleaner surface style in places, but still not genuinely correct or dependable |

## Key local findings

### 1. The eval harness overrates bad generations

Evidence:

- base model heuristic score: `0.8479`
- v5 best heuristic score: `0.8664` at step `200` and again at step `800`
- both still produced obviously bad outputs

Examples from stored generations:

- sky explanation answers mention reflection, storms, meteors, or generic color talk instead of Rayleigh scattering
- Ada Lovelace answers hallucinate actresses, Oscars, fictional films, and wrong biographies
- debugging prompts collapse into code fragments, HTML, or nonsense workflows
- rewrite prompts often pass because they are short, not because they are good

Lesson:

- a 9-prompt heuristic suite is useful as a smoke test
- it is not strong enough to choose checkpoints or declare success

### 2. v4 and v5 improved process more than they improved capability

v4 was a process improvement:

- real held-out split
- regular eval cadence
- stored generations for checkpoint review
- longer training budget

That was correct and should be kept.

But v4 generations still showed:

- repetition
- template drift
- code and markup intrusions
- factual hallucination
- poor task grounding

v5 narrowed the problem aggressively:

- assistant cap reduced from `384` tokens to `192`
- max messages reduced from `8` to `6`
- programming/code/JSON/URL-heavy examples were filtered out
- supervised target length dropped sharply

Dataset summary shift:

- v4 supervised tokens `p50=589`, `p90=769`
- v5 supervised tokens `p50=196`, `p90=353`

That made responses shorter, but not reliably better. It mostly changed style, not competence.

### 3. v5 is also a data reduction story

`sft_chat_v5` loaded `156,253` rows, not `195,316`.

Why:

- shards `001`, `002`, `003` are full `50M` token shards
- shard `004` is only a `10M` token shard

So v5 was trained on a meaningfully smaller and narrower SFT corpus. That may have helped reduce some bad behaviors, but it also cut diversity and left little room for broad task recovery.

### 4. SFT is trying to compensate for missing pretraining coverage

The base pretraining mix had no explicit code dataset and no deliberate small-model curriculum for math, reasoning, or chat. Then v5 SFT filtered out programming-heavy requests almost entirely.

That means:

- the base model never learned much code or structured tool-use behavior
- the SFT stage explicitly removed many examples that could have improved it

This is coherent if the target is "tiny general chit-chat bot." It is not coherent if the target is "best small assistant we can build."

### 5. The repo already shows the right engineering instincts

Positive takeaways:

- pretraining and SFT scripts are functional
- tokenizer / model vocabulary alignment checks exist
- throughput instrumentation exists
- prompt-based checkpoint reports exist
- the project has already moved from "blind fine-tuning" toward "behavior-aware fine-tuning"

v2 should keep that discipline and strengthen it.

## Lessons learned

1. Better SFT did not rescue a mediocre capability prior. The next big gain is more likely to come from better pretraining data and better eval than from more SFT filtering.
2. Loss is necessary but not sufficient. Human-readable generations need to be treated as first-class metrics.
3. The current heuristic evaluator is too easy to game with short, format-compliant, wrong answers.
4. Removing code, JSON, long answers, and multi-turn complexity from SFT reduced noise, but it also removed useful capability targets.
5. Small models need narrower post-training than large models, but not artificially tiny post-training.
6. Training a single "general assistant" SFT dataset is less effective than using a staged plan: broad capability in pretraining, narrow helpfulness in SFT, optional preference tuning last.
7. A smaller but better-trained model is more attractive than a slightly larger but data-poorer model.

## Strategy for v2

### Recommended model target

Primary recommendation:

- aim for `~360M-400M` parameters

Why:

- it stays in the proven operating range of the current codebase
- it should fit comfortably on DGX Spark with room for practical batch sizes
- it preserves enough capacity for a meaningful live demo
- it avoids paying a large compute tax for a modest parameter bump

Why not jump straight to `450M+`:

- your main bottleneck is likely token budget and data quality, not raw parameter count
- a 10-20% larger model is only worth it if you can also support a larger clean-token budget and longer iteration cycle

My specific recommendation is:

- `384M` as the default v2 target
- keep `450M` as a stretch option only after a throughput pilot

This is an inference from your local run history plus current small-model practice, not a hard theorem.

### Recommended pretraining plan

### Token budget

Target:

- preferred: `8B-12B` tokens
- minimum serious run: `6B-8B` tokens

Rationale:

- Chinchilla-style compute-optimal training suggests token count should scale with model size rather than staying fixed
- modern small-model projects such as SmolLM2 push this idea much further and overtrain small models heavily on very large high-quality corpora
- your single DGX Spark cannot realistically chase SmolLM2-scale token counts, so the right move is to be much more selective about data quality

### Data mix

Recommended starting mix:

- `60-70%` high-quality educational / reference-heavy web text
- `15-20%` general high-quality deduplicated web text
- `5-10%` books / PDFs / long-form expository text
- `5-10%` code and technical text
- `3-5%` math / reasoning-oriented text
- `0-3%` conversational or forum-style data

Concrete implication for SparkNet:

- keep `FineWeb-Edu`
- keep some `DCLM`
- reduce or remove `ELI5` from the main run
- add a small curated code slice instead of leaving code to chance
- consider a late-stage curriculum shift toward code, math, and QA instead of baking too much of that into the full run

### Curriculum

Recommended structure:

1. Stage A: `6B-8B` tokens of general high-quality text
2. Stage B: `1B-2B` tokens with higher proportions of code, math, and reference-style data
3. Stage C: optional `100M-300M` conversational continuation tokens before SFT if the base model still feels too "document-like"

This is closer to how current small-model projects are trained than doing one static mix and hoping SFT fixes the rest.

### Tokenizer and architecture

Recommendation:

- keep a `32k` tokenizer unless you intentionally go much heavier on code or multilingual data
- retrain the tokenizer on the final v2 pretraining corpus, not an older proxy corpus
- keep grouped-query attention
- keep the architecture depth-heavy rather than making it very wide

Context length:

- preferred: `2048` if the throughput pilot is acceptable
- fallback: `1024` pretraining plus later context extension if `2048` is too slow

For a live demo, `2048` is worth trying. For a single-node research cycle, `1024` may still be the better throughput trade.

### SFT plan

Recommendation:

- stop treating SFT as the main source of intelligence
- use SFT to shape tone, instruction following, abstention, and answer structure

For v2 SFT:

- broaden beyond the current v5 ultra-narrow filter regime
- keep short helpful answers as a major target
- reintroduce some structured tasks, rewrites, summaries, clarifications, and lightweight technical help
- do not completely exclude code / JSON / debugging prompts
- prefer small-model-adapted instruct data over generic big-model chat dumps

Suggested public starting points:

- `HuggingFaceTB/smol-smoltalk` as a small-model-friendly SFT base
- selective additions from `HuggingFaceTB/smoltalk`
- optional small preference stage using `openbmb/UltraFeedback` after SFT

### Evaluation plan

This needs the biggest upgrade.

Keep:

- automated loss tracking
- checkpointed generations
- throughput logging

Replace or add:

- a `50-100` prompt eval suite instead of `9-10` prompts
- side-by-side checkpoint review against the base model
- explicit factual, reasoning, rewrite, summarization, planning, and clarification sections
- a human scorecard with `correct / acceptable / wrong / degenerate`
- failure tags such as `hallucination`, `format drift`, `repetition`, `prompt misunderstanding`, `unsafe certainty`

Checkpoint selection rule:

- never choose the best checkpoint by eval loss alone
- never choose the best checkpoint by heuristic average score alone
- choose the checkpoint that wins human review on the fixed prompt set

### Expected feasibility on DGX Spark

Official DGX Spark hardware docs list:

- `128 GB` LPDDR5x unified memory
- `273 GB/s` memory bandwidth
- support for PyTorch and fine-tuning workloads

NVIDIA's official DGX Spark fine-tuning material reports about `13.5k` tokens/sec for one Llama `3.2 3B` full-fine-tuning recipe, and forum users have reported lower real-world numbers. That is not a direct pretraining benchmark for SparkNet, but it is a useful order-of-magnitude reference.

My rough expectation is:

- a `360M-400M` pretraining run should be feasible
- `8B-12B` tokens is ambitious but realistic if you accept a multi-day to multi-week run window
- exact wall clock should be benchmarked with a `50M-100M` token pilot before locking the architecture

### Recommended v2 decision

If I had to lock the plan today, I would choose:

- model: `384M`
- tokenizer: new `32k` tokenizer trained on final v2 corpus
- context length: start with `1024`, test `2048`, keep whichever gives the better quality-per-day trade
- pretraining budget: `10B` tokens target
- data: heavily `FineWeb-Edu` centered, less `ELI5`, add modest code and math slices
- SFT: small-model-oriented instruct mix, much broader than v5, around `50M-150M` supervised tokens
- final model selection: human eval first, automated metrics second

### Immediate next steps

1. Build a v2 pretraining data spec with explicit percentages and dataset choices.
2. Add a stronger eval prompt suite and a simple human review sheet before starting the next long run.
3. Run a DGX Spark throughput pilot on `1024` and `2048` context with a `~360M-400M` draft architecture.
4. Retrain the tokenizer on the final pretraining corpus sample.
5. Prepare a small-model-oriented SFT recipe using `smol-smoltalk` style data instead of only UltraChat/OASST filtering.

## External references

- NVIDIA DGX Spark hardware overview: https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- NVIDIA DGX Spark performance blog: https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/
- NVIDIA DGX Spark fine-tuning forum thread: https://forums.developer.nvidia.com/t/llama-3-2-3b-full-finetuning-much-slower-than-benchmark/353011
- Chinchilla paper summary: https://huggingface.co/papers/2203.15556
- FineWeb-Edu dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- SmolLM2 paper summary: https://huggingface.co/papers/2502.02737
- SmolLM2 360M Instruct model card: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct
- SmolTalk dataset: https://huggingface.co/datasets/HuggingFaceTB/smoltalk
- smol-smoltalk dataset: https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
- UltraFeedback dataset: https://huggingface.co/datasets/openbmb/UltraFeedback
