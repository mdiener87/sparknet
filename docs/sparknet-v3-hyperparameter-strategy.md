# SparkNet v3: Hyperparameter Strategy

Status: planning — drafted 2026-04-30

## Why this document exists

v1 and v2 set hyperparameters manually, inheriting values from prior runs without a principled anchor. This document captures a concrete plan to derive better values before locking the v3 training run, based on compute-optimal scaling laws and lightweight pre-run experiments.

The expected payoff: more efficient use of every token in the training budget, without paying any additional compute cost.

---

## Key insight: compute-optimal anchoring

Three hyperparameters dominate training quality and are currently set by intuition:

- **Token budget** — anchored by Chinchilla scaling
- **Learning rate** — anchored by μP width transfer
- **Batch size** — coupled to learning rate; must be set first

### Token budget (Chinchilla)

The 6ND rule approximates training FLOPs: `C ≈ 6 × N × D`

Chinchilla-optimal token count for a given model size N: `D_opt ≈ 20 × N`

For a ~360-400M parameter model this gives ~7-8B tokens as the compute-optimal point. Training beyond this ("overtraining") improves inference quality at the cost of training efficiency — SmolLM2 360M pushes to 100-200× N for exactly this reason.

**v3 recommendation:** 10B tokens is a reasonable minimum. If the DGX Spark throughput pilot shows headroom, push toward 15-20B. Data quality matters more than raw token count past the Chinchilla knee.

### Learning rate and μP transfer

In a standard transformer, the forward-pass magnitude through a hidden layer scales with `d_model` (fan-in). The μP transfer rule corrects for this:

```
LR_target = LR_ref × (d_ref / d_target)
```

This means: tune LR on a small proxy model (same depth, reduced width), then divide by the width ratio to get the full-model LR.

**SparkNet history:**
- 70m (d=512): `LR = 1e-4`
- 400m (d=1024): `LR = 1.5e-4`

The 400m LR is *higher* than the 70m LR, which is the opposite of what μP predicts. Both values are likely conservative. SmolLM2 360M uses `3e-4`; nanoGPT 124M uses `6e-4`. The prior for v3 is that the optimal LR sits somewhere in `[2e-4, 5e-4]` — the experiments below will confirm this.

### Batch size

LR and batch size are coupled via the linear or square-root scaling rules:

- **Linear scaling**: if tokens/step doubles, LR doubles
- **Square-root scaling** (safer at large batch): if tokens/step doubles, LR × √2

Current 400m batch: `32 × 16 × 1024 = 524K tokens/step` — below the ~1-2M sweet spot common in the literature. Increasing grad_accum is the cheapest path.

**Lock batch size before running any LR experiments.** Optimizing LR at one batch size and then changing the batch invalidates the result.

---

## Experiment plan

Run these in order. Each phase is cheap relative to a full training run.

### Phase 1 — Decide v3 batch size

Determine the batch target based on memory budget and throughput. A good starting point is `grad_accum = 32` (keeping `per_device_train_batch_size = 32`, `block_size = 1024`), giving `~1M tokens/step`.

Run a short throughput test to confirm this fits and measure tokens/sec. Lock this value before continuing.

### Phase 2 — LR range test (~400 steps)

Sweep LR exponentially from `1e-5` to `1e-1` over 400 steps using the v3 architecture and batch size. Log loss and grad norm at each step.

```
LR_step = 1e-5 × (1e-1 / 1e-5)^(step / 400)
```

Plot loss vs. LR. The last stable (non-spiking, still-declining) LR is the ceiling. The optimal is roughly one order of magnitude below the ceiling.

Signals:
- Loss declining smoothly → still in safe range
- Loss flattening or reversing → approaching ceiling
- Spikes → above ceiling

This run costs ~400 steps of compute and gives the search range for Phase 3.

### Phase 3 — 3-point grid (3 × 200M token runs)

Center on `(ceiling / 5)`, sweep one step above and below in log space. Example, if ceiling is `~1e-3`:

| Run | LR |
|---|---|
| low | `1e-4` |
| center | `3e-4` |
| high | `6e-4` |

Evaluate by loss curve shape (not just final value):
- Smooth, steep, sustained descent = good
- Spikes at any point after warmup = LR too high
- Sluggish warmup exit, flat slope = LR too low
- High and unstable gradient norm = LR too hot

200M tokens is enough to see a clear ordering. Convergence is not required.

### Phase 4 — Warmup sensitivity (optional)

If the Phase 3 winner is `≥ 2e-4`, run two short variants with `warmup_ratio = 0.01` and `warmup_ratio = 0.03`. Higher LRs are more sensitive to warmup length. If loss spikes right after warmup ends, extend warmup.

Current v2 used `warmup_ratio = 0.01` for 400m. That is likely fine; this phase is a sanity check only.

---

## What to measure across all experiments

Beyond validation loss, track:

- **Gradient norm** — instability signature; should decrease and stabilize after warmup
- **Loss curve shape** — ordering across runs is more reliable than absolute values at 200M tokens
- **Tokens/sec** — confirm no throughput regression vs. v2

Do not rely on the heuristic prompt evaluator to assess pretraining runs. It was shown in v2 to be unreliable at that stage.

---

## Remaining open questions for v3

The LR experiments above address the most critical hyperparameter. Other areas worth revisiting:

- **Loss function / label smoothing**: standard cross-entropy is the default; label smoothing (`0.1`) can sometimes improve generalization for small models but adds a hyperparameter.
- **LR schedule tail**: cosine decay minimum is currently `0` (or default). A non-zero floor (e.g., `min_lr = 0.1 × max_lr`) sometimes prevents over-decay late in training.
- **Data curriculum**: v2 lessons doc recommends a staged mix (general → code/math uplift → optional conversational). This is independent of the hyperparameter experiments but should be decided alongside them.
- **Context length**: v2 docs suggest piloting `2048` vs `1024` — longer context has a direct throughput cost that affects the effective token budget.

---

## Decision checklist before starting v3 full run

- [ ] Batch size locked and throughput-tested
- [ ] LR range test completed, ceiling identified
- [ ] 3-point grid completed, optimal LR selected
- [ ] Warmup ratio confirmed (or adjusted based on Phase 4)
- [ ] Data curriculum finalized (see `sparknet-400m-v2-lessons-and-strategy.md`)
- [ ] Context length decided (`1024` vs `2048`)
- [ ] Tokenizer retrained on final v3 corpus sample (per v2 recommendation)
