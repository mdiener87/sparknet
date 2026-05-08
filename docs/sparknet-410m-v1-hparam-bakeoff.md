# SparkNet-410M v1 Hparam Bakeoff

Written: 2026-05-08

This note records the final learning-rate debate before launching the long
SparkNet-410M v1 pretraining run. The goal is simple: spend a small amount of
extra overnight compute to avoid guessing on the most important scalar in a
multi-week run.

## Current State

The original 410M launch config still carried the aggressive candidate:

- `learning_rate = 1.83e-3`
- `warmup_ratio = 0.02`
- `scheduler = cosine_with_min_lr`
- `cosine_min_lr_ratio = 0.1`
- `grad_accum = 32`
- effective batch = `1,048,576` tokens/step

That `1.83e-3` value came from the earlier Phase 2 LR range result:

- estimated ceiling LR: `5.495e-3`
- heuristic center candidate: `ceiling / 3 = 1.83e-3`

The heuristic was plausible, but it was not a production-schedule validation.
The Phase 2 sweep intentionally runs without warmup and maps the instability
boundary. It gives candidates; it does not prove the full-run optimum.

## Existing Evidence

The 410M Phase 3 screen tested:

| LR | Final train loss | Final eval loss | Spikes |
|---:|---:|---:|---|
| `5.5e-4` | `7.0301` | `6.9459` | no |
| `9e-4` | `7.1104` | `7.0145` | no |
| `1.2e-3` | `7.2522` | `7.1714` | no |
| `1.83e-3` | `7.1428` | `7.0805` | no |

This screen favored `5.5e-4`, and the older 400M hparam grid also favored the
same neighborhood over `1.83e-3`.

However, the Phase 3 evidence has a real caveat: each candidate ran for only
`100M` tokens, or `96` optimizer steps at the selected batch size, with plain
cosine decay to zero. That is not representative of the final `~9,538` step
production run, where the scheduler uses a nonzero 10% LR floor. Short cosine
collapse can bias the screen toward lower LRs.

The first production-scheduler canary was run at `5.5e-4`:

- target tokens: `250M`
- steps: `239`
- scheduler: `cosine_with_min_lr`
- warmup ratio: `0.02`
- LR floor: `0.1 * peak_lr`
- final train loss: `5.9159`
- final eval loss: `5.8841`
- post-warmup spike: no

This proves `5.5e-4` is viable and stable. It does not prove that `1.83e-3` or
an intermediate LR would be worse under the same production schedule.

## Codex vs Claude

The small rivalry is useful because the two answers emphasized different
failure modes.

Codex initially leaned hard toward `5.5e-4` because both Phase 3 screens favored
it and the `5.5e-4` canary was clean. That was directionally conservative:
choose the lowest LR that already has strong evidence.

Claude pushed back on methodology: the Phase 3 screen uses a 96-step cosine
schedule that collapses to zero, so it should not override the Phase 2 ceiling
analysis by itself. Claude's position was that `1.83e-3` deserves a direct
production-scheduler canary before being rejected.

The combined view is better than either answer alone:

- `5.5e-4` is the best proven-safe value so far.
- `1.83e-3` remains an unproven but plausible aggressive candidate.
- `9e-4` is the most useful intermediate candidate because it tests whether the
  true optimum moves upward under the production scheduler without jumping all
  the way to the Phase 2 center value.

## Bakeoff Plan

Run two additional 250M-token canaries with the exact same script and settings
as the successful `5.5e-4` canary.

Claude candidate:

```bash
cd /home/mdiener/projects/sparknet
python scripts/sparknet-410m/hparam_lr_canary.py \
  --lr 0.00183 \
  --target-tokens 250000000
```

Codex candidate:

```bash
cd /home/mdiener/projects/sparknet
python scripts/sparknet-410m/hparam_lr_canary.py \
  --lr 0.0009 \
  --target-tokens 250000000
```

Expected runtime is about `10` hours per canary at roughly `7.3k` tok/s, plus
evaluation overhead. This is cheap compared with a multi-week full pretraining
run.

## Decision Rule

Use the existing `5.5e-4` canary as the baseline:

| Result | Decision |
|---|---|
| `1.83e-3` clearly wins and is smooth | Use `1.83e-3` |
| `9e-4` wins or ties | Use `9e-4` |
| Both new canaries are worse than `5.5e-4` by more than `0.02` eval nats | Use `5.5e-4` |
| A candidate is within `0.02` eval nats of the best lower LR | Prefer the lower LR |
| Any candidate shows NaN/inf, post-warmup jump, or sustained grad-norm instability | Reject it |

If `1.83e-3` is only marginally better than the lower LRs, the safer launch
choice is still the lower LR. The full run is long enough that stability and
generalization margin matter more than a tiny early-canary loss edge.

## What Would Change the Launch Config

Before starting the full run, update:

```json
{
  "learning_rate": "<winner>",
  "warmup_ratio": 0.02,
  "scheduler": "cosine_with_min_lr",
  "cosine_min_lr_ratio": 0.1
}
```

The rest of the launch plan remains unchanged unless the canaries reveal a
warmup or stability issue.

