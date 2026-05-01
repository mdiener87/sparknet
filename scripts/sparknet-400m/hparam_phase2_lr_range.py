"""
Phase 2 — LR Range Test: find the learning rate ceiling.

HOW IT WORKS:
  We sweep LR geometrically from lr_min=1e-5 to lr_max=1e-1 over 400 steps,
  using the formula:

      LR(t) = lr_min × (lr_max / lr_min)^(t / T)

  This means LR doubles roughly every ~27 steps. We watch two signals:
    - Training loss: should decline, then flatten, then spike at the ceiling.
    - Gradient norm: should be stable (< 1.0), then start spiking above the ceiling.

  The LR where loss starts consistently rising is the "ceiling". The optimal
  LR for a full training run lives roughly 3–10× below that ceiling.

  Example: ceiling ≈ 3e-3  →  Phase 3 grid: [3e-4, 1e-3, 2e-3]

WHY NO WARMUP:
  We intentionally skip warmup here. The range test is mapping the loss landscape,
  not training to convergence. Warmup would obscure the low-LR signal in the first
  ~50 steps, where we want to see genuine slow-progress behavior, not optimizer ramp-up.

USAGE:
  # Use grad_accum from Phase 1 output (throughput_report.json)
  python hparam_phase2_lr_range.py --grad-accum 32

OUTPUT:
  logs/hparam-phase2-lr-range/
    lr_range_records.json    # step, loss, grad_norm, lr at each log step
    lr_range_analysis.json   # identified ceiling, recommended grid
  checkpoints/hparam-phase2-lr-range/  (no checkpoints saved; just logs)
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hparam_utils import (
    REPO_ROOT,
    build_model,
    build_eval_wikitext,
    load_prepacked,
    load_sparknet_tokenizer,
    resolve_repo_path,
    set_tf32,
    setup_env,
)

LR_MIN = 1e-5
LR_MAX = 1e-1
RANGE_TEST_STEPS = 400
LOG_EVERY_STEPS = 5   # dense logging so we capture the transition clearly
BLOCK_SIZE = 1024
PER_DEVICE_BATCH = 32


# ------------------------------------------------------------------ #
# Custom LR scheduler: geometric (exponential) sweep
# ------------------------------------------------------------------ #

class GeometricLRRangeTrainer(Trainer):
    """
    Trainer subclass that installs a geometric LR sweep instead of the
    standard cosine/linear schedule.

    The optimizer is initialized with learning_rate = LR_MIN (set in
    TrainingArguments). The LambdaLR scheduler multiplies that base by
    a factor that grows geometrically from 1.0 to LR_MAX/LR_MIN,
    producing LR_MIN → LR_MAX over the full training run.
    """

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer

        log_ratio = math.log(LR_MAX / LR_MIN)

        def lr_lambda(step: int) -> float:
            # At step 0 → multiplier 1.0 → effective LR = LR_MIN
            # At step T → multiplier LR_MAX/LR_MIN → effective LR = LR_MAX
            return math.exp(log_ratio * step / max(1, num_training_steps))

        self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
        self._created_lr_scheduler = True
        return self.lr_scheduler


# ------------------------------------------------------------------ #
# Callback: record loss, grad_norm, and LR at each log step
# ------------------------------------------------------------------ #

class LRRangeRecorder(TrainerCallback):
    def __init__(self):
        self.records = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return
        self.records.append({
            "step": state.global_step,
            "loss": logs.get("loss"),
            "grad_norm": logs.get("grad_norm"),
            "lr": logs.get("learning_rate"),
        })


# ------------------------------------------------------------------ #
# Analysis: identify the LR ceiling from the recorded curve
# ------------------------------------------------------------------ #

def analyze_range_test(records: list) -> dict:
    """
    Identify the LR ceiling from the range test records.

    Algorithm:
      1. Apply EMA smoothing (alpha=0.7) — fast enough to catch gradual rises,
         smooth enough to suppress single-step noise.
      2. Find the global minimum of the smoothed loss (the "sweet spot").
      3. Ceiling is the first step after the minimum where EITHER:
           (a) raw loss > min × 2.5  (sharp spike — the classic instability signal)
           (b) smoothed loss > min × 1.30  (slower sustained rise)

    In practice, sharp spikes (condition a) are the dominant signal above the
    ceiling. The smooth condition (b) catches gradual divergence that occurs
    when the LR is slightly too high but hasn't fully exploded yet.

    NOTE: The automated ceiling is a starting estimate. Always confirm by
    looking at the TensorBoard loss-vs-step curve; the visual is authoritative.

    Interpretation:
      - ceiling_lr: LR where training becomes unstable. Do not train here.
      - sweet_spot_lr: LR at the smoothed-loss minimum.
      - Recommended Phase 3 grid: [ceiling/10, ceiling/3, ceiling/1.5]
    """
    data = [(r["step"], r["loss"], r["lr"]) for r in records if r["loss"] is not None and r["lr"] is not None]
    if len(data) < 10:
        return {"error": "Not enough records for analysis"}

    steps, losses, lrs = zip(*data)

    # EMA smoothing — alpha=0.7 balances noise suppression with responsiveness
    alpha = 0.7
    smoothed = [losses[0]]
    for l in losses[1:]:
        smoothed.append(alpha * smoothed[-1] + (1 - alpha) * l)

    min_loss = min(smoothed)
    min_idx = smoothed.index(min_loss)
    sweet_spot_lr = lrs[min_idx]

    # Detect ceiling: raw spike OR sustained smooth rise after the minimum
    ceiling_lr = None
    for i in range(min_idx, len(smoothed)):
        raw_spike   = losses[i]   > min_loss * 2.5   # sharp single-step explosion
        smooth_rise = smoothed[i] > min_loss * 1.30  # gradual sustained rise
        if raw_spike or smooth_rise:
            ceiling_lr = lrs[i]
            break

    if ceiling_lr is None:
        ceiling_lr = lrs[-1]
        warning = (
            f"Loss did not clearly diverge within the sweep range "
            f"({LR_MIN:.0e} → {LR_MAX:.0e}). Ceiling estimate is approximate. "
            f"Consider widening LR_MAX or inspecting the TensorBoard curve manually."
        )
    else:
        warning = None

    return {
        "ceiling_lr": ceiling_lr,
        "sweet_spot_lr": sweet_spot_lr,
        "min_smoothed_loss": min_loss,
        "min_loss_step": steps[min_idx],
        "recommended_grid": {
            "low":    ceiling_lr / 10.0,
            "center": ceiling_lr / 3.0,
            "high":   ceiling_lr / 1.5,
        },
        "warning": warning,
        "total_records": len(records),
    }


def print_analysis(analysis: dict):
    print("\n" + "=" * 60)
    print("LR RANGE TEST ANALYSIS")
    print("=" * 60)

    if "error" in analysis:
        print(f"Analysis failed: {analysis['error']}")
        return

    if analysis.get("warning"):
        print(f"⚠  {analysis['warning']}")

    print(f"\nSmoothest loss point (sweet spot):")
    print(f"  LR = {analysis['sweet_spot_lr']:.2e}   (step {analysis['min_loss_step']})")
    print(f"  Smoothed loss = {analysis['min_smoothed_loss']:.4f}")

    print(f"\nEstimated LR ceiling (where loss diverges):")
    print(f"  LR = {analysis['ceiling_lr']:.2e}")

    print(f"\nRecommended Phase 3 grid:")
    g = analysis["recommended_grid"]
    print(f"  low    = {g['low']:.2e}   (ceiling / 10)")
    print(f"  center = {g['center']:.2e}   (ceiling / 3)")
    print(f"  high   = {g['high']:.2e}   (ceiling / 1.5)")

    print()
    print(f"Next step:")
    print(f"  python hparam_phase3_lr_grid.py --ceiling-lr {analysis['ceiling_lr']:.2e} --grad-accum <N>")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grad-accum", type=int, default=32,
                        help="Effective batch multiplier (from Phase 1). Default: 32")
    parser.add_argument("--train-root", default="datasets/sparknet-v6-pretrain")
    parser.add_argument("--tokenizer-path", default="./tokenizer-v6")
    parser.add_argument("--steps", type=int, default=RANGE_TEST_STEPS,
                        help="Total LR sweep steps. 400 is standard.")
    parser.add_argument("--limit-shards", type=int, default=4,
                        help="Limit dataset shards loaded (default: 4)")
    args = parser.parse_args()

    setup_env()
    set_tf32(True)
    set_seed(42)

    tok = load_sparknet_tokenizer(resolve_repo_path(args.tokenizer_path))
    train_ds = load_prepacked(resolve_repo_path(args.train_root), limit_shards=args.limit_shards)
    # Range test uses no eval — we're watching training dynamics only
    eval_ds = None

    model = build_model(vocab_size=tok.vocab_size, block_size=BLOCK_SIZE)

    tokens_per_step = BLOCK_SIZE * PER_DEVICE_BATCH * args.grad_accum
    print(f"\nLR sweep: {LR_MIN:.0e} → {LR_MAX:.0e} over {args.steps} steps")
    print(f"Tokens/step: {tokens_per_step:,}   Total: {tokens_per_step * args.steps / 1e6:.0f}M tokens")

    run_name = "hparam-phase2-lr-range"
    run_dir = str(REPO_ROOT / "checkpoints" / run_name)
    log_dir = str(REPO_ROOT / "logs" / run_name)

    train_args = TrainingArguments(
        output_dir=run_dir,
        bf16=True,

        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=args.grad_accum,

        # learning_rate = LR_MIN; our custom scheduler drives it up to LR_MAX
        learning_rate=LR_MIN,
        weight_decay=0.1,
        warmup_ratio=0.0,       # intentionally no warmup — see module docstring
        max_grad_norm=1.0,

        max_steps=args.steps,

        logging_dir=log_dir,
        logging_steps=LOG_EVERY_STEPS,

        eval_strategy="no",

        save_strategy="no",     # short run — no checkpoints needed

        optim="adamw_torch_fused",
        report_to=["tensorboard"],
        remove_unused_columns=False,

        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
    )

    recorder = LRRangeRecorder()

    trainer = GeometricLRRangeTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        callbacks=[recorder],
    )

    print(f"\nStarting LR range test ...")
    trainer.train()

    # ------------------------------------------------------------------ #
    # Save records + analysis
    # ------------------------------------------------------------------ #
    out_dir = REPO_ROOT / "logs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    records_path = out_dir / "lr_range_records.json"
    with open(records_path, "w") as f:
        json.dump(recorder.records, f, indent=2)
    print(f"\nRecords saved: {records_path}  ({len(recorder.records)} entries)")

    analysis = analyze_range_test(recorder.records)
    analysis_path = out_dir / "lr_range_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print_analysis(analysis)
    print(f"\nFull analysis: {analysis_path}")
    print(f"TensorBoard:   tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
