"""
Phase 3 — 3-Point LR Grid: confirm the optimal learning rate.

WHAT THIS DOES:
  Runs three mini-training sessions (200M tokens each) at different LRs,
  all starting from the same random weights and seeing the same data in the
  same order. The only variable is the learning rate.

  WHY IDENTICAL WEIGHTS AND DATA?
    Random initialization introduces variance. If run A happened to get a
    "lucky" init, we'd attribute LR credit to luck. Using the same init
    removes that noise entirely — the loss difference IS the LR signal.

  WHY 200M TOKENS?
    Full convergence isn't needed. At 200M tokens, the ordering of loss
    curves is already clear and stable. The right LR will descend faster
    and smoother than the wrong ones. Extending to convergence would
    confirm but not change the ranking.

WHAT TO LOOK FOR:
  Good curve:
    - Steep, sustained decline after warmup
    - Stable gradient norm (slowly decreasing)
    - Smooth — no loss spikes after warmup ends

  LR too high:
    - Loss spikes appear mid-training, even after warmup
    - Gradient norm stays elevated or spikes
    - Final training loss may be lower (Adam is aggressive) but eval loss
      will be worse — unstable training generalizes poorly

  LR too low:
    - Very slow initial descent
    - The curve never catches up to the medium-LR run by 200M tokens

USAGE:
  # ceiling-lr comes from Phase 2 analysis output
  python hparam_phase3_lr_grid.py --ceiling-lr 3e-3 --grad-accum 32

  # Or specify LRs manually (e.g., if you already have a hypothesis)
  python hparam_phase3_lr_grid.py --lr-list 1e-4 3e-4 6e-4 --grad-accum 32

OUTPUT (per run):
  logs/hparam-phase3-grid-lr{N}/  — TensorBoard + loss curve JSON
  logs/hparam-phase3-grid-summary.json  — cross-run comparison + recommendation
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch
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
    build_eval_wikitext,
    build_model,
    load_prepacked,
    load_sparknet_tokenizer,
    param_count,
    resolve_repo_path,
    set_tf32,
    setup_env,
)

TARGET_TOKENS = 100_000_000   # 100M tokens per run — enough to see loss curve ordering
BLOCK_SIZE = 1024
PER_DEVICE_BATCH = 32
WARMUP_RATIO = 0.01
LOG_EVERY_STEPS = 20
EVAL_EVERY_STEPS = 100


# ------------------------------------------------------------------ #
# Callback: save loss curve to JSON for cross-run comparison
# ------------------------------------------------------------------ #

class LossCurveRecorder(TrainerCallback):
    """Records (step, train_loss, eval_loss) for later comparison."""

    def __init__(self):
        self.train_records = []
        self.eval_records = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return
        self.train_records.append({
            "step": state.global_step,
            "loss": logs["loss"],
            "grad_norm": logs.get("grad_norm"),
            "lr": logs.get("learning_rate"),
        })

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        self.eval_records.append({
            "step": state.global_step,
            "eval_loss": metrics.get("eval_loss"),
            "perplexity": math.exp(metrics["eval_loss"]) if metrics.get("eval_loss") else None,
        })


# ------------------------------------------------------------------ #
# Single run
# ------------------------------------------------------------------ #

def run_one(
    lr: float,
    grad_accum: int,
    train_ds,
    eval_ds,
    init_state_dict: dict,
    tok,
) -> dict:
    """Train for TARGET_TOKENS with the given lr; return the loss curve."""

    tokens_per_step = BLOCK_SIZE * PER_DEVICE_BATCH * grad_accum
    max_steps = math.ceil(TARGET_TOKENS / tokens_per_step)

    lr_tag = f"{lr:.0e}".replace("-0", "-").replace("+0", "")  # "3e-4"
    run_name = f"hparam-phase3-grid-lr{lr_tag}"
    run_dir = str(REPO_ROOT / "checkpoints" / run_name)
    log_dir = str(REPO_ROOT / "logs" / run_name)

    print(f"\n{'='*60}")
    print(f"Phase 3 run: LR = {lr:.2e}   ({max_steps} steps, {TARGET_TOKENS/1e6:.0f}M tokens)")
    print(f"{'='*60}")

    # Fresh model from the shared initialization
    model = build_model(vocab_size=tok.vocab_size, block_size=BLOCK_SIZE)
    model.load_state_dict(copy.deepcopy(init_state_dict))

    train_args = TrainingArguments(
        output_dir=run_dir,
        bf16=True,

        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=grad_accum,

        learning_rate=lr,
        weight_decay=0.1,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,

        max_steps=max_steps,

        logging_dir=log_dir,
        logging_steps=LOG_EVERY_STEPS,

        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=EVAL_EVERY_STEPS,

        save_strategy="no",

        optim="adamw_torch_fused",
        report_to=["tensorboard"],
        remove_unused_columns=False,

        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
    )

    recorder = LossCurveRecorder()

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        callbacks=[recorder],
    )

    trainer.train()

    # Final eval
    final_eval_loss = None
    if eval_ds is not None:
        metrics = trainer.evaluate()
        final_eval_loss = metrics.get("eval_loss")

    final_train_loss = recorder.train_records[-1]["loss"] if recorder.train_records else None

    out_dir = REPO_ROOT / "logs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = out_dir / "loss_curve.json"
    with open(curve_path, "w") as f:
        json.dump({
            "lr": lr,
            "grad_accum": grad_accum,
            "max_steps": max_steps,
            "target_tokens": TARGET_TOKENS,
            "train_records": recorder.train_records,
            "eval_records": recorder.eval_records,
            "final_train_loss": final_train_loss,
            "final_eval_loss": final_eval_loss,
        }, f, indent=2)

    print(f"  Final train loss : {final_train_loss:.4f}" if final_train_loss else "  Final train loss : N/A")
    print(f"  Final eval loss  : {final_eval_loss:.4f}" if final_eval_loss else "  Final eval loss  : N/A (no eval)")
    print(f"  Loss curve saved : {curve_path}")

    del model
    torch.cuda.empty_cache()

    return {
        "lr": lr,
        "run_name": run_name,
        "max_steps": max_steps,
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
        "has_spikes": _detect_spikes(recorder.train_records),
        "log_dir": log_dir,
    }


def _detect_spikes(records: list, threshold: float = 1.5) -> bool:
    """
    Returns True if any training step's loss exceeds the running minimum
    by more than threshold × (indicating a spike after warmup).
    We skip the first 10% of steps to ignore warmup instability.
    """
    if len(records) < 20:
        return False
    skip = max(1, len(records) // 10)
    losses = [r["loss"] for r in records[skip:] if r["loss"] is not None]
    if not losses:
        return False
    min_so_far = losses[0]
    for l in losses[1:]:
        if l > min_so_far * threshold:
            return True
        if l < min_so_far:
            min_so_far = l
    return False


# ------------------------------------------------------------------ #
# Summary
# ------------------------------------------------------------------ #

def print_summary(results: list):
    print("\n" + "=" * 60)
    print("PHASE 3 GRID RESULTS")
    print("=" * 60)

    # Sort by eval_loss (primary) or train_loss (fallback)
    def sort_key(r):
        return r.get("final_eval_loss") or r.get("final_train_loss") or 999.0

    ranked = sorted(results, key=sort_key)

    print(f"\n{'LR':>10}  {'Train loss':>12}  {'Eval loss':>12}  {'Spikes':>7}")
    for r in ranked:
        tl = f"{r['final_train_loss']:.4f}" if r.get("final_train_loss") else "  N/A "
        el = f"{r['final_eval_loss']:.4f}" if r.get("final_eval_loss") else "  N/A "
        spike = "YES" if r.get("has_spikes") else " no"
        print(f"{r['lr']:>10.2e}  {tl:>12}  {el:>12}  {spike:>7}")

    winner = ranked[0]
    print(f"\nBest LR: {winner['lr']:.2e}")

    if winner.get("has_spikes"):
        print("  Note: winner had some spikes. Consider the next candidate if you")
        print("  plan to push to a high token budget where instability compounds.")

    if winner.get("final_eval_loss"):
        second = ranked[1] if len(ranked) > 1 else None
        if second and second.get("final_eval_loss"):
            margin = second["final_eval_loss"] - winner["final_eval_loss"]
            if margin < 0.01:
                print(f"  Note: margin vs. runner-up is only {margin:.4f} nats.")
                print("  Either LR is likely fine. Prefer the lower one for stability.")

    print(f"\nNext step (Phase 4, optional warmup sensitivity):")
    print(f"  python hparam_phase4_warmup.py --lr {winner['lr']:.2e} --grad-accum <N>")

    print(f"\nFor v3 full run — use this LR in pretrain_v3.json:")
    print(f'  "learning_rate": {winner["lr"]}')

    return winner


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    lr_group = parser.add_mutually_exclusive_group(required=True)
    lr_group.add_argument(
        "--ceiling-lr",
        type=float,
        help="LR ceiling from Phase 2. Grid becomes [ceiling/10, ceiling/3, ceiling/1.5].",
    )
    lr_group.add_argument(
        "--lr-list",
        type=float,
        nargs="+",
        help="Explicit LR list (overrides --ceiling-lr).",
    )

    parser.add_argument("--grad-accum", type=int, default=32,
                        help="grad_accum from Phase 1 (default: 32)")
    parser.add_argument("--train-root", default="datasets/sparknet-v6-pretrain")
    parser.add_argument("--tokenizer-path", default="./tokenizer-v6")
    parser.add_argument("--limit-shards", type=int, default=None,
                        help="Limit shards loaded (default: all)")
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS,
                        help=f"Tokens per run (default: {TARGET_TOKENS:,})")
    args = parser.parse_args()

    # Derive grid
    if args.lr_list:
        grid_lrs = sorted(args.lr_list)
    else:
        c = args.ceiling_lr
        grid_lrs = sorted([c / 10.0, c / 3.0, c / 1.5])

    print("Phase 3 LR grid:")
    for lr in grid_lrs:
        print(f"  {lr:.2e}")

    setup_env()
    set_tf32(True)
    set_seed(42)

    tok = load_sparknet_tokenizer(resolve_repo_path(args.tokenizer_path))
    train_ds = load_prepacked(resolve_repo_path(args.train_root), limit_shards=args.limit_shards)
    eval_ds = build_eval_wikitext(tok, BLOCK_SIZE)

    # Build one reference model and save its initial weights.
    # All three runs load from this so the only variable is LR.
    print("\nInitializing reference model (shared across all runs) ...")
    ref_model = build_model(vocab_size=tok.vocab_size, block_size=BLOCK_SIZE)
    print(f"Parameters: {param_count(ref_model) / 1e6:.1f}M")
    init_state_dict = {k: v.cpu().clone() for k, v in ref_model.state_dict().items()}
    del ref_model
    torch.cuda.empty_cache()

    # Optionally shuffle train_ds with a fixed seed for reproducibility across runs
    train_ds = train_ds.shuffle(seed=42)

    results = []
    for lr in grid_lrs:
        result = run_one(lr, args.grad_accum, train_ds, eval_ds, init_state_dict, tok)
        results.append(result)

    winner = print_summary(results)

    # Save summary
    summary_path = REPO_ROOT / "logs" / "hparam-phase3-grid-summary.json"
    with open(summary_path, "w") as f:
        json.dump({"grid": grid_lrs, "results": results, "recommended_lr": winner["lr"]}, f, indent=2)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
