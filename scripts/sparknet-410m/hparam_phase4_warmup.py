"""
Phase 4 — Warmup Sensitivity (Optional): confirm warmup_ratio for the chosen LR.

BACKGROUND:
  Warmup is the initial phase where LR ramps from 0 (or near-0) up to its
  target value. It exists to prevent early instability: on step 0, Adam has
  no momentum history. With a large LR and no momentum, the first few updates
  can be massive and land the model in a bad region that it then has to
  recover from.

  How much warmup you need scales with LR:
    - Low LR (≤ 1e-4): 1% warmup is usually fine
    - Medium LR (2–5e-4): 2–3% is safer
    - High LR (≥ 5e-4): 3–5%, watch for post-warmup spikes

  The diagnostic: if loss spikes immediately AFTER warmup ends (not during),
  warmup was too short — the model didn't build enough momentum to handle the
  full LR without overshooting.

  v2 used warmup_ratio=0.02 for a 12B-token run. At ~19K steps, that's 380
  warmup steps. For a v3 run of similar length, 0.02 is a reasonable baseline.
  This phase just confirms it's not obviously wrong for the chosen LR.

WHAT TO LOOK FOR:
  Post-warmup spike: a sharp loss increase right as warmup completes, that then
  slowly recovers. This means warmup_ratio was too short — use the higher value.

  No post-warmup feature: either value works. Prefer the shorter warmup so the
  model spends more time in the high-LR regime (faster learning).

USAGE:
  python hparam_phase4_warmup.py --lr 3e-4 --grad-accum 32

OUTPUT:
  logs/hparam-410m-phase4-warmup-{ratio}/  — TensorBoard + loss curve JSON
  logs/hparam-410m-phase4-warmup-summary.json  — comparison + recommendation
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
    build_model,
    load_eval_prepacked,
    load_prepacked,
    load_sparknet_tokenizer,
    resolve_repo_path,
    set_tf32,
    setup_env,
)

# Enough tokens to clearly see warmup end and post-warmup behavior
# At 1M tokens/step: 500 steps. At 1% warmup: 5 warmup steps → LR hits target at step 5.
# At 3% warmup: 15 steps. We need to watch at least 50–100 steps post-warmup.
TARGET_TOKENS = 150_000_000   # 150M tokens — warmup instability shows in the first ~5-10% of training
BLOCK_SIZE = 1024
PER_DEVICE_BATCH = 32
LOG_EVERY_STEPS = 10

WARMUP_CANDIDATES = [0.01, 0.03]


class LossCurveRecorder(TrainerCallback):
    def __init__(self):
        self.records = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return
        self.records.append({
            "step": state.global_step,
            "loss": logs["loss"],
            "grad_norm": logs.get("grad_norm"),
            "lr": logs.get("learning_rate"),
        })


def run_one(
    lr: float,
    warmup_ratio: float,
    grad_accum: int,
    target_tokens: int,
    train_ds,
    eval_ds,
    init_state_dict: dict,
    tok,
) -> dict:
    tokens_per_step = BLOCK_SIZE * PER_DEVICE_BATCH * grad_accum
    max_steps = math.ceil(target_tokens / tokens_per_step)
    warmup_steps = int(max_steps * warmup_ratio)

    ratio_tag = f"{int(warmup_ratio * 100):02d}pct"
    run_name = f"hparam-410m-phase4-warmup-{ratio_tag}"
    run_dir = str(REPO_ROOT / "checkpoints" / run_name)
    log_dir = str(REPO_ROOT / "logs" / run_name)

    print(f"\n{'='*60}")
    print(f"Phase 4 run: LR = {lr:.2e}, warmup_ratio = {warmup_ratio} ({warmup_steps} steps)")
    print(f"{'='*60}")

    model = build_model(vocab_size=tok.vocab_size, block_size=BLOCK_SIZE)
    model.load_state_dict(copy.deepcopy(init_state_dict))

    train_args = TrainingArguments(
        output_dir=run_dir,
        bf16=True,

        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=grad_accum,

        learning_rate=lr,
        weight_decay=0.1,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,

        max_steps=max_steps,

        logging_dir=log_dir,
        logging_steps=LOG_EVERY_STEPS,

        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=100,

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

    final_train_loss = recorder.records[-1]["loss"] if recorder.records else None
    post_warmup_spike = _has_post_warmup_spike(recorder.records, warmup_steps)

    out_dir = REPO_ROOT / "logs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = out_dir / "loss_curve.json"
    with open(curve_path, "w") as f:
        json.dump({
            "lr": lr,
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "target_tokens": target_tokens,
            "records": recorder.records,
            "final_train_loss": final_train_loss,
            "post_warmup_spike": post_warmup_spike,
        }, f, indent=2)

    print(f"  Final train loss    : {final_train_loss:.4f}" if final_train_loss else "  N/A")
    print(f"  Post-warmup spike   : {'YES — extend warmup_ratio' if post_warmup_spike else 'no'}")
    print(f"  Loss curve saved    : {curve_path}")

    del model
    torch.cuda.empty_cache()

    return {
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "final_train_loss": final_train_loss,
        "post_warmup_spike": post_warmup_spike,
        "log_dir": log_dir,
    }


def _has_post_warmup_spike(records: list, warmup_steps: int) -> bool:
    """
    Detect a loss spike in the first 30% of post-warmup training.

    Compare loss in the window [warmup_steps, warmup_steps + 30% of remaining]
    against the minimum loss just before warmup ended. A spike is any step
    where loss exceeds that baseline by more than 20%.
    """
    # Split into warmup and post-warmup records by step number
    pre  = [r for r in records if r["step"] <= warmup_steps]
    post = [r for r in records if r["step"] >  warmup_steps]

    if not pre or not post:
        return False

    # Baseline: minimum loss over the last few warmup steps
    # (using min rather than last because the final warmup step might itself be noisy)
    tail = pre[max(0, len(pre) - 5):]
    baseline_losses = [r["loss"] for r in tail if r["loss"] is not None]
    if not baseline_losses:
        return False
    baseline = min(baseline_losses)

    # Check first third of post-warmup window for any value > baseline × 1.2
    check_window = post[:max(1, len(post) // 3)]
    return any(r["loss"] is not None and r["loss"] > baseline * 1.2 for r in check_window)


def print_summary(results: list, lr: float):
    print("\n" + "=" * 60)
    print("PHASE 4 WARMUP SENSITIVITY RESULTS")
    print("=" * 60)
    print(f"\nLR = {lr:.2e}")
    print(f"\n{'warmup_ratio':>14}  {'warmup_steps':>14}  {'final_loss':>12}  {'post-spike':>11}")
    for r in results:
        tl = f"{r['final_train_loss']:.4f}" if r.get("final_train_loss") else "  N/A "
        spike = "YES" if r.get("post_warmup_spike") else " no"
        print(f"{r['warmup_ratio']:>14.2f}  {r['warmup_steps']:>14}  {tl:>12}  {spike:>11}")

    spiked = [r for r in results if r.get("post_warmup_spike")]
    clean = [r for r in results if not r.get("post_warmup_spike")]

    print()
    if not spiked:
        # Both ratios are fine — prefer shorter warmup
        rec = min(results, key=lambda r: r["warmup_ratio"])
        print(f"No post-warmup spikes in either run.")
        print(f"Recommendation: warmup_ratio = {rec['warmup_ratio']} (shorter warmup is fine)")
    else:
        min_spiked = min(r["warmup_ratio"] for r in spiked)
        if clean:
            rec = min(clean, key=lambda r: r["warmup_ratio"])
            print(f"Post-warmup spike detected at warmup_ratio = {min_spiked}.")
            print(f"Recommendation: warmup_ratio = {rec['warmup_ratio']} (prevents early instability)")
        else:
            print(f"Both ratios showed post-warmup spikes.")
            print(f"Consider warmup_ratio = 0.05 for this LR, or choose a slightly lower LR.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lr", type=float, required=True,
                        help="Learning rate from Phase 3 winner (e.g. 3e-4)")
    parser.add_argument("--grad-accum", type=int, default=32,
                        help="grad_accum from Phase 1 (default: 32)")
    parser.add_argument("--warmup-candidates", type=float, nargs="+", default=WARMUP_CANDIDATES,
                        help="warmup_ratio values to test (default: 0.01 0.03)")
    parser.add_argument("--train-root", default="datasets/sparknet-410m-v1-pretrain")
    parser.add_argument("--eval-root", default="datasets/sparknet-410m-v1-pretrain-eval")
    parser.add_argument("--tokenizer-path", default="./tokenizer-v8")
    parser.add_argument("--limit-shards", type=int, default=None)
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS,
                        help=f"Tokens per run (default: {TARGET_TOKENS:,})")
    args = parser.parse_args()

    setup_env()
    set_tf32(True)
    set_seed(42)

    tok = load_sparknet_tokenizer(resolve_repo_path(args.tokenizer_path))
    train_ds = load_prepacked(resolve_repo_path(args.train_root), limit_shards=args.limit_shards)
    eval_ds = load_eval_prepacked(resolve_repo_path(args.eval_root))

    # Shared initialization — same logic as Phase 3
    print("\nInitializing shared model weights ...")
    ref = build_model(vocab_size=tok.vocab_size, block_size=BLOCK_SIZE)
    init_state_dict = {k: v.cpu().clone() for k, v in ref.state_dict().items()}
    del ref
    torch.cuda.empty_cache()

    train_ds = train_ds.shuffle(seed=42)

    results = []
    for wr in sorted(args.warmup_candidates):
        result = run_one(args.lr, wr, args.grad_accum, args.target_tokens, train_ds, eval_ds, init_state_dict, tok)
        results.append(result)

    print_summary(results, args.lr)

    summary_path = REPO_ROOT / "logs" / "hparam-410m-phase4-warmup-summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "lr": args.lr,
            "warmup_candidates": args.warmup_candidates,
            "results": results,
        }, f, indent=2)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
