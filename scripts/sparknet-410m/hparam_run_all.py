"""
Orchestrator: runs SparkNet-410M LR screens and canary in sequence.

Default flow:
  Phase 3 screen → 4 explicit LRs at 100M tokens each
  Canary        → 250M tokens at the selected LR with production scheduler

Optional legacy flow:
  Phase 2 → finds LR ceiling → writes lr_range_analysis.json
  Phase 3 → runs 3-point grid at that ceiling
  Phase 4 → tests warmup sensitivity at Phase 3 winner LR

Legacy phases are kept for comparison with the 400m harness, but the 410m
run-readiness gate is the explicit LR screen plus canary.

USAGE:
  # Full run (recommended — run in tmux or screen so it survives disconnect)
  python3 hparam_run_all.py --grad-accum 32

  # Include the legacy Phase 2 range test before the screen
  python3 hparam_run_all.py --grad-accum 32 --start-from 2 --run-phase4

  # Resume from Phase 3 screens
  python3 hparam_run_all.py --grad-accum 32 --start-from 3

  # Run only the canary with an explicit LR
  python3 hparam_run_all.py --grad-accum 32 --start-from 5 --winner-lr 9e-4
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def banner(title: str):
    line = "=" * 70
    print(f"\n{line}")
    print(f"  {title}")
    print(f"  {ts()}")
    print(f"{line}\n")


def run_phase(cmd: list, phase_name: str):
    """
    Run a phase script and stream its output in real-time.
    Exits the orchestrator if the phase fails.
    """
    banner(f"STARTING: {phase_name}")
    print(f"Command: {' '.join(str(c) for c in cmd)}\n")

    # No stdout/stderr capture — let output stream through to the terminal
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))

    if result.returncode != 0:
        print(f"\n[{ts()}] ERROR: {phase_name} exited with code {result.returncode}")
        print("Orchestrator stopping. Fix the error and resume with --start-from.")
        sys.exit(result.returncode)

    print(f"\n[{ts()}] {phase_name} complete.")


def add_optional_arg(cmd: list, name: str, value):
    if value is not None:
        cmd += [name, str(value)]
    return cmd


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def read_ceiling_lr() -> float:
    path = REPO_ROOT / "logs" / "hparam-410m-phase2-lr-range" / "lr_range_analysis.json"
    if not path.exists():
        print(f"ERROR: Phase 2 output not found at {path}")
        print("Run Phase 2 first, or pass --ceiling-lr explicitly.")
        sys.exit(1)
    data = load_json(path)
    return data["ceiling_lr"]


def read_winner_lr() -> float:
    path = REPO_ROOT / "logs" / "hparam-410m-phase3-grid-summary.json"
    if not path.exists():
        print(f"ERROR: Phase 3 output not found at {path}")
        print("Run Phase 3 first, or pass --winner-lr explicitly.")
        sys.exit(1)
    data = load_json(path)
    return data["recommended_lr"]


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--grad-accum", type=int, default=32,
        help="Locked grad_accum from Phase 1 (default: 32)",
    )
    parser.add_argument(
        "--train-root", default="datasets/sparknet-410m-v1-pretrain",
    )
    parser.add_argument(
        "--eval-root", default="datasets/sparknet-410m-v1-pretrain-eval",
    )
    parser.add_argument(
        "--tokenizer-path", default="./tokenizer-v8",
    )
    parser.add_argument(
        "--start-from", type=int, choices=[2, 3, 4, 5], default=3,
        help="Resume from this phase (3=LR screens by default, 5=canary only)",
    )
    parser.add_argument(
        "--ceiling-lr", type=float, default=None,
        help="Override ceiling LR instead of reading from Phase 2 output",
    )
    parser.add_argument(
        "--winner-lr", type=float, default=None,
        help="Override winner LR instead of reading from Phase 3 output",
    )
    parser.add_argument(
        "--skip-phase4", action="store_true",
        dest="skip_phase4",
        default=True,
        help="Skip legacy warmup sensitivity phase (default)",
    )
    parser.add_argument(
        "--run-phase4", action="store_false",
        dest="skip_phase4",
        help="Run legacy Phase 4 warmup sensitivity after the LR screen",
    )
    parser.add_argument(
        "--skip-canary", action="store_true",
        help="Skip the 250M production-scheduler canary",
    )
    parser.add_argument(
        "--screen-lrs",
        type=float,
        nargs="+",
        default=[5.5e-4, 9e-4, 1.2e-3, 1.83e-3],
        help="Explicit LR screen values for Phase 3",
    )
    parser.add_argument(
        "--limit-shards", type=int, default=None,
        help="Limit dataset shards (useful for a quick smoke test)",
    )
    parser.add_argument(
        "--phase3-target-tokens",
        type=int,
        default=None,
        help="Override Phase 3 screen tokens per LR (for smoke tests)",
    )
    parser.add_argument(
        "--phase4-target-tokens",
        type=int,
        default=None,
        help="Override legacy Phase 4 tokens per run (for smoke tests)",
    )
    parser.add_argument(
        "--canary-target-tokens",
        type=int,
        default=None,
        help="Override canary target tokens (for smoke tests)",
    )
    args = parser.parse_args()

    python = sys.executable

    # Args passed to every phase script
    common = [
        "--grad-accum", str(args.grad_accum),
        "--train-root", args.train_root,
        "--tokenizer-path", args.tokenizer_path,
    ]
    if args.limit_shards:
        common += ["--limit-shards", str(args.limit_shards)]
    eval_common = common + ["--eval-root", args.eval_root]

    ceiling_lr = args.ceiling_lr
    winner_lr  = args.winner_lr

    banner("SparkNet v3 Hyperparameter Experiment Suite")
    print(f"  grad_accum    : {args.grad_accum}  ({args.grad_accum * 32 * 1024:,} tokens/step)")
    print(f"  starting from : Phase {args.start_from}")
    print(f"  skip Phase 4  : {args.skip_phase4}")
    print(f"  skip canary   : {args.skip_canary}")
    if ceiling_lr:
        print(f"  ceiling_lr    : {ceiling_lr:.2e}  (manual override)")
    if winner_lr:
        print(f"  winner_lr     : {winner_lr:.2e}  (manual override)")

    # ---------------------------------------------------------------- #
    # Phase 2 — LR range test
    # ---------------------------------------------------------------- #
    if args.start_from <= 2:
        run_phase(
            [python, str(SCRIPT_DIR / "hparam_phase2_lr_range.py")] + common,
            "Phase 2 — LR Range Test",
        )
        ceiling_lr = read_ceiling_lr()
        print(f"  ceiling_lr from Phase 2: {ceiling_lr:.2e}")

    # ---------------------------------------------------------------- #
    # Phase 3 — 3-point LR grid
    # ---------------------------------------------------------------- #
    if args.start_from <= 3:
        if ceiling_lr is not None:
            phase3_cmd = [python, str(SCRIPT_DIR / "hparam_phase3_lr_grid.py"),
                          "--ceiling-lr", str(ceiling_lr)] + eval_common
        else:
            phase3_cmd = [python, str(SCRIPT_DIR / "hparam_phase3_lr_grid.py"),
                          "--lr-list"] + [str(v) for v in args.screen_lrs] + eval_common
        add_optional_arg(phase3_cmd, "--target-tokens", args.phase3_target_tokens)
        run_phase(phase3_cmd, "Phase 3 — LR Screen Grid")
        winner_lr = read_winner_lr()
        print(f"  winner_lr from Phase 3: {winner_lr:.2e}")

    # ---------------------------------------------------------------- #
    # Phase 4 — warmup sensitivity (optional)
    # ---------------------------------------------------------------- #
    if not args.skip_phase4:
        if winner_lr is None:
            winner_lr = read_winner_lr()
            print(f"  winner_lr from saved Phase 3 output: {winner_lr:.2e}")

        if winner_lr < 2e-4:
            print(f"\nPhase 4 skipped: winner LR {winner_lr:.2e} is below 2e-4.")
            print("At low LR, warmup sensitivity is negligible. Keep warmup_ratio=0.02.")
        else:
            phase4_cmd = [python, str(SCRIPT_DIR / "hparam_phase4_warmup.py"),
                          "--lr", str(winner_lr)] + eval_common
            add_optional_arg(phase4_cmd, "--target-tokens", args.phase4_target_tokens)
            run_phase(phase4_cmd, "Phase 4 — Warmup Sensitivity")

    # ---------------------------------------------------------------- #
    # Canary — production scheduler confirmation
    # ---------------------------------------------------------------- #
    if not args.skip_canary and args.start_from <= 5:
        if winner_lr is None:
            winner_lr = read_winner_lr()
            print(f"  winner_lr from saved Phase 3 output: {winner_lr:.2e}")
        canary_cmd = [
            python, str(SCRIPT_DIR / "hparam_lr_canary.py"),
            "--lr", str(winner_lr),
            "--train-root", args.train_root,
            "--eval-root", args.eval_root,
            "--tokenizer-path", args.tokenizer_path,
        ]
        if args.limit_shards:
            canary_cmd += ["--limit-shards", str(args.limit_shards)]
        add_optional_arg(canary_cmd, "--target-tokens", args.canary_target_tokens)
        run_phase(canary_cmd, "Canary — 250M Production-Scheduler LR Check")

    # ---------------------------------------------------------------- #
    # Final summary
    # ---------------------------------------------------------------- #
    banner("ALL PHASES COMPLETE")

    # Try to load Phase 4 recommendation if it ran
    warmup_rec = None
    phase4_summary = REPO_ROOT / "logs" / "hparam-410m-phase4-warmup-summary.json"
    if phase4_summary.exists():
        p4 = load_json(phase4_summary)
        # Find the cleanest (non-spiking) warmup with the smallest ratio
        clean = [r for r in p4["results"] if not r.get("post_warmup_spike")]
        if clean:
            warmup_rec = min(clean, key=lambda r: r["warmup_ratio"])["warmup_ratio"]

    print("Results:")
    if ceiling_lr:
        print(f"  LR ceiling (Phase 2)  : {ceiling_lr:.2e}")
    if winner_lr:
        print(f"  Optimal LR (Phase 3)  : {winner_lr:.2e}")
    if warmup_rec:
        print(f"  Warmup ratio (Phase 4): {warmup_rec}")

    print()
    print("Update pretrain_v3_base.json with:")
    if winner_lr:
        print(f'  "learning_rate": {winner_lr},')
    if warmup_rec:
        print(f'  "warmup_ratio": {warmup_rec}')
    elif not args.skip_phase4:
        print('  "warmup_ratio": <check Phase 4 logs>')
    else:
        print('  "warmup_ratio": 0.02  (Phase 4 skipped; 0.02 is a safe default)')

    print()
    print(f"TensorBoard (all runs):")
    print(f"  tensorboard --logdir {REPO_ROOT / 'logs'}")


if __name__ == "__main__":
    main()
