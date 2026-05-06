"""
Phase 1 — Throughput Test: lock the effective batch size for v3.

WHY THIS COMES FIRST:
  LR and batch size are mathematically coupled. The Adam optimizer's update
  magnitude scales with the gradient estimate, which changes as you average
  over more samples. Concretely, if you double tokens/step, you roughly need
  to sqrt-scale your LR (square-root scaling rule) or linear-scale it (linear
  scaling rule). Either way: find the optimal LR at one batch size and then
  change the batch — and you need to rescale.

  We lock grad_accum first so that the Phase 2/3 LR experiments are valid for
  the actual v3 training run.

WHAT THIS MEASURES:
  For each grad_accum candidate, we run a raw training loop (no Trainer overhead),
  measure tokens/sec and peak VRAM, then print a recommendation.

  per_device_train_batch_size and block_size are fixed at v2 values (32, 1024).
  Only grad_accum varies, which changes effective tokens/step without changing
  per-step VRAM significantly.

USAGE:
  python hparam_phase1_throughput.py
  python hparam_phase1_throughput.py --candidates 16 32 64

OUTPUT:
  logs/hparam-410m-phase1/throughput_report.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hparam_utils import (
    REPO_ROOT,
    build_model,
    load_prepacked,
    load_sparknet_tokenizer,
    param_count,
    resolve_repo_path,
    set_tf32,
    setup_env,
)

# Fixed across all experiments — matches v2 production config
PER_DEVICE_BATCH = 32
BLOCK_SIZE = 1024

# How many steps to burn before we start the clock.
# CUDA compiles kernels lazily; the first N steps are slower and not representative.
WARMUP_STEPS = 10

# Steps used for the actual measurement window.
# 30 macro steps is plenty — throughput stabilizes within the first few.
# (At grad_accum=128, 100 macro steps would take ~2 hrs just for the benchmark.)
MEASURE_STEPS = 30

DEFAULT_CANDIDATES = [16, 32, 64, 128]


def measure_one(
    model: torch.nn.Module,
    data_loader: DataLoader,
    grad_accum: int,
    device: torch.device,
) -> dict:
    """
    Run a minimal training loop and return throughput stats for this grad_accum.

    We use a raw loop (not HF Trainer) so we can precisely CUDA-sync around the
    measurement window. The Trainer adds overhead that would blur the comparison.
    """
    tokens_per_step = BLOCK_SIZE * PER_DEVICE_BATCH * grad_accum

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    model.train()

    loader_iter = iter(data_loader)

    def next_batch():
        nonlocal loader_iter
        try:
            return next(loader_iter)
        except StopIteration:
            loader_iter = iter(data_loader)
            return next(loader_iter)

    t0 = None
    elapsed = None
    optimizer.zero_grad()
    macro_step = 0

    total_micro = (WARMUP_STEPS + MEASURE_STEPS) * grad_accum

    for micro_step in range(total_micro):
        batch = {k: v.to(device) for k, v in next_batch().items()}

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss / grad_accum

        loss.backward()

        if (micro_step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            macro_step += 1

            if macro_step == WARMUP_STEPS:
                # Start precise measurement after warmup
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                t0 = time.perf_counter()

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    tok_per_s = (MEASURE_STEPS * tokens_per_step) / elapsed
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9

    return {
        "grad_accum": grad_accum,
        "tokens_per_step": tokens_per_step,
        "tok_per_s": round(tok_per_s),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "elapsed_s": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-root", default="datasets/sparknet-410m-v1-pretrain")
    parser.add_argument("--tokenizer-path", default="./tokenizer-v8")
    parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=DEFAULT_CANDIDATES,
        help="grad_accum values to benchmark (default: 16 32 64 128)",
    )
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=2,
        help="Load only N dataset shards — 2 is plenty for a throughput test",
    )
    args = parser.parse_args()

    setup_env()
    set_tf32(True)
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device found. Numbers will not reflect GPU throughput.")
    else:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU : {props.name}")
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")

    tok = load_sparknet_tokenizer(resolve_repo_path(args.tokenizer_path))
    train_ds = load_prepacked(resolve_repo_path(args.train_root), limit_shards=args.limit_shards)

    data_loader = DataLoader(
        train_ds,
        batch_size=PER_DEVICE_BATCH,
        shuffle=True,
        collate_fn=default_data_collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    results = []
    for ga in args.candidates:
        effective_tokens = ga * PER_DEVICE_BATCH * BLOCK_SIZE
        print(f"\n{'='*60}")
        print(f"grad_accum = {ga}   ({effective_tokens:,} tokens/step)")
        print(f"{'='*60}")

        model = build_model(vocab_size=tok.vocab_size, block_size=BLOCK_SIZE).to(device)
        n_params = param_count(model)
        if ga == args.candidates[0]:
            print(f"Model params: {n_params / 1e6:.1f}M")

        try:
            result = measure_one(model, data_loader, ga, device)
            results.append(result)
            print(f"  tok/s     : {result['tok_per_s']:,}")
            print(f"  peak VRAM : {result['peak_vram_gb']:.2f} GB")
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM — grad_accum={ga} exceeds VRAM")
            results.append({"grad_accum": ga, "error": "OOM"})
        finally:
            del model
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Summary + recommendation
    # ------------------------------------------------------------------ #
    valid = [r for r in results if "error" not in r]

    print(f"\n{'='*60}")
    print("THROUGHPUT SUMMARY")
    print(f"{'='*60}")
    print(f"{'grad_accum':>12}  {'tokens/step':>14}  {'tok/s':>10}  {'VRAM GB':>8}")
    for r in results:
        if "error" in r:
            print(f"{r['grad_accum']:>12}  {'— OOM —':>14}")
        else:
            print(
                f"{r['grad_accum']:>12}  {r['tokens_per_step']:>14,}"
                f"  {r['tok_per_s']:>10,}  {r['peak_vram_gb']:>8.2f}"
            )

    recommended_ga = None
    if valid:
        # Choose the largest grad_accum whose throughput is within 5% of the fastest.
        # Beyond the "knee", doubling the batch costs throughput without adding value.
        best_tps = max(r["tok_per_s"] for r in valid)
        for r in reversed(valid):
            if r["tok_per_s"] >= best_tps * 0.95:
                recommended_ga = r["grad_accum"]
                break

        recommended = next(r for r in valid if r["grad_accum"] == recommended_ga)
        print(f"\nRecommendation: grad_accum = {recommended_ga}")
        print(f"  Effective tokens/step : {recommended['tokens_per_step']:,}")
        print(f"  Throughput            : {recommended['tok_per_s']:,} tok/s")
        print()
        print("  Lock this value. Changing batch size after LR experiments")
        print("  requires re-running Phase 2 and Phase 3.")
    else:
        print("\nAll candidates hit OOM. Try reducing per_device_train_batch_size.")

    # ------------------------------------------------------------------ #
    # Save report
    # ------------------------------------------------------------------ #
    out_dir = REPO_ROOT / "logs" / "hparam-410m-phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "per_device_batch": PER_DEVICE_BATCH,
        "block_size": BLOCK_SIZE,
        "warmup_steps": WARMUP_STEPS,
        "measure_steps": MEASURE_STEPS,
        "results": results,
        "recommended_grad_accum": recommended_ga,
    }
    out_path = out_dir / "throughput_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
