from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import torch
from transformers import set_seed

try:
    from lm_eval import evaluator
except ImportError as exc:  # pragma: no cover - guidance for missing dependency
    raise SystemExit(
        "lm-eval-harness is required for this script. Install with `pip install lm-eval`."
    ) from exc


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
MODEL_CONFIGS = [
    {"label": "SparkNet 70M v5", "path": "checkpoints/sparknet-70m-v5"},
    {"label": "GPT-2", "path": "gpt2"},
    {"label": "CodeLion GPT-2 70M", "path": "codelion/gpt-2-70m"},
]

TASKS = [
    "hellaswag",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "mmlu",
    "truthfulqa_mc2",
    "winogrande",
]

LM_EVAL_BATCH_SIZE = 8
LM_EVAL_LIMIT = None  # set to an integer to debug quickly with partial datasets
SEED = 42
OUTPUT_DIR = Path("eval")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def build_model_args(model_cfg: Dict[str, Any]) -> str:
    """Compose lm-eval model_args string per model configuration."""
    args: List[str] = [
        f"pretrained={model_cfg['path']}",
        f"tokenizer={model_cfg.get('tokenizer', model_cfg['path'])}",
    ]

    if model_cfg.get("trust_remote_code"):
        args.append("trust_remote_code=True")

    if DEVICE == "cuda":
        args.append("dtype=float16")
        args.append("device_map=auto")
    else:
        args.append("dtype=float32")

    return ",".join(args)


def format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (int,)):
        return str(value)
    return str(value)


def summarize_results(label: str, model_path: str, run_results: Dict[str, Any]) -> str:
    lines = [f"\n=== {label} ({model_path}) ==="]
    task_results = run_results.get("results", {})

    for task_name in TASKS:
        metrics = task_results.get(task_name)
        if not metrics:
            continue
        formatted = ", ".join(
            f"{metric}: {format_metric_value(val)}" for metric, val in sorted(metrics.items())
        )
        lines.append(f"{task_name}: {formatted}")

    aggregate = run_results.get("aggregate")
    if aggregate:
        aggregate_str = ", ".join(
            f"{metric}: {format_metric_value(val)}" for metric, val in sorted(aggregate.items())
        )
        lines.append(f"Aggregate: {aggregate_str}")

    return "\n".join(lines)


def run_eval_for_model(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\nRunning lm-eval harness for {model_cfg['label']} ({model_cfg['path']}) on {DEVICE} ...")
    model_args = build_model_args(model_cfg)

    return evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=TASKS,
        batch_size=LM_EVAL_BATCH_SIZE,
        limit=LM_EVAL_LIMIT,
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"eval_benchmarks_{timestamp}.txt"

    header = [
        f"Benchmark run: {datetime.now().isoformat()}",
        f"Device: {DEVICE}",
        f"Tasks: {', '.join(TASKS)}",
        f"Batch size: {LM_EVAL_BATCH_SIZE}",
        f"Limit: {LM_EVAL_LIMIT if LM_EVAL_LIMIT is not None else 'full dataset'}",
    ]

    results_log = ["\n".join(header)]
    print(results_log[0])
    print(f"Writing detailed results to {output_file}\n")

    for model_cfg in MODEL_CONFIGS:
        eval_results = run_eval_for_model(model_cfg)
        summary_text = summarize_results(model_cfg["label"], model_cfg["path"], eval_results)
        print(summary_text)
        results_log.append(summary_text)

    with output_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(results_log))

    print(f"\nSaved benchmark comparison to {output_file}")


if __name__ == "__main__":
    main()
