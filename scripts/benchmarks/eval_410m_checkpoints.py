#!/usr/bin/env python3
"""Evaluate all SparkNet 410M v1 checkpoints.

This combines the existing lm-eval benchmark, perplexity, and sample generation
flows into one checkpoint comparison run.
"""

import argparse
import gc
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

torch = None
load_dataset = None
AutoModelForCausalLM = None
AutoTokenizer = None
set_seed = None
evaluator = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "sparknet-410m-v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval"
DEFAULT_TASKS = ["hellaswag", "piqa", "arc_easy", "arc_challenge"]
DEFAULT_TOKENS_PER_STEP = 1_048_576
RECOMMENDED_EXTRA_TASKS = ["winogrande", "openbookqa", "boolq"]
DEFAULT_PROMPTS = [
    "The meaning of life is",
    "In the future, AI will",
    "A good software engineer knows that",
    "Write a Python function that",
    "If I have 12 apples and give away 5,",
    "Explain the difference between supervised and unsupervised learning:",
    "The capital of France is",
]
DATASET_CONFIGS = [
    {
        "label": "WikiText-2 Raw v1 (validation)",
        "dataset_name": "wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "split": "validation",
        "text_field": "text",
    },
]


def ensure_runtime_dependencies(needs_lm_eval: bool, needs_hf: bool) -> None:
    global torch
    global load_dataset
    global AutoModelForCausalLM
    global AutoTokenizer
    global set_seed
    global evaluator

    if needs_hf or needs_lm_eval:
        try:
            import torch as torch_module
            from datasets import load_dataset as load_dataset_fn
            from transformers import AutoModelForCausalLM as auto_model_cls
            from transformers import AutoTokenizer as auto_tokenizer_cls
            from transformers import set_seed as set_seed_fn
        except ImportError as exc:
            raise SystemExit(
                "torch, datasets, and transformers are required for evaluation. "
                "Activate the SparkNet training environment or install requirements.txt."
            ) from exc

        torch = torch_module
        load_dataset = load_dataset_fn
        AutoModelForCausalLM = auto_model_cls
        AutoTokenizer = auto_tokenizer_cls
        set_seed = set_seed_fn

    if needs_lm_eval:
        try:
            from lm_eval import evaluator as evaluator_module
        except ImportError as exc:
            raise SystemExit(
                "lm-eval-harness is required for benchmark tasks. Install with `pip install lm-eval`."
            ) from exc
        evaluator = evaluator_module


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists() or not candidate.exists():
        return repo_candidate
    return candidate.resolve()


def checkpoint_sort_key(path: Path) -> Tuple[int, int, str]:
    name = path.name
    if name == DEFAULT_CHECKPOINT_ROOT.name:
        return (2, 2**63 - 1, name)

    token_match = re.fullmatch(r"checkpoint-(\d+)B-tokens", name)
    if token_match:
        return (1, int(token_match.group(1)) * 1_000_000_000, name)

    step_match = re.fullmatch(r"checkpoint-(\d+)", name)
    if step_match:
        return (1, int(step_match.group(1)) * DEFAULT_TOKENS_PER_STEP, name)

    return (2, 0, name)


def is_model_dir(path: Path) -> bool:
    return (path / "config.json").is_file() and any(
        (path / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )


def discover_checkpoints(root: Path, include_root: bool, pattern: Optional[str]) -> List[Path]:
    checkpoints: List[Path] = []
    if include_root and is_model_dir(root):
        checkpoints.append(root)

    for child in root.iterdir():
        if child.is_dir() and is_model_dir(child):
            checkpoints.append(child)

    if pattern:
        checkpoints = [path for path in checkpoints if pattern in path.name]

    return sorted(checkpoints, key=checkpoint_sort_key)


def load_checkpoint_metadata(path: Path) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"name": path.name, "path": str(path)}

    eval_loss_path = path / ".eval_loss"
    if eval_loss_path.is_file():
        try:
            metadata["training_eval_loss"] = float(eval_loss_path.read_text().strip())
        except ValueError:
            metadata["training_eval_loss_raw"] = eval_loss_path.read_text().strip()

    trainer_state_path = path / "trainer_state.json"
    if trainer_state_path.is_file():
        with trainer_state_path.open("r", encoding="utf-8") as handle:
            trainer_state = json.load(handle)
        metadata["global_step"] = trainer_state.get("global_step")
        metadata["epoch"] = trainer_state.get("epoch")

    run_summary_path = path / "run_summary.json"
    if run_summary_path.is_file():
        with run_summary_path.open("r", encoding="utf-8") as handle:
            run_summary = json.load(handle)
        metadata.setdefault("global_step", run_summary.get("global_step"))
        metadata["finished_at"] = run_summary.get("finished_at")

    return metadata


def build_lm_eval_model_args(
    checkpoint_path: Path,
    tokenizer_path: Path,
    device: str,
    dtype: str,
    trust_remote_code: bool,
) -> str:
    args = [
        f"pretrained={checkpoint_path}",
        f"tokenizer={tokenizer_path}",
    ]
    if trust_remote_code:
        args.append("trust_remote_code=True")
    if device.startswith("cuda"):
        args.append(f"dtype={dtype}")
        args.append(f"device={device}")
    else:
        args.append("dtype=float32")
        args.append("device=cpu")
    return ",".join(args)


def run_lm_eval(
    checkpoint_path: Path,
    tokenizer_path: Path,
    tasks: Sequence[str],
    batch_size: int,
    limit: Optional[int],
    device: str,
    dtype: str,
    trust_remote_code: bool,
) -> Dict[str, Any]:
    print(f"  lm-eval tasks: {', '.join(tasks)}")
    model_args = build_lm_eval_model_args(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    return evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=list(tasks),
        batch_size=batch_size,
        limit=limit,
    )


def prepare_dataset_samples(cfg: Dict[str, Any], sample_limit: int, seed: int) -> List[str]:
    load_args = [cfg["dataset_name"]]
    if cfg.get("dataset_config"):
        load_args.append(cfg["dataset_config"])

    dataset = load_dataset(*load_args, split=cfg.get("split", "validation"))
    dataset = dataset.shuffle(seed=seed)

    text_field = cfg.get("text_field", "text")
    samples = []
    for record in dataset:
        text = str(record.get(text_field, "")).strip()
        if text:
            samples.append(text)
        if len(samples) >= sample_limit:
            break
    return samples


def load_model_and_tokenizer(
    checkpoint_path: Path,
    tokenizer_path: Path,
    device: str,
    dtype: str,
    trust_remote_code: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float32
    if device.startswith("cuda") and dtype in {"float16", "fp16"}:
        torch_dtype = torch.float16
    elif device.startswith("cuda") and dtype in {"bfloat16", "bf16"}:
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.eval()
    return tokenizer, model


def evaluate_model_on_texts(
    model,
    tokenizer,
    texts: Sequence[str],
    batch_size: int,
    max_sequence_length: int,
    device: str,
) -> Dict[str, float]:
    losses = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            list(batch),
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_sequence_length,
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        losses.append(out.loss.detach().cpu().item())

    avg_loss = sum(losses) / max(1, len(losses))
    return {"loss": avg_loss, "perplexity": math.exp(avg_loss)}


def run_perplexity(
    model,
    tokenizer,
    dataset_samples: Dict[str, List[str]],
    batch_size: int,
    max_sequence_length: int,
    device: str,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for dataset_label, samples in dataset_samples.items():
        print(f"  perplexity: {dataset_label} ({len(samples)} samples)")
        metrics = evaluate_model_on_texts(
            model=model,
            tokenizer=tokenizer,
            texts=samples,
            batch_size=batch_size,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        metrics["samples"] = float(len(samples))
        results[dataset_label] = metrics
    return results


def load_prompts(prompt_path: Optional[Path]) -> List[str]:
    if not prompt_path:
        return DEFAULT_PROMPTS

    with prompt_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Prompt file must contain a JSON list: {prompt_path}")

    prompts = []
    for idx, item in enumerate(data):
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            prompts.append(item["prompt"])
        else:
            raise ValueError(
                f"Prompt item {idx} must be a string or object with a string 'prompt' field."
            )
    return prompts


def run_generations(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> List[Dict[str, Any]]:
    records = []
    do_sample = temperature > 0

    for idx, prompt in enumerate(prompts, start=1):
        print(f"  generation: prompt_{idx:02d}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        completion = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
        records.append(
            {
                "prompt_id": f"prompt_{idx:02d}",
                "prompt": prompt,
                "text": full_text,
                "completion": completion,
            }
        )

    return records


def cleanup_model(model: Any = None) -> None:
    del model
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def metric_value(results: Dict[str, Any], task: str) -> Optional[float]:
    task_result = results.get("results", {}).get(task, {})
    for key in ("acc_norm,none", "acc,none", "exact_match,none"):
        value = task_result.get(key)
        if isinstance(value, (float, int)):
            return float(value)
    for value in task_result.values():
        if isinstance(value, (float, int)):
            return float(value)
    return None


def write_markdown_summary(
    path: Path,
    records: Sequence[Dict[str, Any]],
    tasks: Sequence[str],
    include_perplexity: bool,
) -> None:
    lines = [
        f"# SparkNet 410M v1 Checkpoint Evaluation",
        "",
        f"Run finished: {datetime.now().isoformat()}",
        "",
        f"Recommended optional follow-up lm-eval tasks: {', '.join(RECOMMENDED_EXTRA_TASKS)}",
        "",
    ]

    header = ["checkpoint", "training_eval_loss"]
    header.extend(tasks)
    if include_perplexity:
        header.extend(["wikitext_loss", "wikitext_ppl"])

    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for record in records:
        row = [
            str(record["checkpoint"]["name"]),
            format_optional_float(record["checkpoint"].get("training_eval_loss")),
        ]
        benchmark = record.get("benchmark") or {}
        row.extend(format_optional_float(metric_value(benchmark, task)) for task in tasks)

        if include_perplexity:
            ppl = record.get("perplexity") or {}
            wiki = ppl.get("WikiText-2 Raw v1 (validation)", {})
            row.extend(
                [
                    format_optional_float(wiki.get("loss")),
                    format_optional_float(wiki.get("perplexity")),
                ]
            )

        lines.append("| " + " | ".join(row) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_optional_float(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SparkNet 410M v1 checkpoints with lm-eval, perplexity, and generations."
    )
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--tokenizer", default=None, help="Defaults to --checkpoint-root.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--checkpoint-filter", default=None, help="Substring filter for checkpoint names.")
    parser.add_argument("--no-include-root", action="store_true", help="Skip the final model in checkpoint root.")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--lm-eval-batch-size", type=int, default=1)
    parser.add_argument("--lm-eval-limit", type=int, default=200, help="Examples per task for test runs. Use 0 for full datasets.")
    parser.add_argument("--perplexity-samples", type=int, default=250)
    parser.add_argument("--perplexity-batch-size", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Defaults to cuda if available, otherwise cpu.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "fp16", "bfloat16", "bf16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-lm-eval", action="store_true")
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--skip-generations", action="store_true")
    parser.add_argument("--full", action="store_true", help="Run full lm-eval datasets and 2000 perplexity samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.full:
        args.lm_eval_limit = None
        args.perplexity_samples = 2000
    elif args.lm_eval_limit == 0:
        args.lm_eval_limit = None
    needs_lm_eval = not args.skip_lm_eval
    needs_hf = not args.skip_perplexity or not args.skip_generations
    ensure_runtime_dependencies(needs_lm_eval=needs_lm_eval, needs_hf=needs_hf)

    if args.device is None:
        args.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    if set_seed is not None:
        set_seed(args.seed)

    checkpoint_root = resolve_path(args.checkpoint_root)
    tokenizer_path = resolve_path(args.tokenizer) if args.tokenizer else checkpoint_root
    output_dir = resolve_path(args.output_dir)
    prompt_path = resolve_path(args.prompt_file) if args.prompt_file else None
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"sparknet_410m_v1_checkpoint_eval_{timestamp}.jsonl"
    generation_path = output_dir / f"sparknet_410m_v1_generations_{timestamp}.jsonl"
    summary_path = output_dir / f"sparknet_410m_v1_checkpoint_eval_{timestamp}.md"

    checkpoints = discover_checkpoints(
        root=checkpoint_root,
        include_root=not args.no_include_root,
        pattern=args.checkpoint_filter,
    )
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under {checkpoint_root}")

    print(f"Checkpoint root: {checkpoint_root}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Device: {args.device}")
    print(f"Checkpoints: {len(checkpoints)}")
    print(f"lm-eval limit: {args.lm_eval_limit if args.lm_eval_limit is not None else 'full datasets'}")
    print(f"lm-eval batch size: {args.lm_eval_batch_size}")
    print(f"Perplexity samples: {args.perplexity_samples}")
    print(f"Results: {result_path}")

    dataset_samples: Dict[str, List[str]] = {}
    if not args.skip_perplexity:
        for cfg in DATASET_CONFIGS:
            samples = prepare_dataset_samples(cfg, args.perplexity_samples, args.seed)
            dataset_samples[cfg["label"]] = samples
            print(f"Loaded dataset '{cfg['label']}' with {len(samples)} usable samples.")

    prompts = [] if args.skip_generations else load_prompts(prompt_path)
    result_records: List[Dict[str, Any]] = []
    generation_records: List[Dict[str, Any]] = []

    for checkpoint_path in checkpoints:
        metadata = load_checkpoint_metadata(checkpoint_path)
        print(f"\n=== {metadata['name']} ===")

        record: Dict[str, Any] = {
            "run_at": datetime.now().isoformat(),
            "checkpoint": metadata,
            "config": {
                "tokenizer": str(tokenizer_path),
                "tasks": tasks,
                "lm_eval_limit": args.lm_eval_limit,
                "perplexity_samples": args.perplexity_samples,
                "max_sequence_length": args.max_sequence_length,
                "generation_prompts": len(prompts),
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "device": args.device,
                "dtype": args.dtype,
            },
        }

        if not args.skip_lm_eval:
            record["benchmark"] = run_lm_eval(
                checkpoint_path=checkpoint_path,
                tokenizer_path=tokenizer_path,
                tasks=tasks,
                batch_size=args.lm_eval_batch_size,
                limit=args.lm_eval_limit,
                device=args.device,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
            )
            cleanup_model()

        needs_hf_model = not args.skip_perplexity or not args.skip_generations
        model = None
        tokenizer = None
        if needs_hf_model:
            tokenizer, model = load_model_and_tokenizer(
                checkpoint_path=checkpoint_path,
                tokenizer_path=tokenizer_path,
                device=args.device,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
            )

        if not args.skip_perplexity and model is not None and tokenizer is not None:
            record["perplexity"] = run_perplexity(
                model=model,
                tokenizer=tokenizer,
                dataset_samples=dataset_samples,
                batch_size=args.perplexity_batch_size,
                max_sequence_length=args.max_sequence_length,
                device=args.device,
            )

        if not args.skip_generations and model is not None and tokenizer is not None:
            generations = run_generations(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            for generation in generations:
                generation_records.append(
                    {
                        "checkpoint": metadata,
                        **generation,
                    }
                )
            record["generations"] = generations

        cleanup_model(model)
        result_records.append(record)
        write_jsonl(result_path, result_records)
        if generation_records:
            write_jsonl(generation_path, generation_records)

    write_markdown_summary(
        path=summary_path,
        records=result_records,
        tasks=tasks,
        include_perplexity=not args.skip_perplexity,
    )

    print(f"\nSaved JSONL results to {result_path}")
    if generation_records:
        print(f"Saved generation JSONL to {generation_path}")
    print(f"Saved markdown summary to {summary_path}")


if __name__ == "__main__":
    main()
