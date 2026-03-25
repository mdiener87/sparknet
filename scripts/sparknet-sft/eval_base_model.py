#!/usr/bin/env python3
"""
eval_base_model.py

Run a raw base SparkNet model against the v5 evaluation prompt suite and score
its outputs with the same heuristic rubric used during SFT v5 training.
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from train_sft_v5 import (
    build_prompt,
    evaluate_response,
    load_prompt_suite,
    load_sparknet_tokenizer,
    stop_at_next_role_header,
)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/sparknet-400m-v1")
    parser.add_argument("--tokenizer-path", type=str, default="./tokenizer-v6")
    parser.add_argument(
        "--prompts-path",
        type=str,
        default="configs/sparknet-400m/sft_eval_prompts_v5.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/sparknet-400m-v1-base-eval",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-context", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_all_seeds(args.seed)

    prompts = load_prompt_suite(args.prompts_path)
    tokenizer = load_sparknet_tokenizer(args.tokenizer_path, padding_side="right")
    device = resolve_device(args.device)

    torch_dtype = None
    if device == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    do_sample = args.temperature > 0
    prompt_reports: List[Dict[str, object]] = []
    sample_records: List[Dict[str, object]] = []

    print(f"[Eval] model={args.model_path}")
    print(f"[Eval] prompts={len(prompts)} from {args.prompts_path}")
    print(
        f"[Harmony] tok_vocab={len(tokenizer)} emb_vocab={model.get_input_embeddings().weight.shape[0]} "
        f"eos={tokenizer.eos_token_id} pad={tokenizer.pad_token_id} bos={tokenizer.bos_token_id}"
    )

    with torch.no_grad():
        for prompt in prompts:
            prompt_text = build_prompt(prompt["messages"])  # type: ignore[index]
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_context - args.max_new_tokens,
            ).to(device)
            input_len = inputs["input_ids"].shape[1]

            gen_kwargs = dict(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            if do_sample:
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = args.top_p
                gen_kwargs["top_k"] = args.top_k

            output_ids = model.generate(**gen_kwargs)[0]
            new_tokens = output_ids[input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            text = stop_at_next_role_header(text)

            report = evaluate_response(prompt, text)
            prompt_reports.append(report)
            sample_records.append({"prompt_id": str(prompt["id"]), "response": text})
            print(f"[Eval] {prompt['id']} score={report['score']:.2f}")
            print(text)
            print()

    bucket_scores: Dict[str, List[float]] = defaultdict(list)
    for report in prompt_reports:
        bucket_scores[str(report["bucket"])].append(float(report["score"]))

    bucket_summary = {
        bucket: round(sum(scores) / max(1, len(scores)), 4) for bucket, scores in sorted(bucket_scores.items())
    }
    average_score = round(sum(float(report["score"]) for report in prompt_reports) / max(1, len(prompt_reports)), 4)

    passed_checks = 0
    total_checks = 0
    for report in prompt_reports:
        for check in report["checks"]:
            total_checks += 1
            if check["passed"]:
                passed_checks += 1

    summary = {
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path,
        "prompts_path": args.prompts_path,
        "prompt_count": len(prompts),
        "average_score": average_score,
        "bucket_scores": bucket_summary,
        "checks_passed": passed_checks,
        "checks_total": total_checks,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "generation_config": {
            "max_context": args.max_context,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "seed": args.seed,
        },
    }

    doc = dict(summary)
    doc["prompts"] = prompt_reports
    doc["bucket_prompt_counts"] = dict(sorted(Counter(str(prompt.get("bucket", "general")) for prompt in prompts).items()))

    report_path = Path(args.output_dir) / "base_eval_report.json"
    samples_path = Path(args.output_dir) / "base_eval_samples.jsonl"
    summary_path = Path(args.output_dir) / "base_eval_summary.json"

    with open(report_path, "w") as handle:
        json.dump(doc, handle, indent=2)
    with open(samples_path, "w") as handle:
        for record in sample_records:
            handle.write(json.dumps(record) + "\n")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[Eval] average_score={average_score:.4f}")
    print(f"[Eval] bucket_scores={json.dumps(bucket_summary, sort_keys=True)}")
    print(f"[Eval] wrote {report_path}")
    print(f"[Eval] wrote {samples_path}")
    print(f"[Eval] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
