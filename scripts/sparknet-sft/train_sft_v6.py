#!/usr/bin/env python3
"""
train_sft_v6.py

SparkNet-400M SFT trainer, v6.

Changes from v5:
- Targets the v2 pretrain base (checkpoint-12000 of sparknet-400m-v2-12b).
- Uses the sft_chat_v6 dataset (smol-smoltalk primary, OASST secondary).
- Longer sample_max_new_tokens (128 vs 96) to match the relaxed response cap.
- Improved evaluate_response():
    - no_role_leak: catches responses that start with a role header (format
      corruption that always scored well on the old surface checks).
    - topic_adjacent: for non-social prompts, verifies the response contains
      at least one content word from the user's question. This is the check
      that would have caught the "Click the Next button" / Seattle itinerary
      failure from v5.
- New special_checks: is_greeting, is_empathetic, is_persona,
  expresses_uncertainty.
- Expanded eval prompt suite (sft_eval_prompts_v6.json, 23 prompts) with
  social, persona, multi-turn, and factual buckets.
"""

import argparse
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional

import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


ROLE_PREFIX = {
    "system": "### System:\n",
    "user": "### User:\n",
    "assistant": "### Assistant:\n",
}

WORD_RE = re.compile(r"[A-Za-z']+")
LIST_ITEM_RE = re.compile(r"(?m)^\s*(?:[-*]|\d+[.)])\s+")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
CODE_RE = re.compile(
    r"```|<(?:(?:!DOCTYPE)|html|body|script)\b|\bimport\s+[A-Za-z0-9_.]+|\bdef\s+\w+\s*\(|\bpublic\s+class\b|#include\s*<",
    re.IGNORECASE,
)

TOPIC_STOP_WORDS = frozenset({
    "that", "this", "with", "have", "your", "from", "what", "when", "where",
    "which", "there", "their", "about", "would", "could", "should", "does",
    "just", "some", "more", "they", "them", "then", "will", "want", "tell",
    "give", "make", "help", "here", "into", "been", "very", "also", "such",
    "like", "know", "think", "good", "well", "these", "those", "other",
})


@dataclass
class RunConfig:
    run_name: str = "sparknet-400m-v2-instruct-v1"
    seed: int = 42

    model_path: str = "checkpoints/sparknet-400m-v2-12b/checkpoint-12000"
    tokenizer_path: str = "./tokenizer-v6"
    train_root: str = "datasets/sft_chat_v6"
    block_size: int = 1024

    bf16: bool = True
    gradient_checkpointing: bool = False

    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    grad_accum: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"
    max_grad_norm: float = 1.0

    target_tokens: int = 450_000_000

    eval_fraction: float = 0.02
    eval_steps: int = 100

    sample_prompts_path: str = "configs/sparknet-400m/sft_eval_prompts_v6.json"
    sample_max_new_tokens: int = 128
    sample_temperature: float = 0.0
    sample_top_p: float = 0.9
    sample_top_k: int = 0
    sample_repetition_penalty: float = 1.12
    sample_no_repeat_ngram_size: int = 4

    output_dir: str = "checkpoints/sparknet-400m-v2-instruct-v1"
    logging_steps: int = 25
    save_steps: int = 100
    save_total_limit: Optional[int] = 6

    dataloader_num_workers: int = 8
    dataloader_prefetch_factor: int = 4

    limit_shards: Optional[int] = None
    resume_from: Optional[str] = None


def set_tf32(enable: bool = True):
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32" if enable else "ieee"
    except Exception:
        pass
    try:
        torch.backends.cudnn.conv.fp32_precision = "tf32" if enable else "ieee"
    except Exception:
        pass


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_complete_shard_dir(path: Path) -> bool:
    return (path / "dataset_info.json").exists() and (path / "state.json").exists()


def list_shards(train_root: str) -> List[str]:
    root = Path(train_root)
    complete: List[str] = []
    skipped: List[str] = []
    for path in sorted(root.glob("shard-*")):
        if not path.is_dir():
            continue
        if is_complete_shard_dir(path):
            complete.append(str(path))
        else:
            skipped.append(str(path))

    if skipped:
        print(f"[Data] Skipping {len(skipped)} incomplete shard(s):")
        for shard in skipped:
            print(f"[Data]   - {shard}")

    if not complete:
        raise FileNotFoundError(
            f"No complete shard-* datasets found under: {train_root}. "
            "If a previous dataset build was interrupted, delete the incomplete shard dirs and rebuild."
        )
    return complete


def load_dataset_shards(train_root: str, limit: Optional[int] = None) -> Dataset:
    shards = list_shards(train_root)
    if limit is not None:
        shards = shards[:limit]
    print(f"[Data] Loading {len(shards)} shard(s) from {train_root}")
    datasets = [Dataset.load_from_disk(shard_dir) for shard_dir in shards]
    ds = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    print(f"[Data] rows={len(ds):,}")
    return ds


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    out = Path(output_dir)
    if not out.exists():
        return None
    checkpoints = sorted(out.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
    return str(checkpoints[-1]) if checkpoints else None


def check_bf16_or_die(enable_bf16: bool):
    if not enable_bf16:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("bf16 requested but CUDA is not available.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested but this GPU does not support bf16.")


def try_log_sdpa_backend():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel  # type: ignore
        _ = (sdpa_kernel, SDPBackend)
        print("[Attn] SDPA available (backend chosen dynamically per call).")
    except Exception:
        print("[Attn] SDPA backend API not available; using default attention behavior.")


def quantile(values: List[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    idx = max(0, min(idx, len(ordered) - 1))
    return int(ordered[idx])


def summarize_rows(ds: Dataset, block_size: int, n: int = 1024) -> Dict[str, int]:
    n = min(n, len(ds))
    if n == 0:
        raise ValueError("Dataset is empty.")

    lengths: List[int] = []
    supervised_counts: List[int] = []
    first_supervised: List[int] = []

    for idx in range(n):
        example = ds[idx]
        if "input_ids" not in example or "labels" not in example:
            raise ValueError(f"Row {idx} missing input_ids/labels keys.")

        input_ids = example["input_ids"]
        labels = example["labels"]
        if len(input_ids) != len(labels):
            raise ValueError(f"Row {idx} length mismatch: input_ids={len(input_ids)} labels={len(labels)}")
        if len(input_ids) > block_size:
            raise ValueError(f"Row {idx} too long: {len(input_ids)} > block_size={block_size}")

        active = [i for i, label in enumerate(labels) if label != -100]
        if active:
            supervised_counts.append(len(active))
            first_supervised.append(active[0])
        lengths.append(len(input_ids))

    if not supervised_counts:
        raise ValueError(
            "No supervised labels found in the validation window. "
            "This usually means the builder masked everything."
        )

    return {
        "rows_checked": n,
        "supervised_rows": len(supervised_counts),
        "seq_len_p50": int(median(lengths)),
        "seq_len_p90": quantile(lengths, 0.90),
        "supervised_tokens_p50": int(median(supervised_counts)),
        "supervised_tokens_p90": quantile(supervised_counts, 0.90),
        "first_supervised_p50": int(median(first_supervised)),
        "first_supervised_p90": quantile(first_supervised, 0.90),
    }


def build_prompt(messages: List[Dict[str, str]]) -> str:
    chunks: List[str] = []
    for message in messages:
        role = message["role"]
        if role not in ROLE_PREFIX:
            raise ValueError(f"Unsupported role in prompt suite: {role}")
        chunks.append(ROLE_PREFIX[role] + message["content"].strip() + "\n")
    chunks.append(ROLE_PREFIX["assistant"])
    return "".join(chunks)


def stop_at_next_role_header(text: str) -> str:
    cut = None
    for marker in ("\n### User:", "\n### System:", "\n### Assistant:"):
        idx = text.find(marker)
        if idx != -1:
            cut = idx if cut is None else min(cut, idx)
    if cut is not None:
        text = text[:cut]
    return text.strip()


def load_prompt_suite(path: str) -> List[Dict[str, object]]:
    with open(path, "r") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"Prompt suite must be a JSON list: {path}")

    prompts: List[Dict[str, object]] = []
    for idx, item in enumerate(raw):
        prompt_id = f"prompt_{idx + 1:02d}"
        if isinstance(item, str):
            prompts.append(
                {
                    "id": prompt_id,
                    "bucket": "general",
                    "messages": [{"role": "user", "content": item.strip()}],
                }
            )
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Prompt entry {idx} must be a string or object.")

        messages = item.get("messages")
        if messages is None:
            user_text = str(item.get("user", "")).strip()
            if not user_text:
                raise ValueError(f"Prompt entry {idx} is missing `messages` or `user`.")
            messages = []
            system_text = str(item.get("system", "")).strip()
            if system_text:
                messages.append({"role": "system", "content": system_text})
            messages.append({"role": "user", "content": user_text})

        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Prompt entry {idx} has no usable messages.")

        norm_messages: List[Dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(f"Prompt entry {idx} contains a non-dict message.")
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role not in ROLE_PREFIX or not content:
                raise ValueError(f"Prompt entry {idx} has an invalid message: {message}")
            norm_messages.append({"role": role, "content": content})

        prompt = {key: value for key, value in item.items() if key != "messages"}
        prompt.setdefault("id", prompt_id)
        prompt.setdefault("bucket", "general")
        prompt["messages"] = norm_messages
        prompts.append(prompt)

    return prompts


def load_sparknet_tokenizer(tokenizer_path: str, padding_side: str = "right"):
    path = Path(tokenizer_path).expanduser()
    model_path = path / "tokenizer.model" if path.is_dir() else path
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")

    tok = LlamaTokenizer(vocab_file=str(model_path), legacy=True)
    tok.padding_side = padding_side
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


class SFTPadCollator:
    def __init__(self, tokenizer, block_size: int):
        self.tok = tokenizer
        self.block_size = int(block_size)
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        max_len = max(len(feature["input_ids"]) for feature in features)
        max_len = min(max_len, self.block_size)

        input_ids = torch.full((batch_size, max_len), fill_value=self.tok.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), fill_value=-100, dtype=torch.long)

        for idx, feature in enumerate(features):
            ids = feature["input_ids"][:max_len]
            lbs = feature["labels"][:max_len]
            n = len(ids)
            input_ids[idx, :n] = torch.tensor(ids, dtype=torch.long)
            attention_mask[idx, :n] = 1
            labels[idx, :n] = torch.tensor(lbs, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def list_item_count(text: str) -> int:
    return len(LIST_ITEM_RE.findall(text))


def repetition_score(text: str) -> float:
    words = [word.lower() for word in WORD_RE.findall(text)]
    if len(words) < 12:
        return 1.0
    trigrams = list(zip(words, words[1:], words[2:]))
    if not trigrams:
        return 1.0
    unique_ratio = len(set(trigrams)) / len(trigrams)
    return max(0.0, min(1.0, (unique_ratio - 0.45) / 0.45))


def _extract_topic_words(messages: List[Dict[str, str]]) -> set:
    words = set()
    for msg in messages:
        if msg.get("role") == "user":
            for w in WORD_RE.findall(msg["content"]):
                if len(w) > 4 and w.lower() not in TOPIC_STOP_WORDS:
                    words.add(w.lower())
    return words


def special_check(name: str, response: str) -> Dict[str, object]:
    lower = response.lower()
    words = word_count(response)

    if name == "is_greeting":
        greeting_signals = ["hi", "hello", "hey", "doing", "good", "great", "fine", "well", "thanks", "nice"]
        passed = any(g in lower for g in greeting_signals) or "?" in response
        return {"name": name, "passed": passed, "detail": "ok" if passed else "no greeting/social response"}

    if name == "is_empathetic":
        empathy_signals = ["sorry", "rough", "tough", "understand", "sounds", "feel", "hope", "better",
                           "hang in", "that's hard", "must be", "that can be"]
        passed = any(e in lower for e in empathy_signals)
        return {"name": name, "passed": passed, "detail": "ok" if passed else "no empathy signals"}

    if name == "is_persona":
        identity_signals = ["spark", "assistant", "model", "ai", "chatbot", "my name", "i'm", "called", "i am"]
        passed = any(w in lower for w in identity_signals)
        return {"name": name, "passed": passed, "detail": "ok" if passed else "no identity reference"}

    if name == "expresses_uncertainty":
        uncertainty_signals = [
            "can't predict", "cannot predict", "uncertain", "financial advisor",
            "consult", "no way to know", "difficult to", "i don't know",
            "hard to say", "not sure", "risky", "no one knows", "impossible to",
            "nobody can", "depends on", "i wouldn't",
        ]
        passed = any(s in lower for s in uncertainty_signals)
        return {"name": name, "passed": passed, "detail": "ok" if passed else "no uncertainty expressed"}

    if name == "vegetarian":
        banned = ["chicken", "beef", "pork", "turkey", "bacon", "ham", "salmon", "tuna", "shrimp", "fish"]
        hit = next((term for term in banned if term in lower), None)
        return {"name": name, "passed": hit is None, "detail": "ok" if hit is None else f"contains {hit}"}

    if name == "ada_lovelace_basic":
        has_name = "ada lovelace" in lower
        has_signal = any(token in lower for token in ["analytical engine", "mathematic", "computer", "algorithm"])
        wrong = next(
            (token for token in ["french actress", "british actress", "nobel", "dragon tattoo", "tolkien"]
             if token in lower),
            None,
        )
        passed = has_name and has_signal and wrong is None
        detail = "ok" if passed else f"name={has_name} signal={has_signal} wrong={wrong or 'none'}"
        return {"name": name, "passed": passed, "detail": detail}

    if name == "uncertainty_behavior":
        has_caution = any(
            phrase in lower
            for phrase in [
                "not sure", "uncertain", "say so", "clarifying", "clarify",
                "i don't know", "should say", "should be honest",
            ]
        )
        overconfident = any(phrase in lower for phrase in ["make something up", "guess confidently", "pretend"])
        passed = has_caution and not overconfident
        detail = "ok" if passed else f"caution={has_caution} overconfident={overconfident}"
        return {"name": name, "passed": passed, "detail": detail}

    if name == "simple_itinerary":
        has_preference = ("coffee" in lower) and any(token in lower for token in ["bookstore", "bookstores", "bookshop"])
        has_walk = any(token in lower for token in ["walk", "walking", "stroll"])
        loop_like = lower.count("seattle") >= 6
        passed = has_preference and has_walk and not loop_like
        detail = "ok" if passed else f"preference={has_preference} walk={has_walk} loop_like={loop_like}"
        return {"name": name, "passed": passed, "detail": detail}

    if name == "short_rewrite":
        bad = any(token in lower for token in ["dear ", "sincerely", "i hope this email finds you well"])
        passed = (not bad) and words <= 45
        detail = "ok" if passed else f"bad_greeting={bad} words={words}"
        return {"name": name, "passed": passed, "detail": detail}

    if name == "brief_debug_plan":
        items = list_item_count(response)
        has_debug_terms = sum(
            token in lower for token in ["reproduce", "logs", "check", "trace", "rollback", "error", "request"]
        ) >= 2
        passed = items >= 3 and has_debug_terms
        detail = "ok" if passed else f"items={items} debug_terms={has_debug_terms}"
        return {"name": name, "passed": passed, "detail": detail}

    if name == "asks_for_clarification":
        asks = "?" in response and any(
            phrase in lower for phrase in ["what", "which", "can you share", "more detail", "what are you trying"]
        )
        return {"name": name, "passed": asks, "detail": "ok" if asks else "no clear clarifying question"}

    if name == "sky_blue_basic":
        passed = "blue" in lower and any(token in lower for token in ["scatter", "scattering", "atmosphere"])
        return {"name": name, "passed": passed, "detail": "ok" if passed else "missing scattering explanation"}

    return {"name": name, "passed": True, "detail": "unknown_check_ignored"}


def evaluate_response(prompt: Dict[str, object], response: str) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []

    def add(name: str, passed: bool, detail: str):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    words = word_count(response)
    add("non_empty", bool(response.strip()), f"words={words}")
    add("no_role_leak", not response.startswith("###"),
        "ok" if not response.startswith("###") else "role header leaked into response")
    add("no_code", not CODE_RE.search(response), "ok" if not CODE_RE.search(response) else "code-like pattern")
    add("no_url", not URL_RE.search(response), "ok" if not URL_RE.search(response) else "url-like pattern")

    rep_score = repetition_score(response)
    add("low_repetition", rep_score >= 0.45, f"score={rep_score:.2f}")

    # For non-social prompts, check that the response addresses the topic at all.
    # This is the check that catches "Click the Next button on the homepage" for
    # a Seattle itinerary question — a pure format pass but a complete non-sequitur.
    bucket = str(prompt.get("bucket", "general"))
    if bucket != "social":
        messages = prompt.get("messages") or []
        topic_words = _extract_topic_words(messages)  # type: ignore[arg-type]
        if topic_words:
            response_lower = response.lower()
            hit = any(w in response_lower for w in topic_words)
            add("topic_adjacent", hit,
                "ok" if hit else f"none of {sorted(topic_words)[:5]} found in response")

    min_words = prompt.get("min_words")
    if min_words is not None:
        min_words = int(min_words)
        add("min_words", words >= min_words, f"{words}>={min_words}")

    max_words = prompt.get("max_words")
    if max_words is not None:
        max_words = int(max_words)
        add("max_words", words <= max_words, f"{words}<={max_words}")

    expect_list_min_items = prompt.get("expect_list_min_items")
    if expect_list_min_items is not None:
        expected = int(expect_list_min_items)
        actual = list_item_count(response)
        add("list_items", actual >= expected, f"{actual}>={expected}")

    for needle in prompt.get("must_include_all", []) or []:
        needle = str(needle)
        add(f"has:{needle}", needle.lower() in response.lower(),
            "present" if needle.lower() in response.lower() else "missing")

    if prompt.get("must_include_any"):
        needles = [str(item).lower() for item in (prompt.get("must_include_any") or [])]
        hit = next((needle for needle in needles if needle in response.lower()), None)
        add("must_include_any", hit is not None, hit or "missing")

    for needle in prompt.get("avoid_substrings", []) or []:
        needle = str(needle)
        add(f"avoid:{needle}", needle.lower() not in response.lower(),
            "ok" if needle.lower() not in response.lower() else "present")

    for check_name in prompt.get("special_checks", []) or []:
        checks.append(special_check(str(check_name), response))

    passed = sum(1 for check in checks if check["passed"])
    score = passed / max(1, len(checks))
    return {
        "id": str(prompt["id"]),
        "bucket": bucket,
        "score": round(score, 4),
        "checks": checks,
        "response": response,
        "word_count": words,
    }


class PerfCallback(TrainerCallback):
    def __init__(self, approx_tokens_per_step: int):
        self.tokens_per_step = approx_tokens_per_step
        self.last_t = None
        self.last_step = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_t = time.time()
        self.last_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.last_t is None or self.last_step is None or state.global_step <= self.last_step:
            return
        now = time.time()
        dt = now - self.last_t
        ds = state.global_step - self.last_step
        if dt <= 0 or ds <= 0:
            return
        tok_per_s = (ds / dt) * self.tokens_per_step
        print(f"[Perf] ~{tok_per_s:,.0f} tok/s (approx) | step={state.global_step:,}")
        self.last_t = now
        self.last_step = state.global_step


class EvalReportCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        prompts: List[Dict[str, object]],
        output_dir: str,
        max_context: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
    ):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.output_dir = output_dir
        self.report_dir = os.path.join(output_dir, "checkpoint_reports")
        self.sample_output_path = os.path.join(output_dir, "sample_generations.jsonl")
        self.summary_output_path = os.path.join(output_dir, "checkpoint_report_summary.jsonl")
        self.max_context = max_context
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.last_logged_step = None
        self.best_score = -1.0
        self.best_step = None

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if model is None or self.last_logged_step == state.global_step:
            return

        os.makedirs(self.report_dir, exist_ok=True)
        device = next(model.parameters()).device
        model.eval()
        do_sample = self.temperature > 0

        print("\n[EvalReport] =====")
        prompt_reports: List[Dict[str, object]] = []
        sample_records: List[Dict[str, object]] = []

        with torch.no_grad():
            for prompt in self.prompts:
                prompt_text = build_prompt(prompt["messages"])  # type: ignore[index]
                inputs = self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_context - self.max_new_tokens,
                ).to(device)
                input_len = inputs["input_ids"].shape[1]

                gen_kwargs = dict(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=do_sample,
                    repetition_penalty=self.repetition_penalty,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                if do_sample:
                    gen_kwargs["temperature"] = self.temperature
                    gen_kwargs["top_p"] = self.top_p
                    gen_kwargs["top_k"] = self.top_k

                output_ids = model.generate(**gen_kwargs)[0]
                new_tokens = output_ids[input_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                text = stop_at_next_role_header(text)

                report = evaluate_response(prompt, text)
                prompt_reports.append(report)
                sample_records.append({"step": state.global_step, "prompt_id": str(prompt["id"]), "response": text})
                print(f"[EvalReport] {prompt['id']} score={report['score']:.2f}\n{text}\n")

        bucket_scores: Dict[str, List[float]] = defaultdict(list)
        for report in prompt_reports:
            bucket_scores[str(report["bucket"])].append(float(report["score"]))

        bucket_summary = {
            bucket: round(sum(scores) / max(1, len(scores)), 4)
            for bucket, scores in sorted(bucket_scores.items())
        }
        avg_score = round(
            sum(float(report["score"]) for report in prompt_reports) / max(1, len(prompt_reports)), 4
        )

        if avg_score > self.best_score:
            self.best_score = avg_score
            self.best_step = state.global_step

        summary = {
            "step": state.global_step,
            "checkpoint": f"checkpoint-{state.global_step}",
            "eval_loss": None if not metrics else metrics.get("eval_loss"),
            "average_score": avg_score,
            "bucket_scores": bucket_summary,
            "best_score_so_far": round(self.best_score, 4),
            "best_step_so_far": self.best_step,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        report_doc = dict(summary)
        report_doc["prompts"] = prompt_reports

        report_path = os.path.join(self.report_dir, f"step-{state.global_step:05d}.json")
        with open(report_path, "w") as handle:
            json.dump(report_doc, handle, indent=2)

        summary["report_path"] = report_path
        with open(self.summary_output_path, "a") as handle:
            handle.write(json.dumps(summary) + "\n")

        with open(self.sample_output_path, "a") as handle:
            for record in sample_records:
                handle.write(json.dumps(record) + "\n")

        print(
            f"[EvalReport] avg_score={avg_score:.2f} "
            f"best_score={self.best_score:.2f} best_step={self.best_step} "
            f"eval_loss={summary['eval_loss']}"
        )
        print("[EvalReport] =====\n")
        self.last_logged_step = state.global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--eval-fraction", type=float, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args_cli = parser.parse_args()

    default_cfg = RunConfig()
    cfg = RunConfig()

    if args_cli.config:
        with open(args_cli.config, "r") as handle:
            overrides = json.load(handle)
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown config field: {key}")
            setattr(cfg, key, value)

    if args_cli.resume:
        cfg.resume_from = args_cli.resume
    if args_cli.run_name:
        cfg.run_name = args_cli.run_name
        if cfg.output_dir == default_cfg.output_dir:
            cfg.output_dir = f"checkpoints/{cfg.run_name}"
    if args_cli.eval_fraction is not None:
        cfg.eval_fraction = float(args_cli.eval_fraction)
    if args_cli.gradient_checkpointing:
        cfg.gradient_checkpointing = True

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    set_seed(cfg.seed)
    set_tf32(True)
    check_bf16_or_die(cfg.bf16)
    os.makedirs(cfg.output_dir, exist_ok=True)

    full = load_dataset_shards(cfg.train_root, cfg.limit_shards)
    full_stats = summarize_rows(full, cfg.block_size)
    print(f"[Data] summary={json.dumps(full_stats, sort_keys=True)}")

    train_ds = full
    eval_ds = None
    if cfg.eval_fraction and cfg.eval_fraction > 0:
        splits = full.train_test_split(test_size=cfg.eval_fraction, seed=cfg.seed)
        train_ds = splits["train"]
        eval_ds = splits["test"]
        train_stats = summarize_rows(train_ds, cfg.block_size)
        eval_stats = summarize_rows(eval_ds, cfg.block_size)
        print(f"[Data] train={len(train_ds):,} eval={len(eval_ds):,} (eval_fraction={cfg.eval_fraction})")
        print(f"[Data] train_summary={json.dumps(train_stats, sort_keys=True)}")
        print(f"[Data] eval_summary={json.dumps(eval_stats, sort_keys=True)}")

    tok = load_sparknet_tokenizer(cfg.tokenizer_path, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path)

    def die(message: str):
        raise RuntimeError(f"[Harmony] {message}")

    emb_vocab = model.get_input_embeddings().weight.shape[0]
    tok_vocab = len(tok)
    if emb_vocab != tok_vocab:
        die(
            f"Embedding vocab ({emb_vocab}) != tokenizer vocab ({tok_vocab}). "
            "Did you resize embeddings or save the wrong tokenizer?"
        )
    if tok.eos_token_id is None or tok.pad_token_id is None:
        die("Tokenizer missing eos/pad ids.")

    model.config.eos_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id
    if tok.bos_token_id is not None:
        model.config.bos_token_id = tok.bos_token_id

    print(
        f"[Harmony] tok_vocab={tok_vocab} emb_vocab={emb_vocab} "
        f"eos={tok.eos_token_id} pad={tok.pad_token_id} bos={tok.bos_token_id}"
    )

    try:
        model.config.attn_implementation = "sdpa"
    except Exception:
        pass
    try_log_sdpa_backend()

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        print("[Train] Gradient checkpointing enabled")

    ws = world_size()
    approx_tokens_per_step = cfg.block_size * cfg.per_device_train_batch_size * cfg.grad_accum * ws
    max_steps = math.ceil(cfg.target_tokens / approx_tokens_per_step)
    effective_batch = cfg.per_device_train_batch_size * cfg.grad_accum * ws
    steps_per_epoch = math.ceil(len(train_ds) / max(1, effective_batch))
    planned_epochs = max_steps / max(1, steps_per_epoch)
    print(
        f"[Budget] target_tokens={cfg.target_tokens:,} | approx tokens/step={approx_tokens_per_step:,} | "
        f"max_steps={max_steps:,} | world_size={ws} | effective_batch={effective_batch} | "
        f"planned_epochs={planned_epochs:.2f}"
    )
    warmup_steps = int(cfg.warmup_ratio * max_steps)
    print(f"[LR] warmup_ratio={cfg.warmup_ratio} -> warmup_steps={warmup_steps:,}")

    resume_from = cfg.resume_from
    if resume_from == "latest":
        resume_from = find_latest_checkpoint(cfg.output_dir)
        if resume_from:
            print(f"[Resume] latest checkpoint: {resume_from}")

    evaluation_strategy = "no"
    eval_steps = None
    load_best = False
    metric_for_best = None
    greater_is_better = None
    prompts: List[Dict[str, object]] = []

    if eval_ds is not None:
        evaluation_strategy = "steps"
        eval_steps = cfg.eval_steps or cfg.save_steps
        if cfg.save_steps % eval_steps != 0:
            print(
                f"[Eval] save_steps ({cfg.save_steps}) not multiple of eval_steps ({eval_steps}); "
                "setting eval_steps=save_steps."
            )
            eval_steps = cfg.save_steps
        load_best = True
        metric_for_best = "eval_loss"
        greater_is_better = False
        prompts = load_prompt_suite(cfg.sample_prompts_path)
        bucket_counts = Counter(str(prompt.get("bucket", "general")) for prompt in prompts)
        print(f"[Eval] loaded {len(prompts)} prompt(s) from {cfg.sample_prompts_path}")
        print(f"[Eval] bucket_counts={json.dumps(dict(sorted(bucket_counts.items())), sort_keys=True)}")

    collator = SFTPadCollator(tok, block_size=cfg.block_size)

    train_args = TrainingArguments(
        output_dir=cfg.output_dir,
        bf16=cfg.bf16,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.scheduler,
        max_steps=max_steps,
        logging_dir=f"logs/{cfg.run_name}",
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=load_best,
        metric_for_best_model=metric_for_best,
        greater_is_better=greater_is_better,
        optim="adamw_torch_fused",
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=(cfg.dataloader_num_workers > 0),
        dataloader_prefetch_factor=cfg.dataloader_prefetch_factor,
        max_grad_norm=cfg.max_grad_norm,
        save_safetensors=True,
    )

    callbacks: List[TrainerCallback] = [PerfCallback(approx_tokens_per_step)]
    if prompts:
        callbacks.append(
            EvalReportCallback(
                tokenizer=tok,
                prompts=prompts,
                output_dir=cfg.output_dir,
                max_context=cfg.block_size,
                max_new_tokens=cfg.sample_max_new_tokens,
                temperature=cfg.sample_temperature,
                top_p=cfg.sample_top_p,
                top_k=cfg.sample_top_k,
                repetition_penalty=cfg.sample_repetition_penalty,
                no_repeat_ngram_size=cfg.sample_no_repeat_ngram_size,
            )
        )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=resume_from)

    tok.save_pretrained(cfg.output_dir)
    trainer.save_model(cfg.output_dir)

    metadata = {
        "run_name": cfg.run_name,
        "output_dir": cfg.output_dir,
        "model_path": cfg.model_path,
        "tokenizer_path": cfg.tokenizer_path,
        "train_root": cfg.train_root,
        "block_size": cfg.block_size,
        "target_tokens": cfg.target_tokens,
        "world_size": ws,
        "approx_tokens_per_step": approx_tokens_per_step,
        "effective_batch_size": effective_batch,
        "steps_per_epoch": steps_per_epoch,
        "planned_epochs": planned_epochs,
        "max_steps": max_steps,
        "bf16": cfg.bf16,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "grad_accum": cfg.grad_accum,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio,
        "warmup_steps": warmup_steps,
        "scheduler": cfg.scheduler,
        "max_grad_norm": cfg.max_grad_norm,
        "eval_fraction": cfg.eval_fraction,
        "eval_steps": eval_steps,
        "save_steps": cfg.save_steps,
        "sample_prompts_path": cfg.sample_prompts_path if prompts else None,
        "sample_max_new_tokens": cfg.sample_max_new_tokens,
        "sample_temperature": cfg.sample_temperature,
        "sample_repetition_penalty": cfg.sample_repetition_penalty,
        "sample_no_repeat_ngram_size": cfg.sample_no_repeat_ngram_size,
        "checkpoint_reports_dir": os.path.join(cfg.output_dir, "checkpoint_reports") if prompts else None,
        "checkpoint_report_summary": os.path.join(cfg.output_dir, "checkpoint_report_summary.jsonl") if prompts else None,
        "dataset_summary": full_stats,
        "torch_version": torch.__version__,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(cfg.output_dir, "training_metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)

    print("Training complete")


if __name__ == "__main__":
    main()
