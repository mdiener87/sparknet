#!/usr/bin/env python3
"""
SparkNet-125M v1 pretraining script.

A scaled-down shakedown of the 410m pipeline: same tokenizer (tokenizer-v8),
same dataset code, same training loop and architecture *shape* (width, heads,
MLP ratio, rope_theta) — only the depth is reduced (32 -> 8 layers) to land at
~125M parameters. The point is a fast feedback loop (~1.5 days vs 16) for
validating the data + training stack before returning to the 410m run.

Key points:
  - PreTrainedTokenizerFast (ByteLevel BPE), 32k vocab, tied embeddings.
  - Corpus-sampled held-out eval shard (disjoint from train via hash holdout).
  - SmartCheckpointCallback manages three retention buckets:
      1. Top-3 by eval loss (protected from pruning)
      2. Token-cardinal snapshots: 1.5B, 2B, 2.5B (copied to separate dirs)
      3. Last-3 regular step checkpoints (power-failure resume)
  - Cosine schedule with min_lr floor (cosine_with_min_lr, floor = 0.1 x peak LR).
  - Learning rate 8e-4: scaled up from the 410m's 5.5e-4 because smaller models
    tolerate (and want) a higher peak LR. Not re-canaried; revisit if loss is
    unstable early.
  - Architecture: 8 layers, hidden 1024, intermediate 2816 -> ~127M parameters.
  - Dataset built with the hash-partitioned build_dataset.py, which fixes the
    410m-v1 shard-duplication bug (every shard streamed from document 0).

Usage:
  python train_pretrain.py --config configs/sparknet-125m/pretrain_v1.json
  python train_pretrain.py --config configs/sparknet-125m/pretrain_v1.json --resume latest
"""

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
import argparse

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    run_name: str = "sparknet-125m-v1"
    seed: int = 42

    tokenizer_path: str = "./tokenizer-v8"
    train_root: str = "datasets/sparknet-125m-v1-pretrain"
    eval_root: str = "datasets/sparknet-125m-v1-pretrain-eval"
    block_size: int = 1024

    # Architecture (~127M): same shape as 410m, depth reduced 32 -> 8
    hidden_size: int = 1024
    num_layers: int = 8
    num_heads: int = 16
    num_kv_heads: int = 8
    intermediate_size: int = 2816
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5

    # Training
    bf16: bool = True
    per_device_train_batch_size: int = 32
    grad_accum: int = 32

    learning_rate: float = 8e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.02
    cosine_min_lr_ratio: float = 0.1
    scheduler: str = "cosine_with_min_lr"
    max_grad_norm: float = 1.0

    # Budget
    target_tokens: int = 2_500_000_000
    checkpoint_tokens: List[int] = field(
        default_factory=lambda: [1_500_000_000, 2_000_000_000, 2_500_000_000]
    )
    keep_best_checkpoints: int = 3
    keep_last_checkpoints: int = 3

    # Cadence
    logging_steps: int = 50
    eval_steps: int = 250
    save_steps: int = 250

    dataloader_num_workers: int = 16
    dataloader_prefetch_factor: int = 4

    # Optional
    limit_shards: Optional[int] = None
    resume_from: Optional[str] = None
    do_sample_generations: bool = True
    sample_gen_every_eval: bool = True
    sample_prompts_path: Optional[str] = None
    # Per-dataset eval suite (built by build_eval_suite.py). None → single mixed eval.
    eval_suite_root: Optional[str] = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_tf32(enable: bool = True):
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32" if enable else "ieee"
    except Exception:
        pass
    try:
        torch.backends.cudnn.conv.fp32_precision = "tf32" if enable else "ieee"
    except Exception:
        pass


def resolve_repo_path(path: str) -> str:
    p = Path(path).expanduser()
    return str(p) if p.is_absolute() else str((REPO_ROOT / p).resolve())


def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    validate_tokenizer(tok)
    return tok


def validate_tokenizer(tok: PreTrainedTokenizerFast):
    expected_ids = {
        "<|begin_of_text|>": 0,
        "<|end_of_text|>": 1,
        "<|im_start|>": 2,
        "<|im_end|>": 3,
    }
    if tok.vocab_size != 32000 or len(tok) != 32000:
        raise ValueError(f"Expected tokenizer vocab/len 32000, got vocab_size={tok.vocab_size}, len={len(tok)}")
    if tok.bos_token != "<|begin_of_text|>" or tok.bos_token_id != 0:
        raise ValueError(f"Unexpected BOS token/id: {tok.bos_token!r}/{tok.bos_token_id}")
    if tok.eos_token != "<|end_of_text|>" or tok.eos_token_id != 1:
        raise ValueError(f"Unexpected EOS token/id: {tok.eos_token!r}/{tok.eos_token_id}")
    if tok.pad_token_id != tok.eos_token_id:
        raise ValueError(f"Expected PAD to share EOS id, got pad={tok.pad_token_id}, eos={tok.eos_token_id}")
    for token, expected_id in expected_ids.items():
        actual_id = tok.convert_tokens_to_ids(token)
        if actual_id != expected_id:
            raise ValueError(f"{token} id mismatch: expected {expected_id}, got {actual_id}")
    missing_additional = [t for t in ("<|im_start|>", "<|im_end|>") if t not in tok.additional_special_tokens]
    if missing_additional:
        raise ValueError(f"ChatML tokens missing from additional_special_tokens: {missing_additional}")
    if not tok.chat_template:
        raise ValueError("Tokenizer is missing chat_template")


def is_complete_shard(path: Path) -> bool:
    return (path / "dataset_info.json").exists() and (path / "state.json").exists()


def load_prepacked(root: str, limit: Optional[int] = None) -> Dataset:
    shards = sorted(Path(root).glob("shard-*"), key=lambda p: p.name)
    if not shards:
        raise FileNotFoundError(f"No shard-* dirs found under: {root}")
    incomplete = [str(p) for p in shards if not is_complete_shard(p)]
    if incomplete:
        raise RuntimeError("Incomplete shard directories found: " + ", ".join(incomplete))
    if limit is not None:
        shards = shards[:limit]
    print(f"[Data] Loading {len(shards)} shard(s) from {root}")
    dsets = [Dataset.load_from_disk(str(s)) for s in shards]
    ds = concatenate_datasets(dsets) if len(dsets) > 1 else dsets[0]
    ds.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
    print(f"[Data] rows={len(ds):,}")
    return ds


def load_eval_suite(cfg: "RunConfig"):
    """Return (eval_dataset, primary_metric).

    Without eval_suite_root: a single mixed eval Dataset and primary metric
    "eval_loss" (the original 410m behavior).

    With eval_suite_root populated: a dict {subset_name: Dataset, ..., "all":
    mixed}. HF Trainer evaluates each and emits eval_<name>_loss. The mixed set
    is keyed "all" and placed LAST so the per-subset report can print once all
    subsets for a step have been evaluated; "eval_all_loss" is the primary
    metric that drives best-checkpoint selection. The per-domain metrics are
    review-only and never touch the optimizer or checkpoint ranking.
    """
    mixed = load_prepacked(cfg.eval_root)
    if not cfg.eval_suite_root:
        return mixed, "eval_loss"
    root = Path(cfg.eval_suite_root)
    subsets = {}
    if root.is_dir():
        for sub in sorted(root.glob("*")):
            if (sub / "shard-000").is_dir():
                subsets[sub.name] = load_prepacked(str(sub))
    if not subsets:
        print(f"[Eval] eval_suite_root={root} has no subsets; using single mixed eval.")
        return mixed, "eval_loss"
    eval_dataset = {**subsets, "all": mixed}   # "all" last → clean per-step report
    print(f"[Eval] suite subsets: {list(eval_dataset.keys())}")
    return eval_dataset, "eval_all_loss"


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    ckpts = sorted(
        (
            p for p in Path(output_dir).glob("checkpoint-*")
            if p.is_dir() and p.name[len("checkpoint-"):].isdigit()
        ),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(ckpts[-1]) if ckpts else None


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class PerfCallback(TrainerCallback):
    def __init__(self, tokens_per_step: int):
        self.tokens_per_step = tokens_per_step
        self.t0 = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or self.t0 is None or state.global_step <= 0:
            return
        dt = time.time() - self.t0
        if dt <= 0:
            return
        tok_per_s = (state.global_step / dt) * self.tokens_per_step
        if "loss" in logs:
            print(f"[Perf] ~{tok_per_s:,.0f} tok/s | step={state.global_step:,}")


class SampleGenCallback(TrainerCallback):
    def __init__(self, tok, prompts: List[str], every_eval: bool = True):
        self.tok = tok
        self.prompts = prompts
        self.every_eval = every_eval

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not self.every_eval or model is None:
            return
        model.eval()
        device = next(model.parameters()).device
        print("\n[SampleGen] =====")
        with torch.no_grad():
            for p in self.prompts:
                ids = self.tok(p, return_tensors="pt").input_ids.to(device)
                out = model.generate(
                    ids,
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    repetition_penalty=1.05,
                    pad_token_id=self.tok.eos_token_id,
                    eos_token_id=self.tok.eos_token_id,
                )
                text = self.tok.decode(out[0], skip_special_tokens=True)
                print(f"\nPROMPT: {p}\n{text}\n")
        print("[SampleGen] =====\n")


class PerSourceEvalReport(TrainerCallback):
    """Print a per-subset loss + perplexity table once per eval round.

    With a dict eval_dataset, HF fires on_evaluate once per subset, each with
    only that subset's eval_<name>_loss. We accumulate those and print the full
    table when the primary metric (the mixed "all" set, ordered last) arrives.
    Review-only: this reads metrics, it does not influence training.
    """

    def __init__(self, primary_metric: str):
        self.primary_metric = primary_metric
        self._acc: dict = {}
        self._last_printed_step: int = -1

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        for k, v in metrics.items():
            if k.startswith("eval_") and k.endswith("_loss"):
                self._acc[k] = float(v)
        if self.primary_metric in metrics and state.global_step != self._last_printed_step:
            self._print(state.global_step)
            self._last_printed_step = state.global_step
            self._acc = {}

    def _print(self, step: int):
        if not self._acc:
            return
        label_w = max(len(k[len("eval_"):-len("_loss")]) for k in self._acc)
        print(f"\n[EvalSuite] step {step:,}")
        # primary ("all") first, then the rest alphabetically
        def sort_key(k):
            return (0 if k == self.primary_metric else 1, k)
        for k in sorted(self._acc, key=sort_key):
            loss = self._acc[k]
            ppl = math.exp(loss) if loss < 30 else float("inf")
            label = k[len("eval_"):-len("_loss")]
            print(f"  {label:<{label_w}}  loss {loss:8.4f}  ppl {ppl:12.2f}")
        print()


class SmartCheckpointCallback(TrainerCallback):
    """
    Three-bucket checkpoint retention policy:

      1. best-K    — top keep_best checkpoints by eval_loss, never pruned.
      2. cardinal  — token milestone snapshots (from cfg.checkpoint_tokens),
                     copied to checkpoint-{N}B-tokens dirs and never pruned.
      3. last-N    — keep_last most recent step checkpoints for power-failure
                     resume; rolling buffer, oldest pruned as new ones arrive.

    Eval loss is associated with each checkpoint by writing a .eval_loss marker
    file into the checkpoint directory at save time, using the most recent eval
    result. With eval_steps == save_steps the association is exact.
    """

    def __init__(self, output_dir: str, token_step_map: dict,
                 keep_best: int = 3, keep_last: int = 3,
                 primary_metric: str = "eval_loss"):
        self.output_dir = Path(output_dir)
        self.token_step_map = dict(token_step_map)   # step -> token_count
        self.keep_best = keep_best
        self.keep_last = keep_last
        # With a dict eval_dataset there is no plain "eval_loss" key; best-K
        # selection must key off the mixed set's metric (e.g. "eval_all_loss").
        self.primary_metric = primary_metric
        self._step_eval_loss: dict = {}
        self._last_eval_loss: Optional[float] = None
        self._triggered_cardinal: set = set()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and self.primary_metric in metrics:
            loss = float(metrics[self.primary_metric])
            self._step_eval_loss[state.global_step] = loss
            self._last_eval_loss = loss

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step in self.token_step_map and step not in self._triggered_cardinal:
            self._triggered_cardinal.add(step)
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        ckpt = self.output_dir / f"checkpoint-{step}"

        # Annotate with the most recent eval loss
        loss = self._step_eval_loss.get(step, self._last_eval_loss)
        if loss is not None and ckpt.exists():
            (ckpt / ".eval_loss").write_text(str(loss))

        # Cardinal snapshot
        if step in self.token_step_map and step in self._triggered_cardinal:
            token_count = self.token_step_map[step]
            n_b = token_count // 1_000_000_000
            cardinal = self.output_dir / f"checkpoint-{n_b}B-tokens"
            if not cardinal.exists() and ckpt.exists():
                shutil.copytree(str(ckpt), str(cardinal))
                print(f"[Checkpoint] Cardinal snapshot → {cardinal.name}")

        self._prune()
        return control

    def _prune(self):
        # Only manage numbered step dirs; cardinal dirs are left alone
        ckpts = sorted(
            (p for p in self.output_dir.iterdir()
             if p.is_dir() and p.name.startswith("checkpoint-")
             and p.name[len("checkpoint-"):].isdigit()),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not ckpts:
            return

        def eval_loss(p: Path) -> float:
            marker = p / ".eval_loss"
            if marker.exists():
                try:
                    return float(marker.read_text().strip())
                except ValueError:
                    pass
            return float("inf")

        best_k = {p for p in sorted(ckpts, key=eval_loss)[:self.keep_best]}
        last_n = set(ckpts[-self.keep_last:])
        protected = best_k | last_n

        for ckpt in ckpts:
            if ckpt not in protected:
                shutil.rmtree(str(ckpt), ignore_errors=True)
                print(f"[Checkpoint] Pruned {ckpt.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help='Checkpoint path or "latest".')
    parser.add_argument("--run-name", type=str, default=None)
    args_cli = parser.parse_args()

    cfg = RunConfig()
    if args_cli.config:
        with open(args_cli.config, "r") as f:
            overrides = json.load(f)
        for k, v in overrides.items():
            if k.startswith("_"):
                continue
            if not hasattr(cfg, k):
                raise ValueError(f"Unknown config field: {k!r}")
            setattr(cfg, k, v)
    if args_cli.resume:
        cfg.resume_from = args_cli.resume
    if args_cli.run_name:
        cfg.run_name = args_cli.run_name

    cfg.tokenizer_path = resolve_repo_path(cfg.tokenizer_path)
    cfg.train_root = resolve_repo_path(cfg.train_root)
    cfg.eval_root = resolve_repo_path(cfg.eval_root)
    if cfg.eval_suite_root:
        cfg.eval_suite_root = resolve_repo_path(cfg.eval_suite_root)
    if cfg.sample_prompts_path:
        cfg.sample_prompts_path = resolve_repo_path(cfg.sample_prompts_path)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_tf32(True)
    set_seed(cfg.seed)

    run_dir = str((REPO_ROOT / "checkpoints" / cfg.run_name).resolve())
    log_dir = str((REPO_ROOT / "logs" / cfg.run_name).resolve())
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Tokenizer
    tok = load_tokenizer(cfg.tokenizer_path)

    # Data
    train_ds = load_prepacked(cfg.train_root, limit=cfg.limit_shards)
    eval_ds, primary_metric = load_eval_suite(cfg)

    # Model
    model_cfg = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_kv_heads,
        max_position_embeddings=cfg.block_size,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(model_cfg)
    model.config.use_cache = False
    model.gradient_checkpointing_disable()
    model.config.attn_implementation = "sdpa"

    if hasattr(torch.backends.cuda, "sdp_kernel"):
        torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=True, enable_math=False
        )

    # Budget
    tokens_per_step = cfg.block_size * cfg.per_device_train_batch_size * cfg.grad_accum
    max_steps = math.ceil(cfg.target_tokens / tokens_per_step)
    checkpoint_steps = [math.ceil(t / tokens_per_step) for t in cfg.checkpoint_tokens]
    token_step_map = dict(zip(checkpoint_steps, cfg.checkpoint_tokens))

    print(f"[Budget] target={cfg.target_tokens:,} | tok/step={tokens_per_step:,} | steps={max_steps:,}")
    print(f"[Budget] cardinal steps: {checkpoint_steps}")

    # Scheduler: cosine_with_min_lr requires transformers >= 4.38.
    # If unavailable, falls back to plain cosine with a logged warning.
    import transformers as _tf
    from packaging.version import Version
    if cfg.scheduler == "cosine_with_min_lr":
        if Version(_tf.__version__) >= Version("4.38.0"):
            scheduler_type = "cosine_with_min_lr"
            scheduler_kwargs = {"min_lr_rate": cfg.cosine_min_lr_ratio}
        else:
            print(
                f"[Warning] cosine_with_min_lr requires transformers>=4.38 "
                f"(found {_tf.__version__}). Falling back to plain cosine."
            )
            scheduler_type = "cosine"
            scheduler_kwargs = {}
    else:
        scheduler_type = cfg.scheduler
        scheduler_kwargs = {}

    train_args = TrainingArguments(
        output_dir=run_dir,
        bf16=cfg.bf16,

        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,  # eval has no grads; bigger batch speeds the now-multi-subset eval
        gradient_accumulation_steps=cfg.grad_accum,

        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=scheduler_type,
        lr_scheduler_kwargs=scheduler_kwargs,

        max_steps=max_steps,
        max_grad_norm=cfg.max_grad_norm,

        logging_dir=log_dir,
        logging_steps=cfg.logging_steps,

        eval_strategy="steps",
        eval_steps=cfg.eval_steps,

        # Track the mixed-set metric as "best". SmartCheckpointCallback owns
        # actual checkpoint retention; this also lets HF report best_metric.
        metric_for_best_model=primary_metric,
        greater_is_better=False,

        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=None,          # SmartCheckpointCallback owns retention

        optim="adamw_torch_fused",

        report_to=["tensorboard"],
        remove_unused_columns=False,

        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=cfg.dataloader_prefetch_factor,
    )

    prompts = [
        "The meaning of life is",
        "In the future, AI will",
        "A good software engineer knows that",
        "Write a Python function that",
        "If I have 12 apples and give away 5,",
        "Explain the difference between supervised and unsupervised learning:",
        "The capital of France is",
    ]
    if cfg.sample_prompts_path:
        with open(cfg.sample_prompts_path, "r") as f:
            prompts = json.load(f)

    smart_ckpt = SmartCheckpointCallback(
        output_dir=run_dir,
        token_step_map=token_step_map,
        keep_best=cfg.keep_best_checkpoints,
        keep_last=cfg.keep_last_checkpoints,
        primary_metric=primary_metric,
    )
    callbacks = [PerfCallback(tokens_per_step), smart_ckpt]
    if isinstance(eval_ds, dict):
        callbacks.append(PerSourceEvalReport(primary_metric))
    if cfg.do_sample_generations:
        callbacks.append(SampleGenCallback(tok, prompts, every_eval=cfg.sample_gen_every_eval))

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    # Resume
    resume_path = None
    if cfg.resume_from:
        resume_path = find_latest_checkpoint(run_dir) if cfg.resume_from == "latest" else cfg.resume_from
        print(f"[Resume] {resume_path}")

    # Persist config before training starts so a crash still leaves a record
    with open(f"{run_dir}/run_config.json", "w") as f:
        json.dump({
            "run_name": cfg.run_name,
            "timestamp": datetime.now().isoformat(),
            "config": asdict(cfg),
            "derived": {
                "tokens_per_step": tokens_per_step,
                "max_steps": max_steps,
                "checkpoint_steps": checkpoint_steps,
            },
        }, f, indent=2)

    trainer.train(resume_from_checkpoint=resume_path)

    # Save final artifacts
    tok.model_max_length = cfg.block_size
    tok.save_pretrained(run_dir)
    model.save_pretrained(run_dir)

    with open(f"{run_dir}/run_summary.json", "w") as f:
        json.dump({
            "finished_at": datetime.now().isoformat(),
            "global_step": trainer.state.global_step,
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
        }, f, indent=2)


if __name__ == "__main__":
    main()
