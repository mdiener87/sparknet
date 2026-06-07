#!/usr/bin/env python3
"""
Terminal chat for SparkNet SFT models using sparknet_chat_v1 scaffold.
"""

import argparse
from pathlib import Path
from typing import Dict, List
import random
import numpy as np

import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer

ROLE_PREFIX = {
    "system": "### System:\n",
    "user": "### User:\n",
    "assistant": "### Assistant:\n",
}

def build_prompt(messages: List[Dict[str, str]]) -> str:
    chunks: List[str] = []
    for msg in messages:
        chunks.append(ROLE_PREFIX[msg["role"]] + msg["content"] + "\n")
    chunks.append(ROLE_PREFIX["assistant"])
    return "".join(chunks)

def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])

def drop_oldest_turn(messages: List[Dict[str, str]]) -> bool:
    if not messages:
        return False
    # Remove the oldest user message and the following assistant reply (if present).
    # Preserve an initial system message at index 0 if present.
    idx = 1 if messages[0]["role"] == "system" else 0
    if idx >= len(messages):
        return False
    # remove oldest non-system message (usually a user message)
    messages.pop(idx)
    # if the following message is an assistant reply, remove it too
    if idx < len(messages) and messages[idx]["role"] == "assistant":
        messages.pop(idx)
    return True

def trim_history(messages, tokenizer, max_context: int, max_new_tokens: int) -> int:
    while True:
        prompt = build_prompt(messages)
        tokens = count_tokens(tokenizer, prompt)
        if tokens <= max_context - max_new_tokens:
            return tokens
        if not drop_oldest_turn(messages):
            return tokens

def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sparknet_tokenizer(tokenizer_path: str):
    path = Path(tokenizer_path).expanduser()
    model_path = path / "tokenizer.model" if path.is_dir() else path
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")

    tok = LlamaTokenizer(vocab_file=str(model_path), legacy=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_tokenizer_with_fallback(tokenizer_path: str):
    """Try HF-style AutoTokenizer first, fall back to the legacy SparkNet loader."""
    path = Path(tokenizer_path).expanduser()
    try:
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=False)
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token = tok.eos_token
        return tok
    except Exception:
        return load_sparknet_tokenizer(tokenizer_path)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="checkpoints/sparknet-410m-v1")
    p.add_argument("--tokenizer-path", type=str, default="./tokenizer-v6",
                   help="Use the canonical tokenizer used for training (recommended).")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--preset", choices=["deterministic", "human"], default="deterministic",
                   help="Choose sensible defaults for deterministic vs human evaluation")

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)

    p.add_argument("--max-context", type=int, default=None)
    p.add_argument("--system", type=str, default="You are Spark, a small and friendly AI assistant. Always identify yourself as Spark. You enjoy chatting and helping with simple questions. When you are not sure about something, you say so.",
                   help="Default empty to match most SFT data. Pass a system prompt explicitly if desired.")
    args = p.parse_args()

    device = resolve_device(args.device)

    # reproducible seeding if requested
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # load tokenizer with a fallback to the legacy tokenizer.model loader
    tokenizer = load_tokenizer_with_fallback(args.tokenizer_path)

    # apply preset sampling defaults only if the user didn't override them
    if args.preset == "human":
        if args.temperature == p.get_default("temperature"):
            args.temperature = 0.2
        if args.top_p == p.get_default("top_p"):
            args.top_p = 0.9
        if args.top_k == p.get_default("top_k"):
            args.top_k = 50
        if args.repetition_penalty == p.get_default("repetition_penalty"):
            args.repetition_penalty = 1.05
        if args.max_new_tokens == p.get_default("max_new_tokens"):
            args.max_new_tokens = 128

    torch_dtype = None
    if device == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    # Harmony prints
    print("tok vocab:", len(tokenizer))
    print("emb vocab:", model.get_input_embeddings().weight.shape[0])
    print("eos:", tokenizer.eos_token_id, "pad:", tokenizer.pad_token_id, "bos:", tokenizer.bos_token_id)

    max_context = args.max_context or getattr(model.config, "max_position_embeddings", 1024)

    messages: List[Dict[str, str]] = []
    if args.system.strip():
        messages.append({"role": "system", "content": args.system.strip()})

    print("SparkNet chat ready. Commands: /reset, /exit, /help")

    while True:
        try:
            user_text = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/help":
            print("Commands: /reset, /exit, /help")
            continue
        if user_text == "/reset":
            messages = []
            if args.system.strip():
                messages.append({"role": "system", "content": args.system.strip()})
            print("Conversation reset.")
            continue

        messages.append({"role": "user", "content": user_text})
        ctx_tokens = trim_history(messages, tokenizer, max_context, args.max_new_tokens)
        print(f"[ctx] {ctx_tokens}/{max_context} tokens")

        prompt = build_prompt(messages)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )

        # Only include sampling args if sampling
        do_sample = args.temperature is not None and args.temperature > 0
        gen_kwargs["do_sample"] = do_sample
        if do_sample:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p
            gen_kwargs["top_k"] = args.top_k

        with torch.no_grad():
            output_ids = model.generate(**gen_kwargs)[0]

        new_tokens = output_ids[input_len:]
        assistant_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # hard stop at next role header if it appears
        stops = ["\n### User:", "\n### System:", "\n### Assistant:"]
        cut = None
        for s in stops:
            i = assistant_text.find(s)
            if i != -1:
                cut = i if cut is None else min(cut, i)
        if cut is not None:
            assistant_text = assistant_text[:cut]

        assistant_text = assistant_text.strip()
        print(f"assistant> {assistant_text}")
        messages.append({"role": "assistant", "content": assistant_text})

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
