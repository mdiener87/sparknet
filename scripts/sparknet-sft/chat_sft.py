#!/usr/bin/env python3
"""
Lightweight terminal chat for SparkNet SFT models.
Uses the same chat template as build_dataset.py (sparknet_chat_v1).
"""

import argparse
import sys
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROLE_PREFIX = {
    "system": "### System:\n",
    "user": "### User:\n",
    "assistant": "### Assistant:\n",
}


def build_prompt(messages: List[Dict[str, str]]) -> str:
    chunks: List[str] = []
    for msg in messages:
        role = msg["role"]
        chunks.append(ROLE_PREFIX[role] + msg["content"] + "\n")
    chunks.append(ROLE_PREFIX["assistant"])
    return "".join(chunks)


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def drop_oldest_turn(messages: List[Dict[str, str]]) -> bool:
    if not messages:
        return False
    start_idx = 1 if messages[0]["role"] == "system" else 0
    if start_idx >= len(messages):
        return False
    messages.pop(start_idx)
    if start_idx < len(messages) and messages[start_idx]["role"] == "assistant":
        messages.pop(start_idx)
    return True


def trim_history(
    messages: List[Dict[str, str]],
    tokenizer,
    max_context: int,
    max_new_tokens: int,
) -> int:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/sparknet-400m-v1-instruct-v2/checkpoint-1000")
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-context", type=int, default=None)
    parser.add_argument("--system", type=str, default="You are a helpful assistant named SparkNet.")
    args = parser.parse_args()

    tok_path = args.tokenizer_path or args.model_path
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = None
    if device == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch_dtype).to(device)
    model.eval()

    print("tok vocab:", len(tokenizer))
    print("emb vocab:", model.get_input_embeddings().weight.shape[0])
    print("eos:", tokenizer.eos_token_id, "pad:", tokenizer.pad_token_id, "bos:", tokenizer.bos_token_id)


    max_context = args.max_context or getattr(model.config, "max_position_embeddings", 1024)
    messages: List[Dict[str, str]] = []

    if args.system:
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
            if args.system:
                messages.append({"role": "system", "content": args.system.strip()})
            print("Conversation reset.")
            continue

        messages.append({"role": "user", "content": user_text})
        ctx_tokens = trim_history(messages, tokenizer, max_context, args.max_new_tokens)
        print(f"[ctx] {ctx_tokens}/{max_context} tokens")

        prompt = build_prompt(messages)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size = 3
            )[0]

        new_tokens = output_ids[input_len:]
        # assistant_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
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
