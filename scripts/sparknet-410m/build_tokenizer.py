#!/usr/bin/env python3
"""
Build tokenizer-v7: ByteLevel BPE tokenizer for SparkNet-410M.

Replaces tokenizer-v6 (SentencePiece). The root cause of the v2 GGUF deployment
failure was SentencePiece absorbing \\n as whitespace while llama.cpp emits an
explicit <0x0A> byte token — a mismatch at every role boundary. ByteLevel BPE
processes every byte position independently, so HF and llama.cpp produce identical
token sequences for all inputs including newlines.

Special tokens baked into the base vocabulary (not added at SFT time):
  <|begin_of_text|>  BOS
  <|end_of_text|>    EOS + PAD
  <|im_start|>       ChatML turn start
  <|im_end|>         ChatML turn end

Including <|im_start|> and <|im_end|> in pretraining vocab means their embeddings
are trained on billions of tokens before SFT rather than being initialized to the
mean embedding at fine-tune time.

Output: tokenizer-v7/  (PreTrainedTokenizerFast-compatible HF directory)

Usage:
  python build_tokenizer.py
  python build_tokenizer.py --vocab-size 32000 --samples 2000000 --output tokenizer-v7
"""

import argparse
import random
from pathlib import Path
from typing import Iterator

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

REPO_ROOT = Path(__file__).resolve().parents[2]

SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|im_start|>",
    "<|im_end|>",
]

# Mirrors v3 pretraining mix so tokenizer reflects actual data distribution
DATA_SOURCES = [
    ("HuggingFaceFW/fineweb-edu",         None,           "train", 0.44),
    ("mlfoundations/dclm-baseline-1.0",   None,           "train", 0.13),
    ("HuggingFaceFW/finepdfs",            "eng_Latn",     "train", 0.11),
    ("HuggingFaceFW/finewiki",            "en",           "train", 0.10),
    ("HuggingFaceTB/smollm-corpus",       "cosmopedia-v2","train", 0.10),
    ("HuggingFaceTB/smollm-corpus",       "python-edu",   "train", 0.07),
    ("open-web-math/open-web-math",        None,           "train", 0.05),
]

# Standard ChatML template — embedded in tokenizer_config.json so llama.cpp
# reads it automatically from the GGUF without needing --chat-template-file.
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--samples", type=int, default=2_000_000,
                        help="Number of text samples to feed to the BPE trainer.")
    parser.add_argument("--output", type=str, default="tokenizer-v7")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stream_texts(n_samples: int, seed: int) -> Iterator[str]:
    rng = random.Random(seed)
    names = [s[0] for s in DATA_SOURCES]
    configs = [s[1] for s in DATA_SOURCES]
    splits = [s[2] for s in DATA_SOURCES]
    weights = [s[3] for s in DATA_SOURCES]

    iterators = {}
    for i, (name, cfg_name, split, _) in enumerate(DATA_SOURCES):
        ds = load_dataset(name, name=cfg_name, split=split, streaming=True, trust_remote_code=False)
        iterators[i] = iter(ds)

    emitted = 0
    while emitted < n_samples:
        idx = rng.choices(range(len(DATA_SOURCES)), weights=weights, k=1)[0]
        try:
            row = next(iterators[idx])
        except StopIteration:
            ds = load_dataset(names[idx], name=configs[idx], split=splits[idx],
                              streaming=True, trust_remote_code=False)
            iterators[idx] = iter(ds)
            row = next(iterators[idx])

        text = row.get("text", "")
        if isinstance(text, str) and text.strip():
            yield text.strip()
            emitted += 1
            if emitted % 100_000 == 0:
                print(f"  Streamed {emitted:,} / {n_samples:,} samples")


def main():
    args = parse_args()
    output_dir = (REPO_ROOT / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training BPE tokenizer: vocab_size={args.vocab_size}, samples={args.samples:,}")
    print(f"Output: {output_dir}")

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    print("Streaming training text...")
    text_iter = stream_texts(args.samples, args.seed)
    tokenizer.train_from_iterator(text_iter, trainer=trainer, length=args.samples)

    print(f"Trained vocabulary size: {tokenizer.get_vocab_size()}")

    # Wrap as PreTrainedTokenizerFast for HF + llama.cpp compatibility
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
        pad_token="<|end_of_text|>",
        unk_token=None,
        model_max_length=1024,
    )

    # ChatML template embedded here so convert_hf_to_gguf.py bakes it into
    # the GGUF automatically — no --chat-template-file flag needed at serve time.
    fast_tok.chat_template = CHATML_TEMPLATE

    fast_tok.save_pretrained(str(output_dir))
    print(f"Tokenizer saved to {output_dir}")

    # Smoke test
    test_cases = [
        "Hello, world!\nThis is a test.",
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>",
        "def fibonacci(n):\n    if n <= 1:\n        return n",
    ]
    print("\nSmoke test:")
    for text in test_cases:
        ids = fast_tok.encode(text)
        decoded = fast_tok.decode(ids)
        match = "OK" if decoded == text else "MISMATCH"
        print(f"  [{match}] {repr(text[:60])}")
        if decoded != text:
            print(f"         decoded: {repr(decoded[:60])}")

    vocab = fast_tok.get_vocab()
    print("\nSpecial token IDs:")
    for tok in SPECIAL_TOKENS:
        print(f"  {tok!r:30s} -> {vocab.get(tok, 'NOT FOUND')}")


if __name__ == "__main__":
    main()
