#!/usr/bin/env python3
"""
Minimal OpenAI-compatible inference server for SparkNet instruct checkpoints.

This server intentionally uses the same Hugging Face tokenizer/model stack that
was used for local evaluation, avoiding the tokenizer/runtime mismatches seen in
other serving stacks for legacy SentencePiece checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ALLOWED_ROLES = {"system", "user", "assistant"}


class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = set(stop_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if input_ids.numel() == 0:
            return False
        return int(input_ids[0, -1].item()) in self.stop_token_ids


def load_sparknet_tokenizer(tokenizer_path: Path):
    model_path = tokenizer_path / "tokenizer.model" if tokenizer_path.is_dir() else tokenizer_path
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
    tok = LlamaTokenizer(vocab_file=str(model_path), legacy=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_tokenizer(tokenizer_path: Path):
    try:
        tok = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=False)
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token = tok.eos_token
        return tok
    except Exception:
        return load_sparknet_tokenizer(tokenizer_path)


def build_chat_prompt(messages: List[Dict[str, str]]) -> str:
    chunks: List[str] = []
    for message in messages:
        role = message["role"]
        if role not in ALLOWED_ROLES:
            raise ValueError(f"Unsupported role: {role}")
        chunks.append(f"{IM_START}{role}\n{message['content'].strip()}{IM_END}\n")
    chunks.append(f"{IM_START}assistant\n")
    return "".join(chunks)


def stop_at_next_role_header(text: str) -> str:
    cut = None
    for marker in (IM_END, IM_START):
        idx = text.find(marker)
        if idx != -1:
            cut = idx if cut is None else min(cut, idx)
    if cut is not None:
        text = text[:cut]
    return text.strip()


class SparkNetModel:
    def __init__(
        self,
        model_path: Path,
        tokenizer_path: Path,
        served_model_name: str,
        device: str,
        max_context: Optional[int],
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.served_model_name = served_model_name
        self.device = self._resolve_device(device)
        self.tokenizer = load_tokenizer(tokenizer_path)

        torch_dtype = None
        if self.device == "cuda":
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.model.eval()

        self.max_context = max_context or int(getattr(self.model.config, "max_position_embeddings", 1024))
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.pad_token_id = int(self.tokenizer.pad_token_id)

        stop_ids = []
        for token in (IM_END, IM_START):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0:
                stop_ids.append(token_id)
        self.stopping_criteria = StoppingCriteriaList([StopOnToken(stop_ids)]) if stop_ids else None

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested != "auto":
            return requested
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _validate_budget(self, prompt_tokens: int, max_tokens: int):
        if prompt_tokens + max_tokens > self.max_context:
            max_input = self.max_context - max_tokens
            raise ValueError(
                f"You passed {prompt_tokens} input tokens and requested {max_tokens} output tokens. "
                f"The model context length is {self.max_context}, so the maximum input length is {max_input}."
            )

    def generate_from_prompt(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
    ) -> Dict[str, Any]:
        prompt_tokens = self._count_tokens(prompt)
        self._validate_budget(prompt_tokens, max_tokens)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = int(inputs["input_ids"].shape[1])
        do_sample = temperature > 0

        gen_kwargs: Dict[str, Any] = dict(
            **inputs,
            max_new_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            do_sample=do_sample,
        )
        if self.stopping_criteria is not None:
            gen_kwargs["stopping_criteria"] = self.stopping_criteria
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k

        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)[0]

        new_tokens = output_ids[input_len:]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        text = stop_at_next_role_header(raw_text)
        completion_tokens = int(new_tokens.shape[0])

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": "stop" if completion_tokens < max_tokens else "length",
        }

    def generate_from_messages(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        prompt = build_chat_prompt(messages)
        return self.generate_from_prompt(prompt, **kwargs)


def parse_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty list")

    parsed: List[Dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"unsupported role: {role}")
        if isinstance(content, list):
            parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
            content = "".join(parts)
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        parsed.append({"role": role, "content": content})
    return parsed


class OpenAIHandler(BaseHTTPRequestHandler):
    server_version = "SparkNetOpenAI/0.1"

    def do_GET(self):
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if self.path == "/v1/models":
            self._send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.state.served_model_name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "sparknet",
                        }
                    ],
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found", "type": "invalid_request_error"}})

    def do_POST(self):
        if not self._authorize():
            return
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
            return
        if self.path == "/v1/completions":
            self._handle_completions()
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found", "type": "invalid_request_error"}})

    def log_message(self, fmt: str, *args: Any):
        return

    def _authorize(self) -> bool:
        api_key = self.server.state.api_key
        if not api_key:
            return True
        auth_header = self.headers.get("Authorization", "")
        if auth_header == f"Bearer {api_key}":
            return True
        self._send_json(
            HTTPStatus.UNAUTHORIZED,
            {"error": {"message": "Invalid API key", "type": "authentication_error"}},
        )
        return False

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("request body is empty")
        body = self.rfile.read(length)
        return json.loads(body)

    def _sampling_params(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "max_tokens": int(payload.get("max_tokens", 256)),
            "temperature": float(payload.get("temperature", 0.0)),
            "top_p": float(payload.get("top_p", 0.9)),
            "top_k": int(payload.get("top_k", 0)),
            "repetition_penalty": float(payload.get("repetition_penalty", 1.12)),
            "no_repeat_ngram_size": int(payload.get("no_repeat_ngram_size", 4)),
        }

    def _handle_chat_completions(self):
        try:
            payload = self._read_json()
            messages = parse_messages(payload)
            if payload.get("model") not in (None, self.server.state.served_model_name):
                raise ValueError(f"unknown model: {payload.get('model')}")
            result = self.server.state.model.generate_from_messages(messages, **self._sampling_params(payload))
        except ValueError as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": str(exc), "type": "invalid_request_error"}},
            )
            return
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": {"message": str(exc), "type": "server_error"}},
            )
            return

        created = int(time.time())
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:18]}",
            "object": "chat.completion",
            "created": created,
            "model": self.server.state.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["text"],
                    },
                    "finish_reason": result["finish_reason"],
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            },
        }
        self._send_json(HTTPStatus.OK, response)

    def _handle_completions(self):
        try:
            payload = self._read_json()
            prompt = payload.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("prompt must be a non-empty string")
            if payload.get("model") not in (None, self.server.state.served_model_name):
                raise ValueError(f"unknown model: {payload.get('model')}")
            result = self.server.state.model.generate_from_prompt(prompt, **self._sampling_params(payload))
        except ValueError as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": str(exc), "type": "invalid_request_error"}},
            )
            return
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": {"message": str(exc), "type": "server_error"}},
            )
            return

        created = int(time.time())
        response = {
            "id": f"cmpl-{uuid.uuid4().hex[:16]}",
            "object": "text_completion",
            "created": created,
            "model": self.server.state.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "text": result["text"],
                    "finish_reason": result["finish_reason"],
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            },
        }
        self._send_json(HTTPStatus.OK, response)

    def _send_json(self, status: HTTPStatus, payload: Dict[str, Any]):
        blob = json.dumps(payload).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)


def parse_args() -> argparse.Namespace:
    root = Path("/home/mdiener/projects/sparknet")
    default_model = root / "checkpoints/sparknet-400m-v2-instruct-v2/checkpoint-800"
    default_tokenizer = root / "checkpoints/sparknet-400m-v2-instruct-v2"

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model-path", type=Path, default=default_model)
    parser.add_argument("--tokenizer-path", type=Path, default=default_tokenizer)
    parser.add_argument("--served-model-name", default="sparknet-400m-v2-instruct-v2")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-context", type=int, default=None)
    parser.add_argument("--api-key", default=os.environ.get("SPARKNET_API_KEY"))
    return parser.parse_args()


def main():
    args = parse_args()
    state = type("ServerState", (), {})()
    state.api_key = args.api_key
    state.served_model_name = args.served_model_name
    state.model = SparkNetModel(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        served_model_name=args.served_model_name,
        device=args.device,
        max_context=args.max_context,
    )

    server = ThreadingHTTPServer((args.host, args.port), OpenAIHandler)
    server.state = state  # type: ignore[attr-defined]

    print(
        f"Serving {args.served_model_name} on http://{args.host}:{args.port} "
        f"(model={args.model_path}, tokenizer={args.tokenizer_path}, device={state.model.device})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
