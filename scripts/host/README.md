# SparkNet Host

Minimal OpenAI-compatible server for SparkNet instruct checkpoints using the
same Hugging Face tokenizer/model stack used during local evaluation.

## Run

From the repo root:

```bash
python scripts/host/openai_hf_server.py
```

Defaults:

- model: `checkpoints/sparknet-400m-v2-instruct-v2/checkpoint-800`
- tokenizer: `checkpoints/sparknet-400m-v2-instruct-v2`
- host: `0.0.0.0`
- port: `8002`

## Example

```bash
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sparknet-400m-v2-instruct-v2",
    "messages": [
      {"role": "system", "content": "You are Spark, a concise helpful assistant."},
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "max_tokens": 64,
    "temperature": 0.0
  }'
```

## Notes

- This server formats chat prompts exactly like `train_sft_v7.py`.
- It stops on `<|im_end|>` / `<|im_start|>` and trims those markers from the response.
- Open WebUI can connect to it as an OpenAI-compatible endpoint.
