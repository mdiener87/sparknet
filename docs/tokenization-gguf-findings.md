# SparkNet: Tokenization Mismatch — Findings and v3 Implications

Status: written 2026-04-26, based on GGUF deployment testing of v2 instruct checkpoint

---

## What we found

When converting the `sparknet-400m-v2-instruct-v1-short` checkpoint to GGUF and running it
via `llama-server`, output was incoherent garbage — symbol salad, repeated punctuation, no
coherent words. The base HF model ran fine with identical temperature/sampling settings.

Root cause: **HF's `LlamaTokenizer` and llama.cpp's pre-tokenizer handle newlines differently**,
and every role-prefix in the training format (`sparknet_chat_v1`) contains a newline.

---

## The mismatch in detail

`sparknet_chat_v1` uses these role prefixes:

```
### System:\n
### User:\n
### Assistant:\n
```

**HF `LlamaTokenizer` (legacy=True, add_prefix_space=True):**
- SentencePiece treats `\n` as generic whitespace.
- It absorbs `\n` and merges it with whitespace prefix logic for the following word.
- `"### User:\nHello"` → `[▁###, ▁User, :, ▁Hello]` — the `\n` disappears.

**llama.cpp pre-tokenizer:**
- Splits on whitespace and newline boundaries before passing to SentencePiece.
- `\n` becomes an explicit byte token: `<0x0A>` (token ID 14).
- `"### User:\nHello"` → `[▁###, ▁User, :, <0x0A>, ▁Hello]` — the `\n` is a real token.

**Effect:** Every role prefix in the prompt produces a different token sequence depending on
which tokenizer runs it. The model was trained on HF-tokenized sequences. At GGUF inference,
it receives llama.cpp-tokenized sequences. The context the model uses to begin generating is
entirely different from what it learned from.

This is not a GGUF vocabulary bug. MD5 checksums of `tokenizer.model` are identical between
the HF checkpoint and the extracted GGUF vocabulary — the vocab mapping is correct. The bug
is in how each stack *applies* that vocabulary.

The mismatch is structural: it occurs at every newline in every role prefix (at minimum 2–3
times per conversation), not just in edge cases.

---

## Why the base model appeared fine

The raw HF model tested directly via `chat_sft.py` (with `AutoModelForCausalLM` +
`LlamaTokenizer`) runs perfectly. That path uses HF tokenization end-to-end, so training and
inference are consistent. The problem only surfaces when a GGUF-based inference server
(llama.cpp, Ollama, llama-server) tokenizes the same prompt.

---

## The ChatML v7 mitigation (immediate fix)

For the v2 SFT re-run (`sft_chat_v7` / `instruct-v2`), the chat format was changed to
**ChatML**:

```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
```

`<|im_start|>` (id=32000) and `<|im_end|>` (id=32001) are registered as
`additional_special_tokens` in the HF tokenizer. Special tokens are always emitted as their
own single IDs and are never passed to SentencePiece — both HF and llama.cpp handle them
identically.

**What this fixes:**
- Role boundary tokens are now identical in HF training and GGUF inference.
- The model can reliably learn to start and end turns at special token boundaries.
- The Jinja2 ChatML template is embedded in `tokenizer_config.json` during `save_pretrained`,
  so `convert_hf_to_gguf.py` bakes it into the GGUF automatically.

**What this does NOT fix:**
- The `\n` *after* the role name (e.g., `<|im_start|>system\n`) is still tokenized
  differently between HF and llama.cpp. The role header is fully masked during training
  so this does not affect supervised loss, but it does mean the model receives a slightly
  different token context for each turn header at inference time.
- `\n` characters *inside* message content are still tokenized differently. HF absorbs
  them as whitespace; llama.cpp emits `<0x0A>`. Assistant responses in the training data
  that span multiple lines will have different token sequences at HF vs GGUF inference.

**Net effect of ChatML v7:**
- Role boundaries are clean and consistent — the model reliably learns turn structure.
- Internal newlines in content still diverge, but single-paragraph assistant responses
  (which the v7 dataset strongly encourages via the 200-word cap) have few internal `\n`,
  so the practical impact is small.
- ChatML is a meaningful improvement over `sparknet_chat_v1` for GGUF deployment.

---

## The proper fix: BPE tokenizer for v3

The fundamental problem is SentencePiece's whitespace/newline absorption behavior. This is
inherent to the algorithm when used in byte-fallback mode with `add_prefix_space=True`.

**The right fix is to train v3 with a BPE tokenizer (tiktoken-style, as used by LLaMA 3).**

BPE tokenizers treat every byte position independently. There is no whitespace-absorption
ambiguity. Both HF's `tiktoken`-compatible tokenizer and llama.cpp's BPE pre-tokenizer
produce identical token sequences for identical strings — including newlines.

This eliminates the train/inference distribution shift completely, for all content, in all
positions.

### Tokenizer recommendations for v3 pretraining

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Algorithm | BPE (tiktoken-style) | Identical HF/llama.cpp tokenization, no newline mismatch |
| Vocabulary size | 32k–64k | 32k is fine for English-centric demo; 64k adds headroom for code/math |
| Training corpus | Sample from the v3 pretraining corpus | Tokenizer should reflect actual data distribution, not a prior corpus |
| Special tokens | Reserve slots for ChatML (`<\|im_start\|>`, `<\|im_end\|>`) | Avoids embedding resize at SFT time |
| BOS/EOS | Follow LLaMA 3 conventions (`<\|begin_of_text\|>`, `<\|end_of_text\|>`) | Known-good, well-tested by llama.cpp |

If building from scratch, `sentencepiece` with BPE mode is one option, but training a
`tiktoken`-compatible tokenizer with `tokenizers` (HuggingFace) is simpler and produces
a format that both HF and llama.cpp handle natively.

---

## SFT chat format recommendation for v3

Since v3 will use a BPE tokenizer, and since llama.cpp natively supports ChatML as a
built-in template, the recommendation is:

- **Use standard ChatML for v3 SFT.** No custom format needed.
- Register `<|im_start|>` and `<|im_end|>` in the base vocabulary (not as additional_special_tokens
  added at SFT time) so the model sees them during pretraining. Even a small exposure in Stage C
  (conversational continuation tokens) will help.
- At GGUF conversion time, embed the standard ChatML Jinja2 template via `tokenizer_config.json`
  — the same pattern already established in `train_sft_v7.py`.
- At `llama-server` launch, no `--chat-template-file` flag is needed; the template is embedded
  in the GGUF and `llama-server` reads it automatically.

---

## Embedding resize at SFT time (v2 workaround)

For v2, because `<|im_start|>` and `<|im_end|>` were not in the base pretraining vocabulary,
the SFT trainer must grow the embedding table:

1. `LlamaTokenizer.add_special_tokens({"additional_special_tokens": [...]})` → vocab 32000 → 32002
2. `model.resize_token_embeddings(len(tok))` → HF initializes the two new rows to the mean embedding

This is a reasonable workaround but it means:
- The new token embeddings start from a generic initialization, not learned signal.
- The model has to learn `<|im_start|>` / `<|im_end|>` semantics purely from the SFT pass.
- A single ~5-hour SFT epoch is not a lot of signal for truly new vocabulary entries.

For v3, by including these tokens in the base pretraining vocabulary from the start, the
embeddings will be properly trained on billions of tokens instead of being initialized at SFT
time from the mean.

---

## Deployment checklist for v2 GGUF (ChatML v7)

After `train_sft_v7.py` completes:

1. Verify `tokenizer_config.json` contains `chat_template` (written by `tok.save_pretrained`).
2. Run `convert_hf_to_gguf.py` — the template is embedded automatically.
3. Quantize to Q4_K_M.
4. Launch `llama-server` with **no** `--chat-template-file` and **no** `--chat-template` flag.
   llama-server reads the embedded template from the GGUF.
5. Smoke test: `curl -s http://localhost:8080/v1/chat/completions` with a simple user message.
   Verify the response is coherent and the model identifies as Spark.
6. If the template is somehow not detected, fall back to `--chat-template chatml` (built-in alias
   in llama.cpp for the standard ChatML template).

---

## Summary

| Issue | Scope | Status |
|-------|-------|--------|
| `\n` token mismatch — role prefixes | All `sparknet_chat_v1` SFT runs | Addressed by ChatML v7 (role boundaries are now special tokens) |
| `\n` token mismatch — role name header | ChatML v7 (masked region) | Residual; low impact since header is masked during training |
| `\n` token mismatch — assistant content | ChatML v7 (supervised region) | Residual; low impact for single-paragraph responses |
| Full mismatch elimination | v3 BPE tokenizer | Proper fix; eliminates all SentencePiece whitespace issues |
| Embedding resize at SFT time | v2 ChatML v7 | Workaround; v3 avoids by including special tokens in base vocab |
