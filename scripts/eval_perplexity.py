import math, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH   = "checkpoints/sparknet-70m-v1-final"
CTX          = 1024        # your training block_size
STRIDE       = 512         # evaluate only the new 512 tokens each window
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_grad_enabled(False)

# --- load
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE).eval()

print(f"sanity: vocab={tok.vocab_size}, eos_id={tok.eos_token_id}, "
      f"max_pos={getattr(model.config,'n_positions', getattr(model.config,'max_position_embeddings', None))}")

# --- data
ds = load_dataset("codelion/fineweb-edu-1B", split="train[:1%]")
text = "\n\n".join(ds["text"])

# Tokenize WITHOUT truncation; warning is harmless, but we’ll only slice windows to model
enc = tok(text, add_special_tokens=False, return_tensors="pt")
ids = enc.input_ids.squeeze(0).to(DEVICE)

nll_sum = 0.0
count   = 0

for i in range(0, ids.size(0) - CTX, STRIDE):
    # window [i : i+CTX]
    chunk = ids[i : i + CTX].unsqueeze(0)              # (1, CTX)
    # mask loss to only last STRIDE tokens in the window
    labels = chunk.clone()
    labels[:, :-STRIDE] = -100                         # ignore context portion
    out = model(input_ids=chunk, labels=labels)
    # out.loss is mean over non-masked positions; multiply by #targets to get NLL
    nll_sum += out.loss.item() * STRIDE
    count   += STRIDE

ppl = math.exp(nll_sum / count)
print(f"Perplexity: {ppl:.3f} (tokens evaluated: {count})")
