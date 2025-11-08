import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from textwrap import indent
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
MODEL_PATH = "checkpoints/sparknet-70m-instruct-v1"  # your model
REFERENCE_MODEL = "gpt2"                          # optional baseline
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 30
MAX_TOKENS = 100
SEED = 42
OUTPUT_DIR = Path("logs")

# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------
def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path).to(DEVICE)
    model.eval()
    return tok, model

tok_spark, model_spark = load_model(MODEL_PATH)
tok_ref, model_ref = load_model(REFERENCE_MODEL)

set_seed(SEED)
print(f"Loaded SparkNet and reference model on {DEVICE}")

# Prep output file
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_file = OUTPUT_DIR / f"eval_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
run_header = (
    f"Eval run: {datetime.now().isoformat()}\n"
    f"Spark model: {MODEL_PATH}\n"
    f"Reference model: {REFERENCE_MODEL}\n"
    f"Device: {DEVICE}\n"
    f"Temperature: {TEMPERATURE}, top_p: {TOP_P}, max_tokens: {MAX_TOKENS}, seed: {SEED}\n"
)
print(f"Writing detailed results to {output_file}")

# ---------------------------------------------------------
# Generation helper
# ---------------------------------------------------------
def generate(prompt, model, tok, temp=TEMPERATURE, top_p=TOP_P, top_k=TOP_K):
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tok.eos_token_id
        )
    return tok.decode(output[0], skip_special_tokens=True)

# ---------------------------------------------------------
# Intrinsic probing prompts
# ---------------------------------------------------------
PROMPTS = [
    # factual
    "The capital of France is",
    "The largest ocean on Earth is",
    # reasoning / logic
    "If all cats are animals and some animals are black, then some cats are",
    # arithmetic
    "2 + 2 =",
    "The square root of 9 is",
    # completion / coherence
    "In the quiet heart of Mechanus,",
    "Once upon a time,",
    "The scientist adjusted the lens and said,"
]

# ---------------------------------------------------------
# Run tests
# ---------------------------------------------------------
results = [run_header]

for p in PROMPTS:
    section_lines = [f"\n=== Prompt: {p!r} ==="]

    spark_out = generate(p, model_spark, tok_spark)
    ref_out   = generate(p, model_ref, tok_ref)

    section_lines.append("🟡 SparkNet:")
    spark_text = indent(spark_out[len(p):].strip(), "  ")
    section_lines.append(spark_text)
    section_lines.append("🔵 GPT-2:")
    ref_text = indent(ref_out[len(p):].strip(), "  ")
    section_lines.append(ref_text)

    section_block = "\n".join(section_lines)
    print(section_block)
    results.append(section_block)

with output_file.open("w", encoding="utf-8") as f:
    f.write("\n".join(results))
