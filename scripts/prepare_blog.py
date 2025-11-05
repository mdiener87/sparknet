# tools/prepare_blog_jsonl.py
import pathlib, re, json

in_dir = pathlib.Path("data/dienertech")
out = open("data/dienertech_processed/diener_blog.jsonl", "w", encoding="utf-8")

fm_re = re.compile(r"^---.*?---\s*", re.S)         # YAML front-matter
code_re = re.compile(r"```.*?```", re.S)           # fenced code blocks
html_re = re.compile(r"<[^>]+>")                   # naive HTML strip

for md in sorted(in_dir.glob("*.md")):
    txt = md.read_text(encoding="utf-8")
    txt = fm_re.sub("", txt)
    txt = code_re.sub("", txt)
    txt = html_re.sub(" ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    if len(txt) > 0:
        out.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")

out.close()
print("Wrote data/diener_blog.jsonl")
