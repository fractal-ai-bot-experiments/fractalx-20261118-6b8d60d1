"""Text check (no number of the result uses it): greedy raw-text continuation of the use-eliciting prompt
for 30 random reporting nouns, to read what the untouched model says. Writes out/generations.jsonl."""
import sys, json, numpy as np, pandas as pd, torch
sys.path.insert(0, "/scratch"); sys.path.insert(0, "/scratch/exp")
from recipes.common import load_model
from extract import USE, art
w = pd.read_csv("/scratch/exp/data/words.csv"); w = w[w.split == "reporting"].sample(30, random_state=0)
model, tok = load_model("/weights/qwen3-8b", dtype="bfloat16")
rows = []
for n, lab in zip(w.w, w.top):
    p = USE.format(A=art(n), n=n)
    b = tok(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        g = model.generate(**b, max_new_tokens=24, do_sample=False)
    out = tok.decode(g[0, b["input_ids"].shape[1]:], skip_special_tokens=True)
    rows.append(dict(condition="base_model_use_prompt", seed=0, prompt=p, output=out, label=lab, word=n))
    print(n, lab, "->", repr(out))
with open("/scratch/exp/out/generations.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
