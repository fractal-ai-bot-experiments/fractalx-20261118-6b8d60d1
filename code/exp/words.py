"""Step 1: build the tool-noun list from the Lancaster Sensorimotor Norms (plan step 1)."""
import json, re, sys
import numpy as np, pandas as pd
from nltk.corpus import wordnet as wn
import gensim.downloader as api

OUT = "/scratch/exp/data"
import os; os.makedirs(OUT, exist_ok=True)
df = pd.read_csv("/scratch/data_dl/lancaster.csv")
df["w"] = df["Word"].astype(str).str.lower()
glove = api.load("glove-wiki-gigaword-300")

LONG = ["hand", "foot", "feet", "mouth", "finger", "thumb", "tongue", "tooth", "teeth", "knee", "wrist",
        "ankle", "palm", "elbow", "heel", "lip", "shin", "thigh", "toenail", "fingernail", "knuckle", "fist", "throat", "cheek"]
SHORT = ["arm", "leg", "toe", "jaw", "gum"]

def has_bodypart(w):
    if any(s in w for s in LONG):
        return True
    return any(w.startswith(s) or w.endswith(s) for s in SHORT)

INSTR = wn.synset("instrumentality.n.03")
def first_noun(w):
    s = wn.synsets(w, pos=wn.NOUN)
    return s[0] if s else None

def under_instr(s):
    return any(INSTR in path for path in s.hypernym_paths())

cols = {"hand": "Hand_arm.mean", "foot": "Foot_leg.mean", "mouth": "Mouth.mean"}
CLASSES = ["hand", "foot", "mouth"]
A = df[[cols[c] for c in CLASSES]].values
top = A.argmax(1)
srt = np.sort(A, 1)
margin = srt[:, 2] - srt[:, 1]
df["top"] = [CLASSES[i] for i in top]; df["topval"] = srt[:, 2]; df["margin"] = margin

base_ok = []
info = {}
for i, w in enumerate(df["w"]):
    ok = w.isalpha() and w.isascii() and (w in glove.key_to_index) and not has_bodypart(w)
    s = first_noun(w) if ok else None
    if s is None:
        base_ok.append(None); continue
    food = s.lexname() == "noun.food"
    info[w] = dict(lex=s.lexname(), instr=under_instr(s), food=food, syn=s.name())
    base_ok.append(w)
df["base"] = [b is not None for b in base_ok]

def candidate(tier, m):
    rows = []
    for _, r in df[df.base].iterrows():
        inf = info[r.w]
        if inf["food"]: continue
        if tier == "A" and not inf["instr"]: continue
        if tier == "B" and inf["lex"] != "noun.artifact": continue
        if r.topval >= 2.5 and r.margin >= m:
            rows.append(r)
    return pd.DataFrame(rows).drop_duplicates("w")

tries = {}
chosen = None
for tier, m in [("A", 1.0), ("A", 0.5), ("B", 1.0), ("B", 0.5)]:
    c = candidate(tier, m)
    counts = c["top"].value_counts().reindex(CLASSES).fillna(0).astype(int)
    tries[f"{tier},{m}"] = counts.to_dict()
    print(tier, m, counts.to_dict())
    if chosen is None and counts.min() >= 80:
        chosen = (tier, m, c)
if chosen is None:
    best = max(tries, key=lambda k: min(tries[k].values()))
    t, m = best.split(","); chosen = (t, float(m), candidate(t, float(m)))
tier, m, c = chosen
smallest = int(c["top"].value_counts().min())
cap = 2 * smallest
parts = []
for cl in CLASSES:
    parts.append(c[c.top == cl].sort_values("margin", ascending=False).head(cap))
words = pd.concat(parts)
rng = np.random.RandomState(0)
split = {}
for cl in CLASSES:
    ws = sorted(words[words.top == cl].w.tolist()); rng.shuffle(ws)
    k = int(round(len(ws) / 3))
    for j, w in enumerate(ws):
        split[w] = "selection" if j < k else "reporting"
words["split"] = words.w.map(split)
words = words[["w", "top", "topval", "margin", "split", "Hand_arm.mean", "Foot_leg.mean", "Mouth.mean"]]
words.to_csv(f"{OUT}/words.csv", index=False)

# extra non-reporting nouns for ladder rung 1: any category, margin 1.0, not in the main list
used = set(words.w)
extra = []
for cl in CLASSES:
    e = df[df.base & (df.top == cl) & (df.topval >= 2.5) & (df.margin >= 1.0) & (~df.w.isin(used))]
    # any category (food included), per the plan
    e = e.drop_duplicates("w").sort_values("margin", ascending=False).head(1000)
    extra.append(e[["w", "top", "topval", "margin"]])
extra = pd.concat(extra); extra.to_csv(f"{OUT}/extra.csv", index=False)
n_rep = int((words.split == "reporting").sum())
meta = dict(tier=tier, margin=m, tries=tries, counts=words.top.value_counts().to_dict(),
            n_selection=int((words.split == "selection").sum()), n_reporting=n_rep,
            extra_counts=extra.top.value_counts().to_dict(),
            # smallest detectable edge (80% power, two-sided 0.05) for a paired accuracy difference,
            # assuming 40% discordant nouns: 2.8*sqrt(0.4/n)
            mde=float(2.8 * np.sqrt(0.4 / n_rep)))
json.dump(meta, open(f"{OUT}/words_meta.json", "w"), indent=1)
print(json.dumps(meta, indent=1))
for cl in CLASSES:
    print(cl, words[words.top == cl].w.tolist()[:40])
print("extra sample", extra.groupby("top").head(8).w.tolist())
