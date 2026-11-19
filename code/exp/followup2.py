"""Follow-up 2 (exploratory, chosen after the main read): is the gap an artefact of GloVe being a weak word
vector? Same probe, same C selection on selection nouns, same reporting folds, on fastText wiki-news 300d
(subword-trained, 1M words). Nouns without a fastText vector get a zero vector (count reported).
Paired bootstrap of the main hidden-state predictions against fastText."""
import sys, json, numpy as np, pandas as pd
sys.argv = [sys.argv[0]]
sys.path.insert(0, "/scratch/exp")
import analyze as A
import gensim.downloader as api
D = A.D
words = pd.read_csv(f"{D}/words.csv"); y = words.top.map({c: i for i, c in enumerate(A.CL)}).values
sel = np.where(words.split == "selection")[0]; rep = np.where(words.split == "reporting")[0]
ft = api.load("fasttext-wiki-news-subwords-300")
miss = [w for w in words.w if w not in ft.key_to_index]
X = np.stack([ft[w] if w in ft.key_to_index else np.zeros(300, np.float32) for w in words.w])
s = A.select_cv(X, y, sel, A.CS); C = A.pick([(v, (0, 0, c), c) for c, v in s.items()])[2]
pf = {sd: A.cv_preds(X, y, sel, A.strat_folds(rep, y, 10, sd), [C])[C][rep] for sd in [0, 1, 2]}
pn = pd.read_csv("/scratch/exp/out/per_noun_predictions.csv")
ph = {sd: pn[f"hidden_seed{sd}"].values for sd in [0, 1, 2]}
yy = y[rep]
rng = np.random.RandomState(0); I = rng.randint(0, len(yy), (10000, len(yy))); Y = yy[I]
def ba(c): return np.mean([(c[I] * (Y == k)).sum(1) / np.maximum((Y == k).sum(1), 1) for k in range(3)], 0)
d = np.mean([ba((ph[t] == yy).astype(float)) - ba((pf[t] == yy).astype(float)) for t in [0, 1, 2]], 0)
out = dict(fasttext_missing=len(miss), fasttext_C=C, fasttext_selection_cv=s,
           fasttext_acc_mean=float(np.mean([A.bal_acc(yy, pf[t]) for t in pf])),
           fasttext_acc_per_seed=[A.bal_acc(yy, pf[t]) for t in [0, 1, 2]],
           hidden_minus_fasttext_mean=float(np.mean([A.bal_acc(yy, ph[t]) - A.bal_acc(yy, pf[t]) for t in [0, 1, 2]])),
           hidden_minus_fasttext_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
json.dump(out, open("/scratch/exp/out/followup2.json", "w"), indent=1); print(json.dumps(out, indent=1))
