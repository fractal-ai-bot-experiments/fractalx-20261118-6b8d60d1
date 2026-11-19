"""Follow-up (named after the main read, exploratory): does the edge over GloVe hold with the NEUTRAL carriers
alone (prompt a at its own best selection layer), i.e. without a prompt that asks the model for the body part?
Same probe, same folds, paired bootstrap against the same GloVe predictions. Writes out/followup.json."""
import sys, json, numpy as np, pandas as pd
sys.argv = [sys.argv[0]]
sys.path.insert(0, "/scratch/exp")
import analyze as A
D = A.D
words = pd.read_csv(f"{D}/words.csv"); y = words.top.map({c: i for i, c in enumerate(A.CL)}).values
sel = np.where(words.split == "selection")[0]; rep = np.where(words.split == "reporting")[0]
det = json.load(open("/scratch/exp/out/details.json"))
pr, L, C = det["own_best"]["a"]; L = int(L); C = float(C)
Fa = np.load(f"{D}/feat_a.npy")[:, L].astype(np.float32)
pn = pd.read_csv("/scratch/exp/out/per_noun_predictions.csv")
out = {}
pa = {}
for sd in [0, 1, 2]:
    folds = A.strat_folds(rep, y, 10, sd)
    pa[sd] = A.cv_preds(Fa, y, sel, folds, [C])[C][rep]
pg = {sd: pn[f"glove_seed{sd}"].values for sd in [0, 1, 2]}
ph = {sd: pn[f"hidden_seed{sd}"].values for sd in [0, 1, 2]}
yy = y[rep]
def boot(p1, p2, nb=10000):
    rng = np.random.RandomState(0); I = rng.randint(0, len(yy), (nb, len(yy))); Y = yy[I]
    def ba(c):
        return np.mean([(c[I] * (Y == k)).sum(1) / np.maximum((Y == k).sum(1), 1) for k in range(3)], 0)
    d = np.mean([ba((p1[s] == yy).astype(float)) - ba((p2[s] == yy).astype(float)) for s in [0, 1, 2]], 0)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
out["neutral_layer"] = L; out["neutral_C"] = C
out["neutral_carrier_acc_mean"] = float(np.mean([A.bal_acc(yy, pa[s]) for s in pa]))
out["neutral_minus_glove_mean"] = out["neutral_carrier_acc_mean"] - float(np.mean([A.bal_acc(yy, pg[s]) for s in pg]))
out["neutral_minus_glove_ci"] = boot(pa, pg)
# per-class recall of the main hidden probe and the GloVe probe (seed-mean)
for name, p in [("hidden", ph), ("glove", pg), ("neutral", pa)]:
    for k, c in enumerate(A.CL):
        out[f"{name}_recall_{c}"] = float(np.mean([np.mean(p[s][yy == k] == k) for s in p]))
# nouns GloVe misses and the hidden probe gets (seed 0)
m = (ph[0] == yy) & (pg[0] != yy)
out["examples_hidden_right_glove_wrong"] = [f"{w}:{A.CL[l]}->glove {A.CL[g]}" for w, l, g in zip(pn.word[m], yy[m], pg[0][m])][:25]
m2 = (ph[0] != yy) & (pg[0] == yy)
out["examples_glove_right_hidden_wrong"] = [f"{w}:{A.CL[l]}->hidden {A.CL[h]}" for w, l, h in zip(pn.word[m2], yy[m2], ph[0][m2])][:25]
json.dump(out, open("/scratch/exp/out/followup.json", "w"), indent=1)
print(json.dumps(out, indent=1))
