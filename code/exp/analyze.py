"""Steps 3-7 (probe solver: scikit-learn; an earlier torch L-BFGS version was too slow): probes, selection on selection nouns, one reporting read, controls, ladder rungs.
Probe = standardise -> rotate into the training rows' span (SVD; exact for L2) -> L2 multinomial logistic
regression with balanced class weights (scikit-learn)."""
import sys, json, time, numpy as np, pandas as pd, torch
sys.path.insert(0, "/scratch")
D = "/scratch/exp/data"; OUT = "/scratch/exp/out"
import os; os.makedirs(OUT, exist_ok=True)
CL = ["hand", "foot", "mouth"]; CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
dev = "cuda"

def bal_acc(y, p, k=3):
    return float(np.mean([np.mean(p[y == c] == c) for c in range(k) if np.any(y == c)]))

def lr_fit_multi(Ztr, ytr, Zte, Cs):
    """scikit-learn LogisticRegression (lbfgs, multinomial, L2, balanced class weights) per C. Returns {C: (pred, prob)}."""
    from sklearn.linear_model import LogisticRegression
    Ztr = Ztr.cpu().numpy() if torch.is_tensor(Ztr) else np.asarray(Ztr)
    Zte = Zte.cpu().numpy() if torch.is_tensor(Zte) else np.asarray(Zte)
    out = {}
    for C in Cs:
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000 if len(ytr) < 1000 else 500).fit(Ztr, ytr)
        out[C] = (m.predict(Zte), m.predict_proba(Zte))
    return out

def lr_fit_predict(Ztr, ytr, Zte, C):
    return lr_fit_multi(Ztr, ytr, Zte, [C])[C]

def prep(Xtr, Xte, pca=None):
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32, device=dev); Xte = torch.as_tensor(Xte, dtype=torch.float32, device=dev)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    if pca is not None:  # PCA on the training fold (centred, standardised)
        U, S, Vt = torch.linalg.svd(A, full_matrices=False)
        V = Vt[:pca].T; A = A @ V; B = B @ V
        mu2 = A.mean(0); sd2 = A.std(0) + 1e-9; A = (A - mu2) / sd2; B = (B - mu2) / sd2
        return A, B
    if A.shape[1] > A.shape[0]:  # exact rotation into the training rows' span
        U, S, Vt = torch.linalg.svd(A, full_matrices=False)
        keep = S > S[0] * 1e-10
        V = Vt[keep].T; return A @ V, B @ V
    return A, B

def cv_preds(X, y, tr_idx_fixed, folds, Cs, pca=None, Xextra=None, yextra=None):
    """folds: list of test index arrays (into X). Training = tr_idx_fixed + other folds (+ extra). Returns preds per C."""
    P = {C: np.full(len(y), -1) for C in Cs}
    for te in folds:
        tr = np.setdiff1d(np.concatenate([tr_idx_fixed] + [f for f in folds]), te)
        Xtr, ytr = X[tr], y[tr]
        if Xextra is not None:
            Xtr = np.concatenate([Xtr, Xextra]); ytr = np.concatenate([ytr, yextra])
        A, B = prep(Xtr, X[te], pca)
        R = lr_fit_multi(A, ytr, B, Cs)
        for C in Cs:
            P[C][te] = R[C][0]
    return P

def strat_folds(idx, y, k, seed):
    rng = np.random.RandomState(seed); f = [[] for _ in range(k)]
    for c in range(3):
        ii = idx[y[idx] == c].copy(); rng.shuffle(ii)
        off = rng.randint(k)
        for j, i in enumerate(ii): f[(j + off) % k].append(i)
    return [np.array(sorted(x), dtype=int) for x in f]

def select_cv(X, y, sel, Cs, pca=None, Xextra=None, yextra=None, repeats=3):
    """5-fold x repeats CV on selection nouns only; returns mean balanced accuracy per C."""
    sc = {C: [] for C in Cs}
    for r in range(repeats):
        folds = strat_folds(sel, y, 5, 100 + r)
        P = cv_preds(X, y, np.array([], int), folds, Cs, pca, Xextra, yextra)
        for C in Cs: sc[C].append(bal_acc(y[sel], P[C][sel]))
    return {C: float(np.mean(v)) for C, v in sc.items()}

def _r1_job(X, y, sel, Cs, Xe, ye):
    global dev
    dev = "cpu"
    import os; os.environ["OMP_NUM_THREADS"] = "4"
    return select_cv(X, y, sel, Cs, Xextra=Xe, yextra=ye, repeats=1)

def pick(cands, tol=0.02):
    """cands: list of (score, prefkey tuple, name); best score, then within tol the lowest prefkey."""
    best = max(c[0] for c in cands)
    ok = [c for c in cands if c[0] >= best - tol]
    return min(ok, key=lambda c: c[1])

if __name__ == "__main__":
    t0 = time.time()
    if "--check" in sys.argv:
        from sklearn.linear_model import LogisticRegression
        rng = np.random.RandomState(0); X = rng.randn(150, 500); y = rng.randint(0, 3, 150); X[:, 0] += y
        A, B = prep(X[:100], X[100:])
        R = lr_fit_multi(A, y[:100], B, [1e-3, 1e-1, 1])
        for C in [1e-3, 1e-1, 1]:
            p1 = R[C][1]
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, tol=1e-10).fit(A.cpu().numpy(), y[:100])
            print(C, "max prob diff vs sklearn", np.abs(p1 - m.predict_proba(B.cpu().numpy())).max())
        sys.exit()
    words = pd.read_csv(f"{D}/words.csv"); extra = pd.read_csv(f"{D}/extra.csv")
    y = words.top.map({c: i for i, c in enumerate(CL)}).values
    ye = extra.top.map({c: i for i, c in enumerate(CL)}).values
    sel = np.where(words.split == "selection")[0]; rep = np.where(words.split == "reporting")[0]
    Fa = np.load(f"{D}/feat_a.npy"); Fb = np.load(f"{D}/feat_b.npy")
    Fc = np.load(f"{D}/feat_c.npy"); Fbody = np.load(f"{D}/feat_body.npy")
    G = np.load(f"{D}/glove_main.npy"); Ge = np.load(f"{D}/glove_extra.npy"); Gbody = np.load(f"{D}/glove_body.npy")
    E = np.load(f"{D}/emb_main.npy"); Fq = np.load(f"{D}/freq_main.npy")
    yc = np.array([CL.index(c) for c in json.load(open(f"{D}/pc_labels.json"))])
    body = json.load(open(f"{D}/body_words.json")); yb = np.array([CL.index(c) for _, c in body])
    NT = np.load(f"{D}/nexttok_logp.npy")
    feats = {"a": Fa, "b": Fb}
    SEEDS = [0, 1, 2]
    res = {"selection_grid": {}}

    # ---- selection (selection nouns only) over layer x prompt x C
    grid = {}
    for pr in ["a", "b"]:
        for L in range(36):
            s = select_cv(feats[pr][:, L].astype(np.float32), y, sel, CS)
            for C in CS: grid[(pr, L, C)] = s[C]
        print("selected grid prompt", pr, "max", max(v for k, v in grid.items() if k[0] == pr), f"{time.time()-t0:.0f}s", flush=True)
    cands = [(v, (0 if k[0] == "a" else 1, k[1], k[2]), k) for k, v in grid.items()]
    best = pick(cands)[2]; PR, LAY, CH = best
    print("chosen", best, grid[best])
    # each prompt at its own best layer
    own = {pr: pick([c for c in cands if c[2][0] == pr])[2] for pr in ["a", "b"]}

    def sel_C(X, pca=None, Xextra=None, yextra=None):
        s = select_cv(X, y, sel, CS, pca, Xextra, yextra)
        return pick([(v, (0, 0, C), C) for C, v in s.items()])[2], s

    def report(X, C, yy=y, pca=None, Xextra=None, yextra=None):
        out = {}
        for sd in SEEDS:
            folds = strat_folds(rep, yy, 10, sd)
            P = cv_preds(X, yy, sel, folds, [C], pca, Xextra, yextra)[C]
            out[sd] = P[rep]
        return out

    H = feats[PR][:, LAY].astype(np.float32)
    Cg, sg = sel_C(G)
    preds = {"hidden": report(H, CH), "glove": report(G, Cg)}
    sc = {k: {sd: bal_acc(y[rep], p) for sd, p in v.items()} for k, v in preds.items()}

    def boot_diff(pa, pb, yy, nb=10000, seed=0):
        rng = np.random.RandomState(seed); n = len(yy)
        I = rng.randint(0, n, (nb, n)); Y = yy[I]
        def ba(corr):
            r = []
            for c in range(3):
                m = (Y == c); r.append((corr[I] * m).sum(1) / np.maximum(m.sum(1), 1))
            return np.mean(r, 0)
        d = np.mean([ba((pa[s] == yy).astype(float)) - ba((pb[s] == yy).astype(float)) for s in SEEDS], 0)
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    diff = {sd: sc["hidden"][sd] - sc["glove"][sd] for sd in SEEDS}
    ci = boot_diff(preds["hidden"], preds["glove"], y[rep])
    print("hidden", sc["hidden"], "glove", sc["glove"], "diff CI", ci, flush=True)
    controls = {"glove_probe": sc["glove"]}
    other = {"hidden_minus_glove_mean": float(np.mean(list(diff.values()))),
             "hidden_minus_glove_ci_low": ci[0], "hidden_minus_glove_ci_high": ci[1],
             "hidden_probe_selection_cv": grid[best], "glove_probe_selection_cv": sg[Cg]}
    per_noun = pd.DataFrame({"word": words.w.values[rep], "label": y[rep]})
    for k in preds:
        for sd in SEEDS: per_noun[f"{k}_seed{sd}"] = preds[k][sd]

    # ---- controls
    Ce, se = sel_C(E); controls["input_embeddings"] = {sd: bal_acc(y[rep], p) for sd, p in report(E, Ce).items()}
    Cp, sp = sel_C(H, pca=300); controls["hidden_pca300"] = {sd: bal_acc(y[rep], p) for sd, p in report(H, Cp, pca=300).items()}
    GH = np.concatenate([G, H], 1); Cgh, _ = sel_C(GH)
    pgh = report(GH, Cgh); controls["glove_plus_hidden"] = {sd: bal_acc(y[rep], p) for sd, p in pgh.items()}
    ci_gh = boot_diff(pgh, preds["glove"], y[rep])
    other["glove_plus_hidden_minus_glove_ci_low"], other["glove_plus_hidden_minus_glove_ci_high"] = ci_gh
    Cf, _ = sel_C(Fq); controls["frequency_only"] = {sd: bal_acc(y[rep], p) for sd, p in report(Fq, Cf).items()}
    print("controls done", f"{time.time()-t0:.0f}s", flush=True)
    # positive control: labels = randomly assigned named part
    Hc = Fc[:, LAY].astype(np.float32)
    s_pc = select_cv(Hc, yc, sel, CS); Cpc = pick([(v, (0, 0, C), C) for C, v in s_pc.items()])[2]
    controls["context_named_part"] = {sd: bal_acc(yc[rep], p) for sd, p in report(Hc, Cpc, yy=yc).items()}
    s_pcg = select_cv(G, yc, sel, CS); Cpcg = pick([(v, (0, 0, C), C) for C, v in s_pcg.items()])[2]
    pcg = {sd: bal_acc(yc[rep], p) for sd, p in report(G, Cpcg, yy=yc).items()}
    other["glove_on_named_part_labels_mean"] = float(np.mean(list(pcg.values())))
    # shuffled labels: 200 shuffles in total (67/67/66 over the 3 fold assignments), hidden probe at chosen setting
    shuf = {}
    allidx = np.concatenate([sel, rep]); nsh = {0: 67, 1: 67, 2: 66}
    for sd in SEEDS:
        vals = []
        for k in range(nsh[sd]):
            rng = np.random.RandomState(1000 * sd + k); ys = y.copy(); ys[allidx] = rng.permutation(y[allidx])
            folds = strat_folds(rep, ys, 10, sd)
            P = cv_preds(H, ys, sel, folds, [CH])[CH]; vals.append(bal_acc(ys[rep], P[rep]))
        shuf[sd] = float(np.mean(vals)); other[f"shuffled_labels_p95_seed{sd}"] = float(np.percentile(vals, 95))
    controls["shuffled_labels"] = shuf
    print("shuffles done", f"{time.time()-t0:.0f}s", flush=True)
    # anchor-word cosine (no training): prompt (a) at its own best layer
    La = own["a"][1]
    anchors = {"hand": ["hand", "arm", "hands", "fingers"], "foot": ["foot", "leg", "feet", "legs"],
               "mouth": ["mouth", "lips", "tongue", "teeth"]}
    bidx = {w: i for i, (w, _) in enumerate(body)}
    def anchor_acc(X, Xb):
        mu = X.mean(0); Xc = X - mu
        cent = np.stack([np.mean([Xb[bidx[w]] for w in anchors[c]], 0) - mu for c in CL])
        cs = (Xc / np.linalg.norm(Xc, axis=1, keepdims=True)) @ (cent / np.linalg.norm(cent, axis=1, keepdims=True)).T
        return bal_acc(y[rep], cs[rep].argmax(1))
    a_h = anchor_acc(Fa[:, La].astype(np.float32), Fbody[:, La].astype(np.float32))
    controls["anchor_cosine"] = {sd: a_h for sd in SEEDS}
    other["anchor_cosine_glove"] = anchor_acc(G, Gbody)
    # tool-trained probe (prompt a, its own best layer, its C) on body-part words
    Ca = own["a"][2]; Xa = Fa[:, La].astype(np.float32)
    A_, B_ = prep(Xa, Fbody[:, La].astype(np.float32)); pb = lr_fit_predict(A_, y, B_, Ca)[0]
    controls["part_words_transfer"] = {sd: bal_acc(yb, pb) for sd in SEEDS}
    A_, B_ = prep(G, Gbody); Cg2 = Cg; pbg = lr_fit_predict(A_, y, B_, Cg2)[0]
    other["part_words_transfer_glove"] = bal_acc(yb, pbg)
    pd.DataFrame({"word": [w for w, _ in body], "label": yb, "pred_hidden": pb, "pred_glove": pbg}).to_csv(f"{OUT}/body_words_preds.csv", index=False)
    # each prompt at its own best selection layer
    for pr in ["a", "b"]:
        _, L_, C_ = own[pr]
        v = {sd: bal_acc(y[rep], p) for sd, p in report(feats[pr][:, L_].astype(np.float32), C_).items()}
        other[f"prompt_{pr}_own_best_layer"] = float(L_); other[f"prompt_{pr}_reporting_mean"] = float(np.mean(list(v.values())))
    # zero-training behavioural readout (exploratory): which part the model names next after the use prompt
    other["next_token_readout"] = bal_acc(y[rep], NT[rep].argmax(1))
    print("main analysis done", f"{time.time()-t0:.0f}s", flush=True)

    ladder = {}
    # ---- rung 1: extra non-reporting nouns as training data for both probes; reselect on selection nouns
    Hex = {"a": np.load(f"{D}/feat_a_extra.npy"), "b": np.load(f"{D}/feat_b_extra.npy")}
    grid1 = {}
    from joblib import Parallel, delayed
    R1_LAYERS = sorted(set(list(range(0, 36, 6)) + [LAY, own["a"][1], own["b"][1]]))
    R1_CS = [1e-4, 1e-3, 1e-2, 1e-1]
    jobs = [(pr, L) for pr in ["a", "b"] for L in R1_LAYERS]
    outs = Parallel(n_jobs=len(jobs), backend="loky")(delayed(_r1_job)(feats[pr][:, L].astype(np.float32), y, sel, R1_CS,
                    Hex[pr][:, L].astype(np.float32), ye) for pr, L in jobs)
    for (pr, L), s in zip(jobs, outs):
        for C in R1_CS: grid1[(pr, L, C)] = s[C]
    c1 = [(v, (0 if k[0] == "a" else 1, k[1], k[2]), k) for k, v in grid1.items()]
    b1 = pick(c1)[2]
    H1 = feats[b1[0]][:, b1[1]].astype(np.float32); X1e = Hex[b1[0]][:, b1[1]].astype(np.float32)
    s1g = select_cv(G, y, sel, R1_CS, Xextra=Ge, yextra=ye, repeats=1); C1g = pick([(v, (0, 0, C), C) for C, v in s1g.items()])[2]
    p1h = report(H1, b1[2], Xextra=X1e, yextra=ye); p1g = report(G, C1g, Xextra=Ge, yextra=ye)
    r1h = {sd: bal_acc(y[rep], p) for sd, p in p1h.items()}; r1g = {sd: bal_acc(y[rep], p) for sd, p in p1g.items()}
    ci1 = boot_diff(p1h, p1g, y[rep])
    ladder["rung1_extra_nouns"] = dict(hidden=r1h, glove=r1g, ci=ci1, chosen=list(map(str, b1)), sel_cv=grid1[b1])
    print("rung1", ladder["rung1_extra_nouns"], f"{time.time()-t0:.0f}s", flush=True)

    # ---- rung 2: one-hidden-layer MLP on PCA-100 (base training data), weight decay chosen on selection nouns
    def mlp_fit_predict(Xtr, ytr, Xte, wd, seed):
        A, B = prep(Xtr, Xte, pca=100); A = A.float(); B = B.float()
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(100, 128), torch.nn.ReLU(), torch.nn.Linear(128, 3)).to(dev)
        yt = torch.as_tensor(ytr, device=dev); cnt = torch.bincount(yt, minlength=3).float()
        wcls = len(ytr) / (3 * cnt.clamp(min=1))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
        for _ in range(400):
            opt.zero_grad(); l = torch.nn.functional.cross_entropy(net(A), yt, weight=wcls); l.backward(); opt.step()
        with torch.no_grad(): return net(B).argmax(1).cpu().numpy()
    def mlp_cv(X, folds, fixed, wd, seed):
        P = np.full(len(y), -1)
        for te in folds:
            tr = np.setdiff1d(np.concatenate([fixed] + folds), te)
            P[te] = mlp_fit_predict(X[tr], y[tr], X[te], wd, seed)
        return P
    WDS = [1e-2, 1e-1, 1.0, 10.0]
    def mlp_select(X):
        s = {}
        for wd in WDS:
            v = []
            for r in range(3):
                folds = strat_folds(sel, y, 5, 100 + r); P = mlp_cv(X, folds, np.array([], int), wd, r); v.append(bal_acc(y[sel], P[sel]))
            s[wd] = float(np.mean(v))
        return pick([(v, (0, 0, -wd), wd) for wd, v in s.items()])[2], s
    wdh, sh2 = mlp_select(H); wdg, sg2 = mlp_select(G)
    p2h = {sd: mlp_cv(H, strat_folds(rep, y, 10, sd), sel, wdh, sd)[rep] for sd in SEEDS}
    p2g = {sd: mlp_cv(G, strat_folds(rep, y, 10, sd), sel, wdg, sd)[rep] for sd in SEEDS}
    r2h = {sd: bal_acc(y[rep], p) for sd, p in p2h.items()}; r2g = {sd: bal_acc(y[rep], p) for sd, p in p2g.items()}
    ci2 = boot_diff(p2h, p2g, y[rep])
    ladder["rung2_mlp_pca100"] = dict(hidden=r2h, glove=r2g, ci=ci2, wd_hidden=wdh, wd_glove=wdg, sel_hidden=sh2, sel_glove=sg2)
    print("rung2", ladder["rung2_mlp_pca100"], f"{time.time()-t0:.0f}s", flush=True)
    for k, (ph, pg) in {"rung1": (p1h, p1g), "rung2": (p2h, p2g)}.items():
        for sd in SEEDS: per_noun[f"{k}_hidden_seed{sd}"] = ph[sd]; per_noun[f"{k}_glove_seed{sd}"] = pg[sd]
    per_noun.to_csv(f"{OUT}/per_noun_predictions.csv", index=False)

    # ---- write results
    from recipes.seeds import write_results
    for name, rr in ladder.items():
        other[f"{name}_hidden_mean"] = float(np.mean(list(rr["hidden"].values())))
        other[f"{name}_glove_mean"] = float(np.mean(list(rr["glove"].values())))
        other[f"{name}_diff_mean"] = other[f"{name}_hidden_mean"] - other[f"{name}_glove_mean"]
        other[f"{name}_diff_ci_low"], other[f"{name}_diff_ci_high"] = rr["ci"]
    other["rung1_selection_cv"] = ladder["rung1_extra_nouns"]["sel_cv"]
    other["rung1_chosen_layer"] = float(eval(ladder["rung1_extra_nouns"]["chosen"][1]))
    other["chosen_layer"] = float(LAY); other["chosen_C"] = CH; other["chosen_prompt_is_use_eliciting"] = float(PR == "b")
    for sd in SEEDS: other[f"hidden_minus_glove_seed{sd}"] = diff[sd]
    meta = json.load(open(f"{D}/words_meta.json"))
    setup = {"tool nouns": len(words), "selection nouns": len(sel), "reporting nouns": len(rep),
             "smallest class": int(min(meta["counts"].values())), "extra training nouns": len(extra),
             "layers": 36, "carrier sentences": 12, "bootstrap resamples": 10000, "label shuffles": 200,
             "smallest detectable edge": meta["mde"]}
    selection = {"criterion": "selection_cv_accuracy",
                 "candidates": {f"prompt={k[0]} layer={k[1]} C={k[2]}": v for k, v in grid.items()},
                 "chosen": f"prompt={PR} layer={LAY} C={CH}",
                 "selection_items": words.w.values[sel].tolist(), "report_items": words.w.values[rep].tolist()}
    write_results("body_part_probe_accuracy", {str(k): v for k, v in sc["hidden"].items()},
                  {k: {str(s): x for s, x in v.items()} for k, v in controls.items()},
                  other_numbers=other, setup=setup, selection=selection)
    json.dump({"ladder": ladder, "own_best": {k: list(map(str, v)) for k, v in own.items()},
               "grid": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in grid.items()},
               "grid_rung1": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in grid1.items()}},
              open(f"{OUT}/details.json", "w"), indent=1, default=str)
    print("ALL DONE", f"{time.time()-t0:.0f}s")
