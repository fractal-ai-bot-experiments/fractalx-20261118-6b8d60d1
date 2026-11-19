"""Step 2: hidden-state features from Qwen3-8B (raw text, no chat template), all 36 block outputs.
Every prompt ends at the token that is read (noun's last token, or 'the' for the use-eliciting prompt)."""
import sys, json, numpy as np, pandas as pd, torch
sys.path.insert(0, "/scratch")
from recipes.common import load_model

D = "/scratch/exp/data"
CARRIERS = ["Here is a word: {n}", "The next item on the list is the {n}", "She looked at the {n}",
            "He wrote down the word {n}", "In the picture there was the {n}", "They talked about the {n}",
            "On the table, next to the window, lay the {n}", "The catalogue entry read: {n}",
            "Yesterday I noticed the {n}", "This sentence mentions the {n}", "Somebody asked about the {n}",
            "The last word of this line is {n}"]
def art(n): return "An" if n[0] in "aeiou" else "A"
USE = "{A} {n} is something a person mainly operates with the"
PC = "Using only her {p}, she worked the {n}"
PARTS = {"hand": "hands", "foot": "feet", "mouth": "mouth"}
BODY = {
 "hand": ["hand", "finger", "thumb", "palm", "wrist", "knuckle", "fingernail", "fingertip", "arm", "elbow", "forearm",
          "fist", "shoulder", "bicep", "hands", "fingers", "thumbnail", "armpit", "pinky", "wrists", "elbows", "palms",
          "knuckles", "forefinger"],
 "foot": ["foot", "toe", "heel", "ankle", "sole", "leg", "knee", "shin", "calf", "thigh", "toenail", "instep",
          "kneecap", "feet", "toes", "heels", "ankles", "legs", "knees", "shins", "thighs", "hamstring", "kneecaps", "arch"],
 "mouth": ["mouth", "tongue", "lip", "lips", "tooth", "teeth", "gum", "gums", "jaw", "palate", "throat", "molar",
           "incisor", "saliva", "cheek", "chin", "tonsil", "uvula", "tongues", "mouths", "jaws", "molars", "larynx", "gullet"]}

tok, model = None, None
@torch.no_grad()
def run(texts, bs=192):
    out = np.zeros((len(texts), 36, 4096), dtype=np.float16)
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], return_tensors="pt", padding=True).to("cuda")
        hs = model(**b, output_hidden_states=True).hidden_states  # 37: emb + 36 blocks
        last = b["attention_mask"].sum(1) - 1
        idx = torch.arange(len(last), device="cuda")
        H = torch.stack([h[idx, last] for h in hs[1:]], 1)
        out[i:i + bs] = H.float().cpu().numpy().astype(np.float16)
    return out

def carrier_feats(nouns):
    T = [c.format(n=n) for n in nouns for c in CARRIERS]
    F = run(T).astype(np.float32).reshape(len(nouns), len(CARRIERS), 36, 4096).mean(1)
    return F.astype(np.float16)

if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    model, tok = load_model("/weights/qwen3-8b", dtype="bfloat16")
    tok.padding_side = "right"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model.eval()
    words = pd.read_csv(f"{D}/words.csv"); extra = pd.read_csv(f"{D}/extra.csv")
    if smoke:
        words = words.head(6); extra = extra.head(4)
    nouns = words.w.tolist()
    rng = np.random.RandomState(1)
    pc_lab = rng.choice(["hand", "foot", "mouth"], size=len(nouns))
    pref = "smoke_" if smoke else ""
    np.save(f"{D}/{pref}feat_a.npy", carrier_feats(nouns))
    np.save(f"{D}/{pref}feat_b.npy", run([USE.format(A=art(n), n=n) for n in nouns]))
    np.save(f"{D}/{pref}feat_c.npy", run([PC.format(p=PARTS[p], n=n) for n, p in zip(nouns, pc_lab)]))
    json.dump(list(pc_lab), open(f"{D}/{pref}pc_labels.json", "w"))
    bw = [(w, c) for c in BODY for w in BODY[c]]
    np.save(f"{D}/{pref}feat_body.npy", carrier_feats([w for w, _ in bw]))
    json.dump(bw, open(f"{D}/{pref}body_words.json", "w"))
    emb = model.get_input_embeddings().weight
    def inemb(ws):
        return np.stack([emb[tok(" " + w, add_special_tokens=False)["input_ids"]].float().mean(0).detach().cpu().numpy() for w in ws])
    np.save(f"{D}/{pref}emb_main.npy", inemb(nouns))
    np.save(f"{D}/{pref}emb_extra.npy", inemb(extra.w.tolist()))
    np.save(f"{D}/{pref}feat_a_extra.npy", carrier_feats(extra.w.tolist()))
    np.save(f"{D}/{pref}feat_b_extra.npy", run([USE.format(A=art(n), n=n) for n in extra.w]))
    # zero-training behavioural readout: next-token log-probs after the use-eliciting prompt
    cand = {"hand": [" hand", " hands", " fingers", " arm", " arms"], "foot": [" foot", " feet", " leg", " legs", " toes"],
            "mouth": [" mouth", " lips", " teeth", " tongue"]}
    ids = {c: [tok(t, add_special_tokens=False)["input_ids"][0] for t in v] for c, v in cand.items()}
    T = [USE.format(A=art(n), n=n) for n in nouns]
    L = []
    with torch.no_grad():
        for i in range(0, len(T), 128):
            b = tok(T[i:i+128], return_tensors="pt", padding=True).to("cuda")
            lg = model(**b).logits
            last = b["attention_mask"].sum(1) - 1
            lp = torch.log_softmax(lg[torch.arange(len(last)), last].float(), -1)
            L.append(torch.stack([torch.logsumexp(lp[:, ids[c]], -1) for c in ["hand", "foot", "mouth"]], 1).cpu().numpy())
    np.save(f"{D}/{pref}nexttok_logp.npy", np.concatenate(L))
    print("done", len(nouns), len(extra))
