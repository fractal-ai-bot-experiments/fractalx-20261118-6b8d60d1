"""GloVe 300d vectors and log word frequency (wordfreq Zipf) for the main, extra and body-part words."""
import json, numpy as np, pandas as pd
import gensim.downloader as api
from wordfreq import zipf_frequency
D = "/scratch/exp/data"
g = api.load("glove-wiki-gigaword-300")
words = pd.read_csv(f"{D}/words.csv"); extra = pd.read_csv(f"{D}/extra.csv")
np.save(f"{D}/glove_main.npy", np.stack([g[w] for w in words.w]))
np.save(f"{D}/glove_extra.npy", np.stack([g[w] for w in extra.w]))
np.save(f"{D}/freq_main.npy", np.array([[zipf_frequency(w, "en")] for w in words.w]))
np.save(f"{D}/freq_extra.npy", np.array([[zipf_frequency(w, "en")] for w in extra.w]))
import sys; sys.path.insert(0, "/scratch/exp")
from extract import BODY
bw = [w for c in BODY for w in BODY[c]]
np.save(f"{D}/glove_body.npy", np.stack([g[w] if w in g.key_to_index else np.zeros(300, np.float32) for w in bw]))
print("missing body glove", [w for w in bw if w not in g.key_to_index])
