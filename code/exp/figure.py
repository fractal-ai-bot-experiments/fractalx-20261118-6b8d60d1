import sys, json, numpy as np
sys.path.insert(0, "/scratch")
import fractal_plot as fp
R = json.load(open("/scratch/results.json")); det = json.load(open("/scratch/exp/out/details.json"))
fig, axes = fp.figure("double", height=3.2, ncols=2)
ax = axes[0]
order = ["body_part_probe_accuracy", "glove_probe", "glove_plus_hidden", "hidden_pca300", "input_embeddings",
         "frequency_only", "shuffled_labels", "anchor_cosine", "part_words_transfer", "context_named_part"]
for i, k in enumerate(order):
    d = R["per_seed"] if k == "body_part_probe_accuracy" else R["controls"][k]
    v = np.array(list(d.values()), float)
    ax.scatter(v, np.full(len(v), i) + np.array([-0.15, 0, 0.15])[:len(v)], s=8, marker="o", alpha=0.8, color="C0" if i == 0 else "0.3", zorder=3)
    ax.vlines(v.mean(), i - 0.3, i + 0.3, color="k", lw=1.2)
ax.axvline(1 / 3, ls=":", color="0.5", lw=0.8)
ax.set_yticks(range(len(order))); ax.set_yticklabels(["Hidden-state probe, Qwen3-8B\n(main condition)"] + [fp.label(k, width=30) for k in order[1:]], fontsize=6)
ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("Balanced accuracy on reporting nouns")
ax.set_title("Reporting nouns (n=255), 3 fold seeds", fontsize=7)
ax = axes[1]
for pr, mk, name in [("a", "o", "Neutral carriers"), ("b", "s", "Use-eliciting prompt")]:
    ys = [max(det["grid"][f"{pr}|{L}|{C}"] for C in [0.0001, 0.001, 0.01, 0.1, 1.0]) for L in range(36)]
    ax.plot(range(36), ys, marker=mk, ms=2.5, lw=0.8, label=name)
ax.axhline(R["other_numbers"]["glove_probe_selection_cv"], ls="--", color="0.4", lw=0.8)
ax.text(1, R["other_numbers"]["glove_probe_selection_cv"] - 0.04, "GloVe probe", fontsize=6, color="0.3")
ax.axhline(1 / 3, ls=":", color="0.5", lw=0.8)
ax.set_ylim(0, 1); ax.set_xlabel("Layer (block output)"); ax.set_ylabel("Selection CV balanced accuracy")
ax.set_title("Selection nouns (n=127), best C per layer", fontsize=7)
ax.legend(frameon=False, loc="lower right", fontsize=6)
fp.save(fig, "body_part_probe")
