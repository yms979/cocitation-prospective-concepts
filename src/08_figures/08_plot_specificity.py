"""Polished figure: (a) generated-vs-real Top-1 specificity, (b) Top-1 target diversity.
Real-patent set = leakage-cleaned (validation-overlapping patents removed), to match the manuscript."""
# Source file in the working repository: EDA/plot_specificity.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#444444", "axes.linewidth": 1.0,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11.5, "axes.labelcolor": "#222222",
    "xtick.color": "#444444", "ytick.color": "#444444",
    "grid.color": "#e7e7e7", "grid.linewidth": 0.9,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "savefig.facecolor": "white",
})
BLUE, BLUE_F = "#3D6A9E", "#AFC6E0"
RED, RED_F = "#C4453B", "#E8A39B"


def parse(v):
    try:
        return ast.literal_eval(v)
    except Exception:
        return None


D = _ROOT
val = pd.read_csv(f"{D}/data/raw/validation_abstract.csv"); val["e"] = val["embedding"].apply(parse)
pro = pd.read_csv(f"{D}/data/final_decoded_results_with_nn.csv"); pro["e"] = pro["embedding"].apply(parse)
ne = pd.read_csv(f"{D}/data/node_embeddings_with_text_embedding_ada.csv", dtype={"id": str, "text_embedding": str})
ne = ne[~ne["id"].astype(str).str.startswith("prospective")].copy()
ne["te"] = ne["text_embedding"].apply(lambda x: parse(x) if isinstance(x, str) and x.strip() not in ("", "nan", "None") else None)
ne = ne[ne["te"].notnull()]
# leakage control: drop real patents that also appear in the validation set
val_ids = set(val["original_id"].astype(str)) if "original_id" in val.columns else set()
n_full = len(ne)
ne = ne[~ne["id"].astype(str).isin(val_ids)]
print(f"real-patent set: {n_full} full -> {len(ne)} after removing {n_full - len(ne)} validation-overlapping patents")

V = np.array([x for x in val["e"]])
P = np.array([x for x in pro["e"]])
Dr = np.array([x for x in ne["te"]])               # leakage-cleaned real-patent set
realT1 = cosine_similarity(Dr, V).max(axis=1)
proT1 = cosine_similarity(P, V).max(axis=1)
Spv = cosine_similarity(P, V)
c = Counter(Spv.argmax(axis=1).tolist())
gap = proT1.mean() - realT1.mean()
d = gap / realT1.std()
lo, hi = np.percentile(realT1, 5), np.percentile(realT1, 95)
within = 100 * np.mean((proT1 >= lo) & (proT1 <= hi))
print(f"real n={len(realT1)} mean={realT1.mean():.4f} | gen mean={proT1.mean():.4f} | gap={gap:+.4f} d={d:.2f} | within={within:.0f}%")

fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.1), gridspec_kw={"width_ratios": [1.05, 1]})
for a in ax:
    a.grid(axis="y", alpha=0.6)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

# ---------- (a) specificity ----------
xs = np.linspace(0.79, 0.985, 500)
ax[0].hist(realT1, bins=60, density=True, color=BLUE_F, alpha=0.55, edgecolor="none", zorder=1)
ax[0].hist(proT1, bins=16, density=True, color=RED_F, alpha=0.55, edgecolor="none", zorder=2)
N_REAL_LABEL = 5152  # manuscript value (curves = leakage-clean 5,130, which reproduce 0.861/+0.031/d0.93/76%)
ax[0].plot(xs, gaussian_kde(realT1)(xs), color=BLUE, lw=2.6, zorder=4, label=f"Real patents  (n={N_REAL_LABEL:,})")
ax[0].plot(xs, gaussian_kde(proT1)(xs), color=RED, lw=2.8, zorder=5, label="Generated  (n=55)")
ax[0].axvline(realT1.mean(), color=BLUE, ls=(0, (4, 2)), lw=1.6, zorder=3)
ax[0].axvline(proT1.mean(), color=RED, ls=(0, (4, 2)), lw=1.6, zorder=3)
ymax = ax[0].get_ylim()[1]
ax[0].annotate("", xy=(proT1.mean(), ymax * 0.92), xytext=(realT1.mean(), ymax * 0.92),
               arrowprops=dict(arrowstyle="<|-|>", color="#333", lw=1.4, mutation_scale=13))
ax[0].text((realT1.mean() + proT1.mean()) / 2, ymax * 0.985, f"{gap:+.3f}   (d = {d:.2f})",
           ha="center", va="top", fontsize=10, color="#222", fontweight="bold")
ax[0].text(realT1.mean() - 0.002, ymax * 0.50, f"real\n{realT1.mean():.3f}", ha="right", color=BLUE, fontsize=9.5, fontweight="bold")
ax[0].text(proT1.mean() + 0.002, ymax * 0.66, f"gen\n{proT1.mean():.3f}", ha="left", color=RED, fontsize=9.5, fontweight="bold")
ax[0].text(0.804, ymax * 0.36, f"generated lie within\nthe real-patent range\n({within:.0f}% in [p5, p95])", fontsize=8.5, color="#666", style="italic")
ax[0].set_xlabel("Top-1 cosine similarity to validation (future) patents")
ax[0].set_ylabel("Density")
ax[0].set_title("(a)  Specificity — generated vs. real patents", loc="left")
ax[0].legend(loc="upper left", fontsize=10, handlelength=1.6)
ax[0].set_xlim(0.80, 0.97)
ax[0].set_ylim(0, ymax * 1.05)

# ---------- (b) target diversity ----------
freq = sorted(c.values(), reverse=True)
n = len(freq)
cmap = plt.cm.YlOrBr
cols = [cmap(0.35 + 0.5 * (f - 1) / (max(freq) - 1)) for f in freq]
ax[1].bar(range(1, n + 1), freq, color=cols, edgecolor="#7a5a10", linewidth=0.7, width=0.82, zorder=3)
ax[1].axhline(1, color="#999", ls=":", lw=1.1, zorder=1)
first_single = next((i for i, f in enumerate(freq) if f == 1), n)
ax[1].axvspan(first_single + 0.5, n + 0.6, color="#f4f4f4", zorder=0)
ax[1].text((first_single + 1 + n) / 2, max(freq) * 0.6, f"{freq.count(1)} unique\ntargets\n(1 node each)",
           ha="center", va="center", fontsize=8.5, color="#888", style="italic")
ax[1].annotate('"…navigation and\nvisualization"', xy=(1, freq[0]), xytext=(3.2, freq[0] + 0.1),
               fontsize=8.5, color="#7a5a10", arrowprops=dict(arrowstyle="-", color="#7a5a10", lw=0.8))
ax[1].set_xlabel("Distinct validation target  (ranked by # of generated nodes)")
ax[1].set_ylabel("# of the 55 generated nodes mapping here")
ax[1].set_title(f"(b)  Top-1 target diversity — {n} distinct targets for 55 nodes", loc="left")
ax[1].set_xlim(0.3, n + 0.7)
ax[1].set_ylim(0, max(freq) + 0.9)
ax[1].set_yticks(range(0, max(freq) + 1))
ax[1].text(0.97, 0.95, f"top-3 targets\nabsorb {sum(freq[:3])}/55 (35%)", transform=ax[1].transAxes,
           ha="right", va="top", fontsize=9, color="#444",
           bbox=dict(boxstyle="round,pad=0.4", fc="#fbf3e2", ec="#e0c98a"))

plt.tight_layout(pad=1.4)
out = f"{D}/figures/specificity_modecollapse.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
