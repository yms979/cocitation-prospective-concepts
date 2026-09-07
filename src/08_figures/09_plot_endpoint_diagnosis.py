"""Diagnosis figure: does endpoint sharing explain the partial mode collapse?"""
# Source file in the working repository: EDA/plot_endpoint_diagnosis.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import ast, re, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#444", "axes.linewidth": 1.0,
    "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.labelsize": 11, "grid.color": "#e7e7e7", "grid.linewidth": 0.9,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
    "savefig.facecolor": "white",
})
SH, NS = "#C4453B", "#6E8FB5"   # shared / not-shared


def parse(v):
    try:
        return ast.literal_eval(v)
    except Exception:
        return None


D = _ROOT
val = pd.read_csv(f"{D}/data/raw/validation_abstract.csv"); val["e"] = val["embedding"].apply(parse)
fd = pd.read_csv(f"{D}/data/final_decoded_results_with_nn.csv"); fd["e"] = fd["embedding"].apply(parse)
pl = pd.read_csv(f"{D}/data/predicted_new_links.csv", dtype=str)
V = np.array([x for x in val["e"]]); P = np.array([x for x in fd["e"]])
rows = [int(re.search(r"row (\d+)", s).group(1)) for s in fd["node_id"]]
eps = [(pl.iloc[r]["node1"], pl.iloc[r]["node2"]) for r in rows]
top1 = cosine_similarity(P, V).argmax(axis=1)
S = cosine_similarity(P)

sh_cos, ns_cos, sh_same, ns_same = [], [], [], []
for i, j in itertools.combinations(range(55), 2):
    shared = len(set(eps[i]) & set(eps[j])) > 0
    (sh_cos if shared else ns_cos).append(S[i, j])
    (sh_same if shared else ns_same).append(int(top1[i] == top1[j]))
real_div = 0.788  # mean pairwise cosine of 55 random real patents (computed earlier)

fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})
for a in ax:
    a.grid(axis="y", alpha=0.6)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

# (a) pairwise cosine: shared vs not
parts = ax[0].violinplot([ns_cos, sh_cos], positions=[0, 1], showmeans=False, showextrema=False, widths=0.8)
for pc, col in zip(parts["bodies"], [NS, SH]):
    pc.set_facecolor(col); pc.set_alpha(0.45); pc.set_edgecolor(col); pc.set_linewidth(1.2)
for x, data, col in [(0, ns_cos, NS), (1, sh_cos, SH)]:
    m = np.mean(data)
    ax[0].plot([x - 0.18, x + 0.18], [m, m], color=col, lw=2.6)
    ax[0].text(x, m + 0.006, f"{m:.3f}", ha="center", color=col, fontweight="bold", fontsize=10.5)
ax[0].axhline(real_div, color="#666", ls=(0, (5, 3)), lw=1.4)
ax[0].text(1.46, real_div, "real-patent\ndiversity 0.788", ha="right", va="center", fontsize=8.5, color="#666", style="italic")
ax[0].set_xticks([0, 1])
ax[0].set_xticklabels([f"No shared\nendpoint\n(n={len(ns_cos):,})", f"Shares ≥1\nendpoint\n(n={len(sh_cos)})"])
ax[0].set_ylabel("Pairwise cosine between generated embeddings")
ax[0].set_title("(a)  Shared-parent pairs are more alike", loc="left")
ax[0].set_ylim(0.74, 0.97)

# (b) same Top-1 target rate
r_ns, r_sh = 100 * np.mean(ns_same), 100 * np.mean(sh_same)
ax[1].bar([0, 1], [r_ns, r_sh], color=[NS, SH], width=0.6, edgecolor="#333", linewidth=0.8)
for x, r in [(0, r_ns), (1, r_sh)]:
    ax[1].text(x, r + 0.4, f"{r:.1f}%", ha="center", fontweight="bold", fontsize=11)
ax[1].annotate(f"{r_sh/r_ns:.1f}x", xy=(1, r_sh), xytext=(0.5, r_sh + 2.2), ha="center",
               fontsize=12, fontweight="bold", color="#333",
               arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(["No shared\nendpoint", "Shares ≥1\nendpoint"])
ax[1].set_ylabel("% of pairs mapping to the SAME Top-1 target")
ax[1].set_title("(b)  …and collapse to the same target more often", loc="left")
ax[1].set_ylim(0, r_sh + 5)
ax[1].text(0.97, 0.97, "but shared-endpoint pairs are\nonly 40/1485 (2.7%) of all pairs\n→ a secondary contributor",
           transform=ax[1].transAxes, ha="right", va="top", fontsize=8.6, color="#555",
           bbox=dict(boxstyle="round,pad=0.4", fc="#f6f0e6", ec="#d8c79a"))

plt.tight_layout(pad=1.4)
out = f"{D}/figures/endpoint_sharing_diagnosis.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
print(f"shared: cos={np.mean(sh_cos):.4f} same={r_sh:.1f}% | nonshared: cos={np.mean(ns_cos):.4f} same={r_ns:.1f}%")
print(f"collisions: shared-endpoint same-target pairs ~{int(round(np.sum(sh_same)))}, "
      f"non-shared ~{int(round(np.sum(ns_same)))} "
      f"-> endpoint sharing explains {100*np.sum(sh_same)/(np.sum(sh_same)+np.sum(ns_same)):.0f}% of collisions")
