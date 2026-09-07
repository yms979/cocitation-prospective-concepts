# Source file in the working repository: EDA/plot_ours_vs_llm.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import os
"""Direct comparison figure: our structure-only decoded text vs the LLM content
baseline text, for the same 55 predicted links.
  (a) 55x55 cosine heatmap (diagonal = same link)
  (b) same-link vs cross-link cosine distribution + link-recovery box
"""
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#444444", "axes.linewidth": 1.0,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11.5, "grid.color": "#e7e7e7", "grid.linewidth": 0.9,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "savefig.facecolor": "white",
})
SAME, CROSS = "#C4453B", "#2A9D8F"  # ours-side red (kept); LLM/cross recolored to teal to avoid clash with real-patent blue


def parse(v):
    try:
        return ast.literal_eval(v)
    except Exception:
        return None


D = _ROOT
K_EMB = os.environ.get("OPENAI_API_KEY")
emb = OpenAI(api_key=K_EMB)
fd = pd.read_csv(f"{D}/data/final_decoded_results_with_nn.csv"); fd["e"] = fd["embedding"].apply(parse)
P_ours = np.array([x for x in fd["e"]])
base = [str(x) for x in pd.read_csv(f"{D}/data/head2head_multirun.csv")["baseline_run1"]][:55]
r = emb.embeddings.create(model="text-embedding-ada-002", input=[t[:8000] for t in base])
P_base = np.array([d.embedding for d in r.data])

S = cosine_similarity(P_ours, P_base)         # rows = ours, cols = baseline
diag = np.diag(S)
off = S[~np.eye(55, dtype=bool)]
recovery = 100 * np.mean(S.argmax(axis=1) == np.arange(55))
print(f"same={diag.mean():.4f} cross={off.mean():.4f} lift={diag.mean()-off.mean():.4f} recovery={recovery:.1f}%")

fig, ax1 = plt.subplots(figsize=(7.2, 5.2))

# ---------- distribution (single panel) ----------
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
ax1.grid(axis="y", alpha=0.6)
xs = np.linspace(0.76, 0.96, 400)
ax1.hist(off, bins=50, density=True, color=CROSS, alpha=0.35, edgecolor="none", label=f"different link  (n={len(off):,})")
ax1.hist(diag, bins=14, density=True, color=SAME, alpha=0.50, edgecolor="none", label="same link  (n=55)")
ax1.plot(xs, gaussian_kde(off)(xs), color=CROSS, lw=2.4)
ax1.plot(xs, gaussian_kde(diag)(xs), color=SAME, lw=2.6)
ax1.axvline(off.mean(), color=CROSS, ls=(0, (4, 2)), lw=1.5)
ax1.axvline(diag.mean(), color=SAME, ls=(0, (4, 2)), lw=1.5)
ymax = ax1.get_ylim()[1]
ax1.annotate("", xy=(diag.mean(), ymax * 0.9), xytext=(off.mean(), ymax * 0.9),
             arrowprops=dict(arrowstyle="<|-|>", color="#333", lw=1.3, mutation_scale=12))
ax1.text((diag.mean() + off.mean()) / 2, ymax * 0.96, "+0.046", ha="center", va="top", fontsize=10, fontweight="bold")
ax1.text(off.mean() - 0.003, ymax * 0.55, f"cross\n{off.mean():.3f}", ha="right", color=CROSS, fontsize=9, fontweight="bold")
ax1.text(diag.mean() + 0.003, ymax * 0.68, f"same\n{diag.mean():.3f}", ha="left", color=SAME, fontsize=9, fontweight="bold")
ax1.set_xlabel("cosine: our description vs LLM description")
ax1.set_ylabel("Density")
ax1.set_title("Link-specific similarity to the matched LLM description", loc="left")
ax1.legend(loc="upper left", fontsize=9)
ax1.set_xlim(0.78, 0.95)
ax1.text(0.975, 0.97, f"same-link is the nearest\nLLM output for {recovery:.0f}% of links\n(≈{recovery/(100/55):.0f}× chance)",
         transform=ax1.transAxes, ha="right", va="top", fontsize=9, color="#444",
         bbox=dict(boxstyle="round,pad=0.4", fc="#fbe9e7", ec="#e0a89a"))

plt.tight_layout(pad=1.0)
out = f"{D}/figures/ours_vs_llm_similarity.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
