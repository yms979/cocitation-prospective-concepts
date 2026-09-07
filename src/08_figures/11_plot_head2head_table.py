"""Head-to-head summary table: structure-only (ours) vs content LLM baseline."""
# Source file in the working repository: EDA/plot_head2head_table.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OURS = "#C4453B"     # ours / structure-only
LLM = "#2A9D8F"      # LLM / content
TIE_BG = "#EFEFEF"
LLM_BG = "#D4ECE7"
GRP_BG = "#F5F1E8"

# (group, metric, ours, llm, verdict, who)  who: 'tie' | 'llm' | 'ours'
rows = [
    ("Resemblance & specificity\n(text present)", "Validation Top-1  (mean)", "0.892", "0.890", "tie  (Δ+0.002, boot p=0.42)", "tie"),
    ("", "Peakedness (specificity)", "0.078", "0.075", "tie", "tie"),
    ("", "Distinct Top-1 targets / 55", "24", "26–31", "tie", "tie"),
    ("", "Internal cosine (diversity)", "0.866", "0.862", "tie", "tie"),
    ("Agreement", "Same-link cosine (ours vs LLM)", "0.896", "—", "ours reproduces LLM", "ours"),
    ("", "  ( cross-link 0.850; nearest in 36% ≈20×)", "", "", "", "ours"),
    ("Concept quality\n(gpt-5.5 agent)", "Mean YES / 5", "2.95", "4.73", "LLM", "llm"),
    ("", "Coherence / 5", "2.16", "3.89", "LLM", "llm"),
    ("", "HIGH-potential rate", "4%", "91%", "LLM", "llm"),
    ("Text-dropout\n(inference cold-start)", "Top-1, title-only evidence", "0.892*", "0.890", "tie  (no LLM drop, p=0.99)", "tie"),
]
cols = ["", "Metric", "Structure-only\n(ours)", "LLM\n(content)", "Verdict"]
cw = [0.20, 0.30, 0.16, 0.14, 0.30]

fig, ax = plt.subplots(figsize=(12.6, 5.6))
ax.axis("off")
ax.set_title("Head-to-head:  structure-only pipeline  vs  content-based LLM baseline (same 55 links)",
             fontsize=13, fontweight="bold", loc="left", pad=14)
n = len(rows)
rh = 1.0 / (n + 1)
xb = [sum(cw[:i]) for i in range(len(cw) + 1)]

# header
for c in range(len(cols)):
    ax.add_patch(plt.Rectangle((xb[c], 1 - rh), cw[c], rh, color="#3A3A3A"))
    ax.text(xb[c] + cw[c] / 2, 1 - rh / 2, cols[c], ha="center", va="center",
            color="white", fontsize=10.5, fontweight="bold")

prev_grp = None
for r, (grp, metric, o, l, verdict, who) in enumerate(rows):
    y = 1 - (r + 2) * rh
    # group cell
    ax.add_patch(plt.Rectangle((xb[0], y), cw[0], rh, facecolor=GRP_BG, edgecolor="white"))
    if grp:
        ax.text(xb[0] + 0.012, y + rh / 2, grp, ha="left", va="center", fontsize=8.6, fontweight="bold", color="#6b5a33")
    # metric
    ax.add_patch(plt.Rectangle((xb[1], y), cw[1], rh, facecolor="white", edgecolor="#e0e0e0"))
    ax.text(xb[1] + 0.012, y + rh / 2, metric, ha="left", va="center", fontsize=9.2,
            color="#333", style="italic" if metric.strip().startswith("(") else "normal")
    # ours value
    ax.add_patch(plt.Rectangle((xb[2], y), cw[2], rh, facecolor="white", edgecolor="#e0e0e0"))
    ax.text(xb[2] + cw[2] / 2, y + rh / 2, o, ha="center", va="center", fontsize=10,
            fontweight="bold" if who == "ours" else "normal", color=OURS if o not in ("", "—") else "#999")
    # llm value
    ax.add_patch(plt.Rectangle((xb[3], y), cw[3], rh, facecolor="white", edgecolor="#e0e0e0"))
    ax.text(xb[3] + cw[3] / 2, y + rh / 2, l, ha="center", va="center", fontsize=10,
            fontweight="bold" if who == "llm" else "normal", color=LLM if l not in ("", "—") else "#999")
    # verdict
    bg = {"tie": TIE_BG, "llm": LLM_BG, "ours": "#F7DEDB"}[who]
    ax.add_patch(plt.Rectangle((xb[4], y), cw[4], rh, facecolor=bg, edgecolor="white"))
    vc = {"tie": "#555", "llm": LLM, "ours": OURS}[who]
    ax.text(xb[4] + 0.012, y + rh / 2, verdict, ha="left", va="center", fontsize=8.8, color=vc,
            fontweight="bold" if who in ("llm", "ours") else "normal")

ax.set_xlim(0, 1); ax.set_ylim(1 - (n + 1) * rh, 1)
fig.text(0.012, 0.012,
         "Takeaway: with text present the structure-only pipeline is on par with the content LLM (and reproduces it link-specifically); "
         "a strong judge rates the LLM's concepts higher. *structure-only reads no endpoint text at inference, so it is invariant to text-dropout.",
         fontsize=8.3, color="#555", style="italic")
plt.tight_layout(rect=[0, 0.04, 1, 1])
out = _ROOT + "/figures/head2head_table.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
