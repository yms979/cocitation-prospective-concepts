"""
Placebo / null-distribution z-score for prospective-node validation.
=====================================================================
Reviewer concern: in-domain mean cosine similarity is already high (~0.814),
so even explanations generated from RANDOM node pairs (not predicted links)
could show a high Top-1 match to the held-out validation set. This script
quantifies how far the PREDICTED prospective nodes sit above two null
distributions, with a permutation test (assumption-free, no normality needed)
and leakage control (validation docs that are also seeds / domain patents are
removed from both the validation target set and the null sampling pool).

Two nulls:
  A) Decoded-prospective-text embeddings  vs  random existing-domain patents.
     -> "Do the decoded explanations match the validation frontier better than
         a random sample of existing domain patents?"  (controls 0.814 baseline)
  B) Predicted-link endpoint pairs (mean of the two ada text embeddings)
     vs  random patent pairs.  (decoding-free; isolates the link-prediction
         signal -> "predicted links vs random links")

Outputs z = (observed_mean - null_mean) / null_std, empirical p, percentile.
"""
# Source file in the working repository: code/12-3.placebo_zscore.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import os, ast
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

SEED = 7
N_PERM = 20000
DATA = _ROOT + "/data"
ACQ = _ROOT + "/data/raw"
rng = np.random.default_rng(SEED)


def parse(v):
    try:
        return ast.literal_eval(v)
    except Exception:
        return None


# --- load embeddings (ada-002, 1536-dim) ---
val = pd.read_csv(f"{ACQ}/validation_abstract.csv", dtype=str)
val["e"] = val["embedding"].apply(parse)
val["original_id"] = val["original_id"].astype(str)

pro = pd.read_csv(f"{DATA}/final_decoded_results_with_nn.csv")
pro["e"] = pro["embedding"].apply(parse)

ne = pd.read_csv(f"{DATA}/node_embeddings_with_text_embedding_ada.csv",
                 dtype={"id": str, "text_embedding": str})
ne = ne[~ne["id"].astype(str).str.startswith("prospective")].copy()
ne["te"] = ne["text_embedding"].apply(
    lambda x: parse(x) if isinstance(x, str) and x.strip() not in ("", "nan", "None") else None)
ne = ne[ne["te"].notnull()]

seeds = set(pd.read_csv(f"{DATA}/raw/collected_data_abstract_with_citation.csv",
                        dtype=str)["original_id"].astype(str))
dom_ids = ne["id"].astype(str).tolist()

# --- leakage control: held-out validation = not a seed and not in domain pool ---
clean = val[~val["original_id"].isin(seeds | set(dom_ids))]
V = np.array([x for x in clean["e"]])
P = np.array([x for x in pro["e"]])
print(f"held-out validation docs: {len(clean)} (removed {len(val) - len(clean)} overlapping of {len(val)})")

# null pool = domain patents not present in the validation set
valids = set(val["original_id"])
pool = ne[~ne["id"].astype(str).isin(valids)]
D = np.array([x for x in pool["te"]])
id2idx = {k: i for i, k in enumerate(pool["id"].astype(str).tolist())}
print(f"null sampling pool: {len(D)} domain patents")


def top1(M):
    """per-row max cosine similarity to the validation matrix V"""
    return cosine_similarity(M, V).max(axis=1)


# ===== NULL A: decoded prospective vs random domain patents =====
obs = top1(P)
obs_mean = obs.mean()
dom_t1 = top1(D)
k = len(P)
nm = np.array([dom_t1[rng.choice(len(D), k, replace=False)].mean() for _ in range(N_PERM)])
zA = (obs_mean - nm.mean()) / nm.std()
pA = (np.sum(nm >= obs_mean) + 1) / (N_PERM + 1)
thr95 = np.percentile(dom_t1, 95)
print("\n[NULL A] decoded prospective vs random domain patents")
print(f"  observed mean Top-1 = {obs_mean:.4f}")
print(f"  null mean Top-1     = {nm.mean():.4f} (sd {nm.std():.4f})")
print(f"  z = {zA:.2f} | empirical p = {pA:.5f}")
print(f"  per-node: {int(np.sum(obs > thr95))}/{k} exceed the 95th pct of random patents (~{0.05*k:.1f} expected)")

# ===== NULL B: predicted-link pairs vs random pairs (decoding-free) =====
pl = pd.read_csv(f"{DATA}/predicted_new_links.csv", dtype=str)
ok = pl[pl["node1"].isin(id2idx) & pl["node2"].isin(id2idx)]
op = np.array([(D[id2idx[r.node1]] + D[id2idx[r.node2]]) / 2 for r in ok.itertuples()])
op_mean = top1(op).mean()
kk = len(ok)
nb = np.array([top1((D[rng.integers(0, len(D), kk)] + D[rng.integers(0, len(D), kk)]) / 2).mean()
               for _ in range(N_PERM)])
zB = (op_mean - nb.mean()) / nb.std()
pB = (np.sum(nb >= op_mean) + 1) / (N_PERM + 1)
print(f"\n[NULL B] predicted-link pairs vs random pairs ({kk}/{len(pl)} usable)")
print(f"  observed predicted-pair mean Top-1 = {op_mean:.4f}")
print(f"  random-pair null mean Top-1        = {nb.mean():.4f} (sd {nb.std():.4f})")
print(f"  z = {zB:.2f} | empirical p = {pB:.5f}")
