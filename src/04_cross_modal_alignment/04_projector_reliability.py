"""
Reliability checks for the cross-modal projector (alignment stage).

(1) Multi-seed stability: train the residual-correction projector across K seeds
    (upstream GAT embeddings held fixed) and report mean +/- SD of the projected
    prospective concepts' Top-1 cosine and peakedness vs the 120 validation patents.
(2) Projector ablation: closed-form linear map (Ridge, deterministic) vs the
    residual-correction variant, on held-out self-retrieval (Top-1/5/10 on the 20%
    held-out real pairs) AND the prospective->validation metrics.

Scope: metrics are computed on PROJECTED embeddings (pre-vec2text-decoding);
upstream GAT/GCN training and vec2text decoding are held fixed.
"""
# Source file in the working repository: code/18.projector_reliability.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity

D = _ROOT
NODE_DIM, TEXT_DIM = 1024, 1536
SEEDS = [11, 23, 37, 51, 67]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse(v):
    try:
        return ast.literal_eval(v)
    except Exception:
        return None


class ResBlock(nn.Module):
    def __init__(self, d, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d), nn.LayerNorm(d))
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))


class Residual(nn.Module):
    def __init__(self, nd, td, h=512, nb=3, p=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nd + td, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p))
        self.blocks = nn.Sequential(*[ResBlock(h, p) for _ in range(nb)])
        self.out = nn.Linear(h, td)
        self.scale = nn.Parameter(torch.tensor(0.0))
    def forward(self, n, lin):
        h = self.blocks(self.inp(torch.cat([n, lin], -1)))
        return lin + torch.sigmoid(self.scale) * self.out(h)


class InfoNCE(nn.Module):
    def __init__(self, t=0.07): super().__init__(); self.t = t
    def forward(self, p, q):
        if p.size(0) <= 1: return torch.tensor(0.0, device=p.device)
        lo = torch.mm(F.normalize(p, 2, 1), F.normalize(q, 2, 1).t()) / self.t
        lab = torch.arange(p.size(0), device=p.device)
        return (F.cross_entropy(lo, lab) + F.cross_entropy(lo.t(), lab)) / 2


def parse_col(s): return np.array([ast.literal_eval(x) if isinstance(x, str) else x for x in s], dtype=np.float32)


# ---- data ----
val = pd.read_csv(f"{D}/data/raw/validation_abstract.csv"); val["e"] = val["embedding"].apply(parse)
V = np.array([x for x in val["e"]])
df = pd.read_csv(f"{D}/data/node_embeddings_with_text_embedding_ada.csv", dtype={"id": str, "node_embedding": str, "text_embedding": str})
real = df[~df["id"].astype(str).str.startswith("prospective")].copy()
real = real[real["text_embedding"].notna() & (real["text_embedding"].astype(str).str.strip().str.lower() != "none") & (real["text_embedding"].astype(str).str.strip() != "")]
N = parse_col(real["node_embedding"]); T = parse_col(real["text_embedding"])
prosp = df[df["id"].astype(str).str.startswith("prospective")]
PN = parse_col(prosp["node_embedding"])
idx = np.arange(len(N)); tr, va = train_test_split(idx, test_size=0.2, random_state=42)
Tn_all = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)   # retrieval gallery (all real texts)


def val_metrics(P):  # projected prospective -> validation
    S = cosine_similarity(P, V)
    return S.max(axis=1).mean(), (S.max(axis=1) - S.mean(axis=1)).mean()


def retrieval(P_va):  # held-out val real, projected -> rank own text among all real texts
    Pn = P_va / (np.linalg.norm(P_va, axis=1, keepdims=True) + 1e-8)
    Sg = Pn @ Tn_all.T
    ranks = np.array([int(np.where(np.argsort(-Sg[r]) == va[r])[0][0]) + 1 for r in range(len(va))])
    return [100 * np.mean(ranks <= k) for k in (1, 5, 10)]


# ---- Stage A: Ridge (deterministic linear) ----
ridge = Ridge(alpha=1.0).fit(N[tr], T[tr])
linA_va, linA_prosp = ridge.predict(N[va]), ridge.predict(PN)
A_t1, A_pk = val_metrics(linA_prosp); A_ret = retrieval(linA_va)
print("=== (2) PROJECTOR ABLATION ===")
print(f"  Linear (Ridge, deterministic):  held-out retrieval Top-1/5/10 = {A_ret[0]:.1f}/{A_ret[1]:.1f}/{A_ret[2]:.1f}%"
      f" | prospective->val Top-1={A_t1:.4f} peakedness={A_pk:.4f}")


def train_residual(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Residual(NODE_DIM, TEXT_DIM).to(dev)
    crit_mse, crit_cos, infonce = nn.MSELoss(), nn.CosineEmbeddingLoss(), InfoNCE()
    opt = optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
    lin_tr = ridge.predict(N[tr])

    class DS(Dataset):
        def __init__(s): s.n, s.l, s.t = map(lambda a: torch.tensor(a, dtype=torch.float32), (N[tr], lin_tr, T[tr]))
        def __len__(s): return len(s.n)
        def __getitem__(s, i): return s.n[i], s.l[i], s.t[i]
    dl = DataLoader(DS(), batch_size=64, shuffle=True, drop_last=True)
    vN = torch.tensor(N[va], dtype=torch.float32).to(dev); vL = torch.tensor(linA_va, dtype=torch.float32).to(dev)
    vT = torch.tensor(T[va], dtype=torch.float32).to(dev)
    best, best_sd, wait = 1e9, None, 0
    for ep in range(2000):
        m.train()
        for n, l, t in dl:
            opt.zero_grad()
            o = m(n.to(dev), l.to(dev)); tt = t.to(dev)
            loss = crit_mse(o, tt) + 0.5 * infonce(o, tt) + 0.5 * crit_cos(o, tt, torch.ones(o.size(0), device=dev))
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad():
            vl = (crit_mse(m(vN, vL), vT) + 0.5 * infonce(m(vN, vL), vT) + 0.5 * crit_cos(m(vN, vL), vT, torch.ones(len(va), device=dev))).item()
        if vl < best: best, best_sd, wait = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 50: break
    m.load_state_dict(best_sd); m.eval()
    with torch.no_grad():
        pv = m(torch.tensor(N[va], dtype=torch.float32).to(dev), torch.tensor(linA_va, dtype=torch.float32).to(dev)).cpu().numpy()
        pp = m(torch.tensor(PN, dtype=torch.float32).to(dev), torch.tensor(linA_prosp, dtype=torch.float32).to(dev)).cpu().numpy()
    t1, pk = val_metrics(pp); ret = retrieval(pv)
    return t1, pk, ret[0], ret[1], ret[2]


print(f"\n  Residual-correction (mean +/- SD over {len(SEEDS)} seeds):")
res = np.array([train_residual(s) for s in SEEDS])  # rows: t1, pk, top1, top5, top10
lab = ["prospective->val Top-1", "prospective->val peakedness", "held-out retrieval Top-1", "Top-5", "Top-10"]
for i, l in enumerate(lab):
    print(f"    {l:30s}: {res[:,i].mean():.4f} +/- {res[:,i].std(ddof=1):.4f}")

print("\n=== (1) MULTI-SEED STABILITY (residual projector) ===")
print(f"  prospective->validation Top-1 cosine : {res[:,0].mean():.4f} +/- {res[:,0].std(ddof=1):.4f}  (per-seed: {[round(x,4) for x in res[:,0]]})")
print(f"  prospective->validation peakedness   : {res[:,1].mean():.4f} +/- {res[:,1].std(ddof=1):.4f}")
print("\n  (linear map is deterministic -> zero run-to-run variance)")
