"""
Section 4.7 sensitivity: re-run the Stage-1 link-prediction (GCN + 5-model XGBoost,
faithful to 'src/02_link_prediction/01_link_prediction_gcn_xgboost.py') to obtain ranked candidate links, then sweep the
probability threshold and the per-node cap to report how the predicted-link set
(number of links, number of unique endpoints) responds.

Scope: downstream validation metrics require re-embedding the resulting prospective
nodes with the GAT (out of scope here). Absolute counts differ slightly from the
deployed run due to stochastic training; the sensitivity TREND is the deliverable.
"""
# Source file in the working repository: code/19.threshold_cap_sensitivity.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
import xgboost as xgb
from sklearn.model_selection import train_test_split as sk_split
from collections import defaultdict

torch.manual_seed(42); np.random.seed(42)
D = _ROOT
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(torch.nn.Module):
    def __init__(self, n, ed, hd, od, p=0.2):
        super().__init__()
        self.emb = torch.nn.Embedding(n, ed)
        self.c1 = GCNConv(ed, hd); self.c2 = GCNConv(hd, od); self.p = p
    def forward(self, ei):
        x = self.emb(torch.arange(self.emb.num_embeddings, device=ei.device))
        x = F.dropout(F.relu(self.c1(x, ei)), p=self.p, training=self.training)
        return self.c2(x, ei)


def feats(emb, ei, deg):
    ei = ei.cpu(); u, v = ei[0], ei[1]
    e = (emb[u] * emb[v]).numpy()
    du = np.array([deg.get(i.item(), 0) for i in u]); dv = np.array([deg.get(i.item(), 0) for i in v])
    return np.hstack([e, (du * dv).reshape(-1, 1)])


# ---- load co-citation network ----
G0 = nx.read_gexf(f"{D}/data/networks/patent_co_citation_network_filtered.gexf")
nmap = {id: i for i, id in enumerate(G0.nodes())}; rmap = {i: id for id, i in nmap.items()}
G = nx.relabel_nodes(G0, nmap)
ei = []
for u, v in G.edges(): ei += [[u, v], [v, u]]
ei = torch.tensor(ei, dtype=torch.long).t().contiguous()
nn_ = G.number_of_nodes()
data = Data(edge_index=ei, num_nodes=nn_)
print(f"co-citation network: {nn_} nodes, {G.number_of_edges()} edges")
tr, va, te = RandomLinkSplit(num_val=int(data.edge_index.size(1)*0.05), num_test=int(data.edge_index.size(1)*0.1),
                             is_undirected=True, split_labels=True, add_negative_train_samples=False)(data)

# ---- train GCN ----
m = GCN(nn_, 64, 128, 64).to(dev); opt = torch.optim.Adam(m.parameters(), lr=0.01)
tr, va = tr.to(dev), va.to(dev)
best, bs, wait = 1e9, None, 0
for ep in range(1, 501):
    m.train(); opt.zero_grad()
    pos = tr.edge_index
    neg = negative_sampling(pos, data.num_nodes, pos.size(1), method="sparse")
    z = m(pos)
    ps = (z[pos[0]]*z[pos[1]]).sum(1); ns = (z[neg[0]]*z[neg[1]]).sum(1)
    loss = F.binary_cross_entropy_with_logits(ps, torch.ones_like(ps)) + F.binary_cross_entropy_with_logits(ns, torch.zeros_like(ns))
    loss.backward(); opt.step()
    if ep % 10 == 0:
        m.eval()
        with torch.no_grad():
            zv = m(va.edge_index)
            vp = (zv[va.pos_edge_label_index[0]]*zv[va.pos_edge_label_index[1]]).sum(1)
            vn = (zv[va.neg_edge_label_index[0]]*zv[va.neg_edge_label_index[1]]).sum(1)
            vl = (F.binary_cross_entropy_with_logits(vp, torch.ones_like(vp)) + F.binary_cross_entropy_with_logits(vn, torch.zeros_like(vn))).item()
        if vl < best - 1e-4: best, bs, wait = vl, m.state_dict(), 0
        else:
            wait += 1
            if wait >= 50: break
if bs: m.load_state_dict(bs)
m.eval()
with torch.no_grad(): emb = m(data.edge_index.to(dev)).cpu()
print("GCN trained.")

# ---- 5 XGBoost ----
deg = dict(G.degree()); npos = int(G.number_of_edges()*0.05)
allpos = tr.edge_index.cpu()
models = []
for i in range(5):
    perm = torch.randperm(allpos.size(1))[:npos]; sp = allpos[:, perm]
    Xp = feats(emb, sp, deg); yp = np.ones(Xp.shape[0])
    sn = negative_sampling(data.edge_index.cpu(), data.num_nodes, npos, method="sparse")
    Xn = feats(emb, sn, deg); yn = np.zeros(Xn.shape[0])
    X = np.vstack([Xp, Xn]); y = np.hstack([yp, yn])
    Xtr, _, ytr, _ = sk_split(X, y, test_size=0.2, random_state=i, stratify=y)
    mdl = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=i)
    mdl.fit(Xtr, ytr); models.append(mdl)
print("5 XGBoost trained.")

# ---- score candidates (retain all-5 >= 0.5) ----
ncand = 2_000_000
base = te.neg_edge_label_index.cpu()
add = negative_sampling(data.edge_index.cpu(), data.num_nodes, max(0, ncand - base.size(1)), method="sparse")
cand = torch.cat([base, add], 1).numpy()
u, v = cand[0], cand[1]
keep = u != v
u, v = u[keep], v[keep]
lo = np.minimum(u, v); hi = np.maximum(u, v)
key = lo.astype(np.int64) * nn_ + hi
_, uniq = np.unique(key, return_index=True)
lo, hi = lo[uniq], hi[uniq]
# remove existing edges
exist = set((min(a, b), max(a, b)) for a, b in G.edges())
mask = np.array([(a, b) not in exist for a, b in zip(lo, hi)])
lo, hi = lo[mask], hi[mask]
print(f"unique candidate non-edges scored: {len(lo):,}")

ce = torch.tensor(np.vstack([lo, hi]), dtype=torch.long)
keep_u, keep_v, keep_avg, keep_min = [], [], [], []
bs_ = 100000
for s in range(0, ce.size(1), bs_):
    b = ce[:, s:s+bs_]
    X = feats(emb, b, deg)
    probs = np.array([md.predict_proba(X)[:, 1] for md in models])  # (5, B)
    mn = probs.min(0); av = probs.mean(0)
    sel = mn >= 0.5
    keep_u += b[0, sel].tolist(); keep_v += b[1, sel].tolist()
    keep_avg += av[sel].tolist(); keep_min += mn[sel].tolist()
ku = np.array(keep_u); kv = np.array(keep_v); kavg = np.array(keep_avg); kmin = np.array(keep_min)
print(f"candidates with all-5 prob >= 0.5: {len(ku):,}")

# ---- sweep threshold x cap ----
def select(thr, cap):
    sel = kmin >= thr
    su, sv, sa = ku[sel], kv[sel], kavg[sel]
    order = np.argsort(-sa)
    cnt = defaultdict(int); links = 0; eps = set()
    for idx in order:
        a, b = su[idx], sv[idx]
        if cap is None or (cnt[a] < cap and cnt[b] < cap):
            links += 1; cnt[a] += 1; cnt[b] += 1; eps.add(a); eps.add(b)
    return links, len(eps)

THR = [0.5, 0.7, 0.9, 0.95, 0.99]; CAP = [1, 2, 3, 5, None]
print("\n=== predicted-link SET SIZE (links) by threshold x per-node cap ===")
print("thr\\cap | " + " | ".join(f"{('inf' if c is None else c):>5}" for c in CAP))
for t in THR:
    print(f"{t:>5} | " + " | ".join(f"{select(t,c)[0]:>5}" for c in CAP))
print("\n=== UNIQUE ENDPOINTS by threshold x per-node cap ===")
print("thr\\cap | " + " | ".join(f"{('inf' if c is None else c):>5}" for c in CAP))
for t in THR:
    print(f"{t:>5} | " + " | ".join(f"{select(t,c)[1]:>5}" for c in CAP))
print(f"\n[deployed setting] threshold=0.5, cap=3 -> links={select(0.5,3)[0]}, endpoints={select(0.5,3)[1]}  (deployed run: 55 links, 81 endpoints)")
np.savez(f"{D}/data/threshold_cap_candidates.npz", u=ku, v=kv, avg=kavg, min=kmin)
print(_ROOT + "/data/saved threshold_cap_candidates.npz")
