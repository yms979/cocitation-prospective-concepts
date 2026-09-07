"""
Re-train the cross-modal alignment (node_emb 1024 -> ada text 1536) and
evaluate with SCALE-INVARIANT metrics on a held-out validation split.

Faithful re-implementation of "src/04_cross_modal_alignment/01_ridge_residual_alignment.py" (Ridge Stage A
+ Residual-MLP Stage B, loss = MSE + 0.5*InfoNCE + 0.5*Cosine), configured for
ada-002 (TEXT_EMB_DIM=1536). Adds, on the validation set:
  - MSE (per-dimension), average cosine similarity, average L2
  - self-retrieval Top-1/5/10 and median rank (project val node -> rank its OWN
    text embedding among ALL real text embeddings) + random baseline
Reports Stage A (Ridge) and Stage A+B (Residual), plus the auto-selected one.
"""
# Source file in the working repository: code/12-4.retrain_eval_alignment.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import os, ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

torch.manual_seed(42); np.random.seed(42)
DATA = _ROOT + "/data"
INPUT = f"{DATA}/node_embeddings_with_text_embedding_ada.csv"
NODE_EMB_DIM, TEXT_EMB_DIM = 1024, 1536
RIDGE_ALPHA = 1.0
HIDDEN_DIM, NUM_RES_BLOCKS, DROPOUT = 512, 3, 0.1
LR, BATCH, MAX_EPOCHS, PATIENCE = 1e-4, 64, 3000, 50
ALPHA, BETA, GAMMA, TEMP = 1.0, 0.5, 0.5, 0.07
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---- model / loss (verbatim from 10-1) ----
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))


class ResidualCorrectionModel(nn.Module):
    def __init__(self, node_dim, text_dim, hidden_dim, n_blocks=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(node_dim + text_dim, hidden_dim),
                                         nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.residual_blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden_dim, text_dim)
        self.residual_scale = nn.Parameter(torch.tensor(0.0))
    def forward(self, node_emb, linear_projected):
        h = self.input_proj(torch.cat([node_emb, linear_projected], dim=-1))
        h = self.residual_blocks(h)
        return linear_projected + torch.sigmoid(self.residual_scale) * self.output_proj(h)


class InfoNCELoss(nn.Module):
    def __init__(self, t=0.07): super().__init__(); self.t = t
    def forward(self, p, tg):
        if p.size(0) <= 1: return torch.tensor(0.0, device=p.device)
        logits = torch.mm(F.normalize(p, 2, 1), F.normalize(tg, 2, 1).t()) / self.t
        lab = torch.arange(p.size(0), device=p.device)
        return (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab)) / 2


class CombinedLoss(nn.Module):
    def __init__(self, a, b, g, t):
        super().__init__(); self.a, self.b, self.g = a, b, g
        self.mse, self.infonce, self.cos = nn.MSELoss(), InfoNCELoss(t), nn.CosineEmbeddingLoss()
    def forward(self, p, tg):
        return self.a * self.mse(p, tg) + self.b * self.infonce(p, tg) + \
               self.g * self.cos(p, tg, torch.ones(p.size(0), device=p.device))


class DS(Dataset):
    def __init__(self, n, l, t): self.n, self.l, self.t = map(lambda a: torch.tensor(a, dtype=torch.float32), (n, l, t))
    def __len__(self): return len(self.n)
    def __getitem__(self, i): return self.n[i], self.l[i], self.t[i]


def parse_col(s): return np.array([ast.literal_eval(x) if isinstance(x, str) else x for x in s], dtype=np.float32)


def metrics(P, Y, all_text_norm, self_global_idx, label):
    mse = float(np.mean((P - Y) ** 2))
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    cos = float(np.mean(np.sum(Pn * Yn, axis=1)))
    l2 = float(np.mean(np.linalg.norm(P - Y, axis=1)))
    S = Pn @ all_text_norm.T
    ranks = np.array([int(np.where(np.argsort(-S[r]) == self_global_idx[r])[0][0]) + 1 for r in range(len(P))])
    t1, t5, t10 = [100 * np.mean(ranks <= k) for k in (1, 5, 10)]
    print(f"  [{label}] MSE={mse:.6f} | Cos={cos:.4f} | L2={l2:.4f} | "
          f"Top-1/5/10={t1:.1f}/{t5:.1f}/{t10:.1f}% | median rank={int(np.median(ranks))}/{all_text_norm.shape[0]}")
    return dict(mse=mse, cos=cos, top1=t1, top5=t5, top10=t10, medrank=int(np.median(ranks)))


# ---- data ----
df = pd.read_csv(INPUT, dtype={'id': str, 'node_embedding': str, 'text_embedding': str})
mask = (df['text_embedding'].notna() & (df['text_embedding'].str.strip() != '') &
        (df['text_embedding'].str.strip().str.lower() != 'none') & df['node_embedding'].notna() &
        (~df['id'].astype(str).str.startswith('prospective')))
pdf = df[mask].reset_index(drop=True)
N, T = parse_col(pdf['node_embedding']), parse_col(pdf['text_embedding'])
print(f"pairs={len(N)} | node{N.shape} text{T.shape} | text L2 norm mean={np.linalg.norm(T,axis=1).mean():.4f}")
idx = np.arange(len(N))
tr, va = train_test_split(idx, test_size=0.2, random_state=42)
all_text_norm = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)   # retrieval gallery = all real texts
rand_top1 = 100.0 / len(T)

# ---- Stage A: Ridge ----
ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True).fit(N[tr], T[tr])
lin_tr, lin_va = ridge.predict(N[tr]), ridge.predict(N[va])
print(f"\n=== Validation ({len(va)} held-out pairs), random Top-1={rand_top1:.3f}% ===")
res_A = metrics(lin_va, T[va], all_text_norm, va, "Stage A (Ridge)   ")

# ---- Stage B: Residual MLP ----
model = ResidualCorrectionModel(NODE_EMB_DIM, TEXT_EMB_DIM, HIDDEN_DIM, NUM_RES_BLOCKS, DROPOUT).to(device)
crit = CombinedLoss(ALPHA, BETA, GAMMA, TEMP).to(device)
opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-6)
loader = DataLoader(DS(N[tr], lin_tr, T[tr]), batch_size=BATCH, shuffle=True, drop_last=True)
vN = torch.tensor(N[va], dtype=torch.float32).to(device)
vL = torch.tensor(lin_va, dtype=torch.float32).to(device)
vT = torch.tensor(T[va], dtype=torch.float32).to(device)
best, best_state, wait = float('inf'), None, 0
for ep in range(MAX_EPOCHS):
    model.train()
    for n, l, t in loader:
        opt.zero_grad()
        loss = crit(model(n.to(device), l.to(device)), t.to(device))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    sch.step()
    model.eval()
    with torch.no_grad():
        vloss = crit(model(vN, vL), vT).item()
    if vloss < best - 0:
        best, best_state, wait = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"early stop @ epoch {ep+1} (best val loss {best:.4f})"); break
model.load_state_dict(best_state); model.eval()
with torch.no_grad():
    final_va = model(vN, vL).cpu().numpy()
res_AB = metrics(final_va, T[va], all_text_norm, va, "Stage A+B (Resid) ")
sel = "A+B" if res_AB['cos'] > res_A['cos'] else "A"
print(f"\nresidual_scale={torch.sigmoid(model.residual_scale).item():.4f} | auto-selected: Stage {sel}")
print(f"\n>>> REPORTABLE (deployed-class, ada-1536, held-out val): "
      f"cos={max(res_A['cos'],res_AB['cos']):.3f}, "
      f"Top-1={(res_AB if sel=='A+B' else res_A)['top1']:.1f}% "
      f"(random {rand_top1:.3f}%, ~{(res_AB if sel=='A+B' else res_A)['top1']/rand_top1:.0f}x)")
