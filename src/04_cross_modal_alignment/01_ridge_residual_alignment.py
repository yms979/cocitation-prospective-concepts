"""
Cross-modal Alignment: Node Embedding → Text Embedding
=======================================================
2-Stage Projection으로 투영 에러를 최소화합니다.

Stage A: Ridge Regression (선형 정렬)
  - 닫힌 해(closed-form)로 최적의 선형 변환 W를 구함
  - 파라미터 수가 적어 overfitting 위험이 낮음
  - 두 공간 사이의 대략적인 정렬을 담당

Stage B: Residual MLP (비선형 보정)
  - Stage A의 출력과 실제 target 사이의 "잔차(residual)"를 학습
  - 선형으로 포착하지 못한 비선형 패턴만 보정
  - 학습 대상이 잔차(작은 값)이므로 수렴이 빠르고 안정적

최종 투영: projected = Linear(node_emb) + ResidualMLP(node_emb)

Loss: MSE + InfoNCE + Cosine
"""
# Source file in the working repository: code/10-1. Cross-modal alignment.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import ast
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


# =============================================================================
# 1. Dataset
# =============================================================================
class AlignmentDataset(Dataset):
    """Stage B 학습용: node_emb, linear_projected, target_text_emb"""
    def __init__(self, node_embs: np.ndarray, linear_projected: np.ndarray, text_embs: np.ndarray):
        self.node_embs = torch.tensor(node_embs, dtype=torch.float32)
        self.linear_projected = torch.tensor(linear_projected, dtype=torch.float32)
        self.text_embs = torch.tensor(text_embs, dtype=torch.float32)

    def __len__(self):
        return len(self.node_embs)

    def __getitem__(self, idx):
        return self.node_embs[idx], self.linear_projected[idx], self.text_embs[idx]


# =============================================================================
# 2. Residual MLP (Stage B 모델)
# =============================================================================
class ResidualBlock(nn.Module):
    """Skip connection이 포함된 블록. 입력 정보를 보존하면서 보정만 학습."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class ResidualCorrectionModel(nn.Module):
    """
    Stage A의 선형 투영 결과를 받아 잔차를 보정합니다.
    
    입력: [node_emb (1024) ; linear_projected (1536)] → concat → 2560dim
    출력: correction vector (1536dim)
    
    최종 결과 = linear_projected + scale * correction
    """
    def __init__(
        self,
        node_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_residual_blocks: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        concat_dim = node_dim + text_dim

        self.input_proj = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_residual_blocks)]
        )

        self.output_proj = nn.Linear(hidden_dim, text_dim)

        # 잔차 크기 제어: 초기값 0 → 학습 초기에는 linear projection만 사용
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, node_emb: torch.Tensor, linear_projected: torch.Tensor):
        x = torch.cat([node_emb, linear_projected], dim=-1)

        h = self.input_proj(x)
        h = self.residual_blocks(h)
        correction = self.output_proj(h)

        scale = torch.sigmoid(self.residual_scale)
        output = linear_projected + scale * correction

        return output


# =============================================================================
# 3. Loss Functions
# =============================================================================
class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, projected: torch.Tensor, target: torch.Tensor):
        batch_size = projected.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=projected.device)

        proj_norm = F.normalize(projected, p=2, dim=1)
        tgt_norm = F.normalize(target, p=2, dim=1)
        logits = torch.mm(proj_norm, tgt_norm.t()) / self.temperature
        labels = torch.arange(batch_size, device=projected.device)

        loss_p2t = F.cross_entropy(logits, labels)
        loss_t2p = F.cross_entropy(logits.t(), labels)
        return (loss_p2t + loss_t2p) / 2


class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.5, temperature=0.07):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        self.infonce = InfoNCELoss(temperature)
        self.cosine = nn.CosineEmbeddingLoss()

    def forward(self, projected, target):
        l_mse = self.mse(projected, target)
        l_infonce = self.infonce(projected, target)
        cos_labels = torch.ones(projected.size(0), device=projected.device)
        l_cosine = self.cosine(projected, target, cos_labels)

        total = self.alpha * l_mse + self.beta * l_infonce + self.gamma * l_cosine
        return total, l_mse.item(), l_infonce.item(), l_cosine.item()


# =============================================================================
# 4. Early Stopping
# =============================================================================
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path=_ROOT + '/data/best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose and self.counter % 10 == 0:
                print(f'  EarlyStopping: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'  Val loss: {self.val_loss_min:.6f} → {val_loss:.6f} ✓ Saved')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# =============================================================================
# 5. 유틸리티
# =============================================================================
def parse_embedding_column(series: pd.Series) -> np.ndarray:
    parsed = []
    for x in series:
        if isinstance(x, str):
            parsed.append(ast.literal_eval(x))
        elif isinstance(x, list):
            parsed.append(x)
        else:
            parsed.append(x)
    return np.array(parsed, dtype=np.float32)


def evaluate_projection(projected: np.ndarray, target: np.ndarray, label: str = ""):
    mse = np.mean((projected - target) ** 2)

    proj_norm = projected / (np.linalg.norm(projected, axis=1, keepdims=True) + 1e-8)
    tgt_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
    cos_sims = np.sum(proj_norm * tgt_norm, axis=1)
    avg_cos = np.mean(cos_sims)

    l2_dists = np.linalg.norm(projected - target, axis=1)
    avg_l2 = np.mean(l2_dists)

    print(f"  [{label}] MSE: {mse:.6f} | Avg CosSim: {avg_cos:.4f} | Avg L2: {avg_l2:.4f}")
    return mse, avg_cos, avg_l2


# =============================================================================
# 6. Main
# =============================================================================
if __name__ == "__main__":

    # =========================================================================
    # 설정
    # =========================================================================
    input_csv_file = _ROOT + "/data/node_embeddings_with_text_embedding_ada.csv"
    ridge_model_path = _ROOT + "/data/ridge_weights.npz"
    residual_model_path = _ROOT + "/data/best_residual_correction_model.pth"

    NODE_EMB_DIM = 1024
    TEXT_EMB_DIM = 1536   # ada-002 (setting used for the reported results); 768 for the gte-base robustness run

    # Stage A
    RIDGE_ALPHA = 1.0

    # Stage B
    HIDDEN_DIM = 512
    NUM_RESIDUAL_BLOCKS = 3
    DROPOUT = 0.1
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 64
    NUM_EPOCHS = 10000

    # Loss
    ALPHA = 1.0
    BETA = 0.5
    GAMMA = 0.5
    TEMPERATURE = 0.07

    # Early Stopping
    PATIENCE = 50

    # =========================================================================
    # Step 1: 데이터 로드
    # =========================================================================
    print("\n" + "=" * 70)
    print(" Step 1: 데이터 로드")
    print("=" * 70)

    if not os.path.exists(input_csv_file):
        print(f"❌ 파일 없음: {input_csv_file}")
        exit()

    df = pd.read_csv(input_csv_file, dtype={'node_embedding': str, 'text_embedding': str})
    print(f"전체 데이터: {len(df)}개")

    mask = (
        df['text_embedding'].notna() &
        (df['text_embedding'].str.strip() != '') &
        (df['text_embedding'].str.strip() != 'None') &
        df['node_embedding'].notna() &
        (df['node_embedding'].apply(lambda x: str(x).strip() not in ('', 'None', 'nan')))
    )
    paired_df = df[mask].copy()
    print(f"학습 가능 (node + text 쌍): {len(paired_df)}개")

    node_embs_all = parse_embedding_column(paired_df['node_embedding'])
    text_embs_all = parse_embedding_column(paired_df['text_embedding'])
    print(f"Node shape: {node_embs_all.shape}, Text shape: {text_embs_all.shape}")

    indices = np.arange(len(node_embs_all))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    node_train, node_val = node_embs_all[train_idx], node_embs_all[val_idx]
    text_train, text_val = text_embs_all[train_idx], text_embs_all[val_idx]
    print(f"Train: {len(train_idx)}개, Val: {len(val_idx)}개")

    # =========================================================================
    # Step 2: Stage A — Ridge Regression
    # =========================================================================
    print("\n" + "=" * 70)
    print(" Step 2: Stage A — Ridge Regression (선형 최적 정렬)")
    print("=" * 70)

    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    ridge.fit(node_train, text_train)

    linear_train = ridge.predict(node_train)
    linear_val = ridge.predict(node_val)

    print("\n  --- Stage A 결과 ---")
    evaluate_projection(linear_train, text_train, "Train")
    evaluate_projection(linear_val, text_val, "Val  ")

    np.savez(ridge_model_path, coef=ridge.coef_, intercept=ridge.intercept_)
    print(f"  Ridge weights 저장: {ridge_model_path}")

    # =========================================================================
    # Step 3: Stage B — Residual Correction
    # =========================================================================
    print("\n" + "=" * 70)
    print(" Step 3: Stage B — Residual MLP Correction")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    train_dataset = AlignmentDataset(node_train, linear_train, text_train)
    val_dataset = AlignmentDataset(node_val, linear_val, text_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ResidualCorrectionModel(
        node_dim=NODE_EMB_DIM,
        text_dim=TEXT_EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        num_residual_blocks=NUM_RESIDUAL_BLOCKS,
        dropout=DROPOUT
    ).to(device)

    criterion = CombinedLoss(alpha=ALPHA, beta=BETA, gamma=GAMMA, temperature=TEMPERATURE).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=residual_model_path)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Loss: α={ALPHA}*MSE + β={BETA}*InfoNCE(τ={TEMPERATURE}) + γ={GAMMA}*Cosine")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_sum = 0
        n_batches = 0

        for node_emb, lin_proj, text_emb in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            node_emb = node_emb.to(device)
            lin_proj = lin_proj.to(device)
            text_emb = text_emb.to(device)

            optimizer.zero_grad()
            output = model(node_emb, lin_proj)
            loss, _, _, _ = criterion(output, text_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = train_loss_sum / n_batches

        model.eval()
        val_loss_sum = 0
        val_batches = 0
        with torch.no_grad():
            for node_emb, lin_proj, text_emb in val_loader:
                node_emb = node_emb.to(device)
                lin_proj = lin_proj.to(device)
                text_emb = text_emb.to(device)
                output = model(node_emb, lin_proj)
                loss, _, _, _ = criterion(output, text_emb)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val = val_loss_sum / max(val_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            scale = torch.sigmoid(model.residual_scale).item()
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:4d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
                  f"ResScale: {scale:.3f} | LR: {lr:.2e}")

        early_stopping(avg_val, model)
        if early_stopping.early_stop:
            print(f"\n  🛑 Early stopping at Epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(residual_model_path))

    # =========================================================================
    # Step 4: 최종 평가 — Stage A vs A+B 자동 선택
    # =========================================================================
    print("\n" + "=" * 70)
    print(" Step 4: 최종 평가 (Stage A vs Stage A+B) → 자동 선택")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        val_node_t = torch.tensor(node_val, dtype=torch.float32).to(device)
        val_lin_t = torch.tensor(linear_val, dtype=torch.float32).to(device)
        final_val = model(val_node_t, val_lin_t).cpu().numpy()

    print("\n  --- Validation Set ---")
    mse_a, cos_a, l2_a = evaluate_projection(linear_val, text_val, "Stage A (Linear only)  ")
    mse_ab, cos_ab, l2_ab = evaluate_projection(final_val, text_val, "Stage A+B (+ Residual) ")

    scale = torch.sigmoid(model.residual_scale).item()
    print(f"\n  Residual scale: {scale:.4f}")

    # ★ 자동 선택: Validation CosSim이 더 높은 쪽을 사용
    use_residual = cos_ab > cos_a
    if use_residual:
        print(f"\n  ✅ Stage A+B 선택 (CosSim: {cos_ab:.4f} > {cos_a:.4f})")
        print(f"     → Residual 보정이 효과적입니다.")
    else:
        print(f"\n  ✅ Stage A only 선택 (CosSim: {cos_a:.4f} ≥ {cos_ab:.4f})")
        print(f"     → 선형 변환만으로 충분합니다. Residual 보정은 overfitting되었습니다.")

    # =========================================================================
    # Step 5: Prospective Node 투영 (자동 선택된 방법 사용)
    # =========================================================================
    print("\n" + "=" * 70)
    print(" Step 5: Prospective Node 투영")
    print("=" * 70)

    prospective_df = df[df['id'].astype(str).str.startswith('prospective node ')].copy()

    if prospective_df.empty:
        print("⚠️ Prospective node 없음. 종료.")
        exit()

    print(f"  Prospective node 수: {len(prospective_df)}개")

    prosp_node_embs = parse_embedding_column(prospective_df['node_embedding'])
    prosp_linear = ridge.predict(prosp_node_embs)

    if use_residual:
        print("  방법: Stage A + Stage B (Residual)")
        model.eval()
        with torch.no_grad():
            prosp_node_t = torch.tensor(prosp_node_embs, dtype=torch.float32).to(device)
            prosp_lin_t = torch.tensor(prosp_linear, dtype=torch.float32).to(device)
            prosp_final = model(prosp_node_t, prosp_lin_t).cpu().numpy()
    else:
        print("  방법: Stage A only (Ridge Regression)")
        prosp_final = prosp_linear

    # =========================================================================
    # Step 6: Nearest Neighbor 검증
    # =========================================================================
    print("\n" + "=" * 70)
    print(" Step 6: 투영 품질 검증 (Nearest Neighbor)")
    print("=" * 70)

    real_text_norm = text_embs_all / (np.linalg.norm(text_embs_all, axis=1, keepdims=True) + 1e-8)

    for i in range(min(len(prosp_final), 5)):
        proj_vec = prosp_final[i]
        proj_norm = proj_vec / (np.linalg.norm(proj_vec) + 1e-8)

        sims = real_text_norm @ proj_norm
        top_k_idx = np.argsort(sims)[-3:][::-1]

        node_id = prospective_df.iloc[i]['id']
        print(f"\n  [{node_id}]")
        for rank, idx in enumerate(top_k_idx):
            neighbor_row = paired_df.iloc[idx]
            sim = sims[idx]
            title = str(neighbor_row.get('title', 'N/A'))[:80]
            print(f"    Top-{rank+1}: sim={sim:.4f} | {neighbor_row['id']} | {title}")

    # =========================================================================
    # Step 7: 저장
    # =========================================================================
    print("\n" + "=" * 70)
    print(" Step 7: 결과 저장")
    print("=" * 70)

    records = []
    for i, row_idx in enumerate(prospective_df.index):
        records.append({
            'id': df.loc[row_idx, 'id'],
            'projected_text_embedding': prosp_final[i].tolist()
        })

    projected_df = pd.DataFrame(records)

    output_file = _ROOT + "/data/prospective_node_projected_embeddings_deep_mlp_ada.csv"
    counter = 1
    while os.path.exists(output_file):
        output_file = f"{_ROOT}/data/prospective_node_projected_embeddings_deep_mlp_{counter}.csv"
        counter += 1

    projected_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 저장 완료: {output_file}")
    print(f"   사용 방법: {'Stage A+B' if use_residual else 'Stage A only (Ridge)'}")
    print(f"   데이터 수: {len(projected_df)}개")