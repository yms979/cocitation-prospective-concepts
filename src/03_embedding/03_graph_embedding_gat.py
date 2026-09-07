"""
GAT Node Embedding with Text Embedding Initialization
=====================================================
텍스트 임베딩(ada-002, 1536dim)을 GAT의 초기 노드 피처로 사용하여,
의미 정보 + 그래프 구조 정보를 동시에 학습하는 스크립트.

변경 사항 요약:
1. 사전 계산된 text embedding을 초기 feature matrix로 로드
2. Abstract가 있는 노드 → text embedding (frozen or fine-tunable)
3. Abstract가 없는 노드 (prospective 등) → learnable embedding
4. Projection layer: 1536dim → GAT input dim으로 축소 후 GAT 통과
"""
# Source file in the working repository: code/9. Emb_expanded_citation_net.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import networkx as nx
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, negative_sampling
from torch_geometric.loader import LinkNeighborLoader
import numpy as np
from tqdm import tqdm
import copy
import ast


# =============================================================================
# 1. GAT 모델 정의 (Text Embedding 초기화 버전)
# =============================================================================
class GATWithTextInit(nn.Module):
    """
    Text embedding을 초기 노드 피처로 활용하는 GAT 모델.
    
    구조:
        [Initial Features] → [Projection] → [GAT Layer] → [Output Embedding]
        
    - Abstract가 있는 노드: text_embedding (1536dim)을 사용
    - Abstract가 없는 노드: learnable embedding을 사용
    - Projection layer가 두 종류의 입력을 동일한 차원으로 맞춤
    """
    def __init__(
        self,
        num_nodes: int,
        text_embedding_dim: int,    # ada-002 = 1536
        projection_dim: int,        # projection 후 차원 (e.g., 256)
        out_channels: int,          # 최종 출력 차원 (e.g., 1024)
        heads: int = 4,
        dropout: float = 0.2,
        freeze_text_embeddings: bool = False  # text embedding 고정 여부
    ):
        super(GATWithTextInit, self).__init__()
        
        self.num_nodes = num_nodes
        self.text_embedding_dim = text_embedding_dim
        self.freeze_text_embeddings = freeze_text_embeddings
        
        # --- 초기 피처 저장용 버퍼/파라미터 ---
        # text embedding이 있는 노드의 feature (학습 가능 여부 선택)
        # register_buffer: 학습하지 않음 / nn.Parameter: 학습함
        # 초기에는 zero로 설정, 나중에 load_initial_features()로 채움
        if freeze_text_embeddings:
            self.register_buffer('text_features', torch.zeros(num_nodes, text_embedding_dim))
        else:
            self.text_features = nn.Parameter(torch.zeros(num_nodes, text_embedding_dim))
        
        # text embedding이 없는 노드를 위한 learnable embedding
        self.fallback_embedding = nn.Embedding(num_nodes, text_embedding_dim)
        nn.init.xavier_uniform_(self.fallback_embedding.weight)
        
        # 어떤 노드가 text embedding을 가지고 있는지 표시하는 마스크
        self.register_buffer('has_text_mask', torch.zeros(num_nodes, dtype=torch.bool))
        
        # --- Projection Layer ---
        # 1536dim → projection_dim으로 축소 (정보 압축 + 차원 맞춤)
        self.projection = nn.Sequential(
            nn.Linear(text_embedding_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # --- GAT Layer ---
        self.conv1 = GATConv(
            projection_dim, out_channels,
            heads=1, concat=False, dropout=dropout
        )
        
        self.dropout_rate = dropout
    
    def load_initial_features(self, node_text_embeddings: dict, node_to_pyg_idx: dict):
        """
        사전 계산된 text embedding을 모델에 로드합니다.
        
        Args:
            node_text_embeddings: {original_node_id: [1536-dim list]} 매핑
            node_to_pyg_idx: {original_node_id: pyg_index} 매핑
        """
        loaded_count = 0
        for node_id, embedding in node_text_embeddings.items():
            if node_id in node_to_pyg_idx and embedding is not None:
                pyg_idx = node_to_pyg_idx[node_id]
                emb_tensor = torch.tensor(embedding, dtype=torch.float32)
                
                if self.freeze_text_embeddings:
                    self.text_features[pyg_idx] = emb_tensor
                else:
                    self.text_features.data[pyg_idx] = emb_tensor
                
                self.has_text_mask[pyg_idx] = True
                loaded_count += 1
        
        print(f"   ✅ {loaded_count}/{self.num_nodes}개 노드에 text embedding 로드 완료")
        print(f"   📝 {self.num_nodes - loaded_count}개 노드는 learnable embedding 사용")
    
    def get_initial_features(self, n_id=None):
        """
        각 노드의 초기 feature를 반환합니다.
        - text embedding이 있는 노드 → text_features
        - 없는 노드 → fallback_embedding
        """
        if n_id is not None:
            # Mini-batch: n_id에 해당하는 노드만
            mask = self.has_text_mask[n_id]  # [batch_size]
            
            text_feat = self.text_features[n_id]        # [batch_size, 1536]
            fallback_feat = self.fallback_embedding(n_id)  # [batch_size, 1536]
            
            # mask가 True인 곳은 text_feat, False인 곳은 fallback_feat
            features = torch.where(
                mask.unsqueeze(-1).expand_as(text_feat),
                text_feat,
                fallback_feat
            )
        else:
            # Full-batch: 모든 노드
            mask = self.has_text_mask  # [num_nodes]
            
            text_feat = self.text_features                          # [num_nodes, 1536]
            fallback_feat = self.fallback_embedding.weight          # [num_nodes, 1536]
            
            features = torch.where(
                mask.unsqueeze(-1).expand_as(text_feat),
                text_feat,
                fallback_feat
            )
        
        return features
    
    def forward(self, data):
        # 1. 초기 feature 구성 (text embedding + fallback)
        if hasattr(data, 'n_id'):
            x = self.get_initial_features(data.n_id)
        else:
            x = self.get_initial_features()
        
        # 2. Projection: 1536 → projection_dim
        x = self.projection(x)
        
        # 3. GAT: projection_dim → out_channels
        x = self.conv1(x, data.edge_index)
        
        return x


# =============================================================================
# 2. Text Embedding 로드 유틸리티
# =============================================================================
def load_text_embeddings_from_csv(csv_path: str, id_col: str = 'id') -> dict:
    """
    text embedding이 포함된 CSV 파일에서 {node_id: embedding_list} 딕셔너리를 생성합니다.
    """
    if not os.path.exists(csv_path):
        print(f"⚠️ Text embedding 파일을 찾을 수 없습니다: {csv_path}")
        return {}
    
    print(f"📖 Text embedding 로드 중: {csv_path}")
    df = pd.read_csv(csv_path, dtype={id_col: str, 'text_embedding': str})
    
    text_embeddings = {}
    loaded = 0
    skipped = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading text embeddings"):
        node_id = str(row[id_col]).strip()
        text_emb_str = row.get('text_embedding', None)
        
        if pd.isna(text_emb_str) or text_emb_str is None or str(text_emb_str).strip() in ('', 'None', 'nan'):
            skipped += 1
            continue
        
        try:
            embedding = ast.literal_eval(str(text_emb_str))
            if isinstance(embedding, list) and len(embedding) > 0:
                text_embeddings[node_id] = embedding
                loaded += 1
            else:
                skipped += 1
        except (ValueError, SyntaxError):
            skipped += 1
    
    print(f"   ✅ {loaded}개 text embedding 로드, {skipped}개 스킵")
    return text_embeddings


# =============================================================================
# 3. 메인 실행
# =============================================================================
if __name__ == "__main__":
    
    # =========================================================================
    # 설정
    # =========================================================================
    network_file = _ROOT + "/data/networks/network_with_prospective_nodes.gexf"
    
    # Text embedding 파일 (generate_text_embeddings.py의 출력 파일)
    text_embedding_file = _ROOT + "/data/text_embeddings_ada.csv"
    
    # 메타데이터 파일 목록 (title, abstract 정보)
    metadata_files = [
        (_ROOT + "/data/raw/collected_data_abstract_with_citation.csv", "original_id"),
        (_ROOT + "/data/raw/Cited_patent_data.csv", "id")
    ]
    
    # 하이퍼파라미터
    TEXT_EMBEDDING_DIM = 1536       # ada-002 (setting used for the reported results); 768 for the gte-base robustness run
    PROJECTION_DIM = 256            # projection 후 차원
    GAT_OUT_CHANNELS = 1024         # 최종 노드 임베딩 차원
    GAT_HEADS = 4
    GAT_DROPOUT = 0.2
    GAT_EPOCHS = 10000
    GAT_LR = 0.005
    BATCH_SIZE = 32
    NUM_NEIGHBORS = [10, 5]
    FREEZE_TEXT_EMBEDDINGS = True    # True: text embedding 고정, False: fine-tune
    
    # Early Stopping
    PATIENCE = 20
    MIN_DELTA = 0.0001

    # =========================================================================
    # Step 1: 메타데이터 로드 (title, abstract)
    # =========================================================================
    paper_info_cache = {}
    print("\n" + "=" * 60)
    print("Step 1: 메타데이터(Title, Abstract) 로드")
    print("=" * 60)

    for file_path, id_col in metadata_files:
        if os.path.exists(file_path):
            print(f"📖 파일 읽는 중: {file_path}")
            try:
                df = pd.read_csv(file_path, dtype={id_col: str}, low_memory=False)
                df.fillna('', inplace=True)
                
                loaded_count = 0
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Caching {os.path.basename(file_path)}"):
                    current_id = str(row[id_col]).strip()
                    title = str(row.get('title', '')).strip()
                    abstract = str(row.get('abstract', '')).strip()
                    
                    if current_id:
                        if current_id not in paper_info_cache or not paper_info_cache[current_id]['title']:
                            paper_info_cache[current_id] = {'title': title, 'abstract': abstract}
                            loaded_count += 1
                print(f"   → {loaded_count}개 처리 완료")
            except Exception as e:
                print(f"   → ❌ 오류: {e}")
        else:
            print(f"   → ⚠️ 파일 없음: {file_path}")

    print(f"✅ 총 {len(paper_info_cache)}개 특허 정보 준비됨")

    # =========================================================================
    # Step 2: Text Embedding 로드
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: 사전 계산된 Text Embedding 로드")
    print("=" * 60)
    
    text_embeddings = load_text_embeddings_from_csv(text_embedding_file, id_col='id')
    
    if not text_embeddings:
        print("⚠️ Text embedding이 로드되지 않았습니다.")
        print("   먼저 text embedding 생성 스크립트를 실행해주세요.")
        print("   Learnable embedding만 사용하여 계속 진행합니다.")

    # =========================================================================
    # Step 3: 네트워크 로드 및 PyG 데이터 생성
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3: 네트워크 로드 및 변환")
    print("=" * 60)
    
    if not os.path.exists(network_file):
        print(f"❌ 네트워크 파일 '{network_file}'이 없습니다.")
        exit()
    
    G = nx.read_gexf(network_file)
    print(f"   노드: {G.number_of_nodes()}, 엣지: {G.number_of_edges()}")
    
    if G.number_of_nodes() == 0:
        exit()

    original_node_ids = list(G.nodes())
    node_to_pyg_idx = {node_id: i for i, node_id in enumerate(original_node_ids)}
    pyg_idx_to_node = {i: node_id for node_id, i in node_to_pyg_idx.items()}
    
    # PyG 데이터 생성
    edge_index_list = [
        [node_to_pyg_idx[u], node_to_pyg_idx[v]]
        for u, v in G.edges()
        if u in node_to_pyg_idx and v in node_to_pyg_idx
    ]
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    data = Data(edge_index=edge_index, num_nodes=len(original_node_ids))
    data.edge_index = to_undirected(data.edge_index)

    # 학습/검증 분할
    num_edges = data.edge_index.size(1) // 2
    num_val = int(num_edges * 0.05)
    perm = torch.randperm(num_edges)
    edge_index_shuffled = data.edge_index[:, ::2][:, perm]
    val_pos_edge_index = edge_index_shuffled[:, :num_val]
    train_pos_edge_index = edge_index_shuffled[:, num_val:]
    train_pos_edge_index = torch.cat([train_pos_edge_index, train_pos_edge_index.flip(0)], dim=1)

    train_data = Data(edge_index=train_pos_edge_index, num_nodes=data.num_nodes)
    val_data = Data(
        edge_index=train_pos_edge_index, num_nodes=data.num_nodes,
        pos_edge_label_index=val_pos_edge_index,
        neg_edge_label_index=negative_sampling(data.edge_index, data.num_nodes, num_val)
    )

    # =========================================================================
    # Step 4: 모델 초기화 및 Text Embedding 로드
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 4: GAT 모델 초기화 (Text Embedding 초기화)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    torch.cuda.empty_cache()
    
    model = GATWithTextInit(
        num_nodes=data.num_nodes,
        text_embedding_dim=TEXT_EMBEDDING_DIM,
        projection_dim=PROJECTION_DIM,
        out_channels=GAT_OUT_CHANNELS,
        heads=GAT_HEADS,
        dropout=GAT_DROPOUT,
        freeze_text_embeddings=FREEZE_TEXT_EMBEDDINGS
    ).to(device)
    
    # ★ 핵심: 사전 계산된 text embedding을 모델에 로드
    if text_embeddings:
        model.load_initial_features(text_embeddings, node_to_pyg_idx)
    
    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")
    print(f"   Text embedding frozen: {FREEZE_TEXT_EMBEDDINGS}")

    # =========================================================================
    # Step 5: 학습
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 5: GAT 학습 시작")
    print("=" * 60)
    
    train_loader = LinkNeighborLoader(
        data=train_data, num_neighbors=NUM_NEIGHBORS,
        neg_sampling_ratio=1.0, batch_size=BATCH_SIZE, shuffle=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=GAT_LR)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, GAT_EPOCHS + 1):
        # --- Train ---
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch:03d}', leave=False):
            batch.to(device)
            optimizer.zero_grad()
            z = model(batch)
            
            curr_bs = batch.edge_label_index.size(1) // 2
            pos_score = (z[batch.edge_label_index[0, :curr_bs]] * z[batch.edge_label_index[1, :curr_bs]]).sum(dim=1)
            neg_score = (z[batch.edge_label_index[0, curr_bs:]] * z[batch.edge_label_index[1, curr_bs:]]).sum(dim=1)
            
            loss = (
                F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) +
                F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            z_val = model(val_data.to(device))
            pos = (z_val[val_data.pos_edge_label_index[0]] * z_val[val_data.pos_edge_label_index[1]]).sum(dim=1)
            neg = (z_val[val_data.neg_edge_label_index[0]] * z_val[val_data.neg_edge_label_index[1]]).sum(dim=1)
            val_loss = (
                F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) +
                F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
            )
            curr_val_loss = val_loss.item()

        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d} | Train: {avg_train_loss:.4f} | Val: {curr_val_loss:.4f} | Best: {best_val_loss:.4f}')

        # --- Early Stopping ---
        if curr_val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = curr_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early stopping at Epoch {epoch}")
                break
    
    if best_model_state:
        print("🔄 Best weights 복원 중...")
        model.load_state_dict(best_model_state)

    # =========================================================================
    # Step 6: 최종 임베딩 추출 및 저장
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 6: 최종 노드 임베딩 추출 및 저장")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        node_embeddings = model(data.to(device)).cpu().numpy()

    output_records = []
    missing_info_count = 0
    skipped_count = 0

    for pyg_idx in tqdm(range(data.num_nodes), desc="Mapping Data"):
        original_node_id = str(pyg_idx_to_node[pyg_idx])
        embedding = node_embeddings[pyg_idx]
        
        info = paper_info_cache.get(original_node_id, {'title': '', 'abstract': ''})
        title = info['title']
        abstract = info['abstract']
        
        # 해당 노드의 text embedding 가져오기 (없으면 None)
        text_emb = text_embeddings.get(original_node_id, None)
        
        # Prospective node
        if original_node_id.startswith("prospective") and not title:
            title = None
            abstract = None
        elif not title:
            missing_info_count += 1
            title = "Title not found in CSV"
            abstract = "No Abstract Found"
        
        # 필터링: abstract 없고 prospective도 아닌 노드 제외
        is_missing_abstract = (abstract == "No Abstract Found")
        is_prospective = ("prospective" in original_node_id)
        
        if is_missing_abstract and not is_prospective:
            skipped_count += 1
            continue

        output_records.append({
            'id': original_node_id,
            'title': title,
            'abstract': abstract,
            'node_embedding': embedding.tolist(),
            'text_embedding': text_emb
        })

    if missing_info_count > 0:
        print(f"\n⚠️ {missing_info_count}개 노드 정보 없음, {skipped_count}개 필터링 제외")
    
    # 파일 저장
    output_csv_file = _ROOT + "/data/node_embeddings_with_text_embedding_ada.csv"
    file_idx = 1
    while os.path.exists(output_csv_file):
        output_csv_file = f"{_ROOT}/data/node_embeddings_with_text_embedding_ada_{file_idx}.csv"
        file_idx += 1

    try:
        output_df = pd.DataFrame(output_records)
        if not output_df.empty:
            output_df = output_df[['id', 'title', 'abstract', 'node_embedding', 'text_embedding']]
            output_df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
            print(f"\n✅ 저장 완료: {output_csv_file}")
            print(f"   컬럼: {list(output_df.columns)}")
            print(f"   총 {len(output_df)}개 데이터")
        else:
            print("\n⚠️ 저장할 데이터가 없습니다.")
    except Exception as e:
        print(f"❌ CSV 저장 실패: {e}")