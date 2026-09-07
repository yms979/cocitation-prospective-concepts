# Source file in the working repository: code/2. Link_prediction.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import requests
import json
import time
from datetime import datetime, timedelta

import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # [수정] GATConv -> GCNConv 변경
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from sklearn.model_selection import train_test_split as sk_train_test_split
import xgboost as xgb
import numpy as np
import itertools
import ast

import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# --- [수정] GCN 모델 정의 (Embedding 사용) ---
class GCN(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_channels, out_channels, dropout=0.2):
        super(GCN, self).__init__()
        # One-hot encoding 대신 학습 가능한 임베딩 레이어 사용 (메모리/속도 최적화)
        self.node_embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        
        # [수정] GATConv -> GCNConv, heads 파라미터 제거
        self.conv1 = GCNConv(embedding_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout

    def forward(self, edge_index):
        # 임베딩 생성
        # edge_index의 디바이스에 맞춰 임베딩 인덱스 생성
        node_indices = torch.arange(self.node_embedding.num_embeddings, device=edge_index.device)
        x = self.node_embedding(node_indices)
        
        # [수정] GCN Forward Pass
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# --- [최적화] 피처 생성 함수 (벡터 연산 적용) ---
def get_link_features(node_embeddings, edge_index, degrees_dict):
    # Tensor 입력 시 CPU로 이동
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu()
    
    # 1. 임베딩 피처 (Hadamard Product) - 벡터 연산으로 한 번에 처리
    u_idx, v_idx = edge_index[0], edge_index[1]
    u_emb = node_embeddings[u_idx]
    v_emb = node_embeddings[v_idx]
    embed_feature = (u_emb * v_emb).numpy() # (Num_Edges, Embed_Dim)
    
    # 2. Preferential Attachment (Degree Product)
    # Numpy 벡터화를 위해 리스트 컴프리헨션 사용 (Dict 접근은 빠름)
    u_deg = np.array([degrees_dict.get(i.item(), 0) for i in u_idx])
    v_deg = np.array([degrees_dict.get(i.item(), 0) for i in v_idx])
    pa_feature = (u_deg * v_deg).reshape(-1, 1) # (Num_Edges, 1)
    
    # 두 피처 결합
    return np.hstack([embed_feature, pa_feature])

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    # --- 1. 네트워크 파일 로드 (기존 코드 유지) ---
    print("\n--- 기존 공동 인용 네트워크 파일 로드 시작 ---")
    input_gexf_file = _ROOT + "/data/networks/patent_co_citation_network_filtered.gexf"
    if not os.path.exists(input_gexf_file):
        print(f"오류: 네트워크 파일 '{input_gexf_file}'을(를) 찾을 수 없습니다.")
        exit()
    try:
        graph_for_gat_original_ids = nx.read_gexf(input_gexf_file)
        print(f"'{input_gexf_file}' 파일에서 네트워크를 성공적으로 로드했습니다.")
        print(f"  노드 수: {graph_for_gat_original_ids.number_of_nodes()}")
        print(f"  엣지 수: {graph_for_gat_original_ids.number_of_edges()}")
        if graph_for_gat_original_ids.number_of_nodes() == 0:
            print("로드된 네트워크에 노드가 없어 링크 예측을 수행할 수 없습니다.")
            exit()
    except Exception as e:
        print(f"네트워크 파일 로드 중 오류 발생: {e}")
        exit()

    # --- 2. PyG Data 객체로 변환 ---
    node_mapping = {id: i for i, id in enumerate(graph_for_gat_original_ids.nodes())}
    reverse_node_mapping = {i: id for id, i in node_mapping.items()}
    graph_for_gat = nx.relabel_nodes(graph_for_gat_original_ids, node_mapping)
    
    edge_index = []
    for u, v in graph_for_gat.edges():
        edge_index.append([u, v])
        edge_index.append([v, u])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    num_nodes = graph_for_gat.number_of_nodes()
    
    # [최적화] x = torch.eye(num_nodes) 제거. Data 객체는 구조 정보만 가짐.
    data = Data(edge_index=edge_index, num_nodes=num_nodes)

    # --- 3. 데이터 분할 (기존 코드 유지) ---
    print("\n--- 데이터 분할 (Train/Validation/Test) ---")
    transform = RandomLinkSplit(
        num_val=int(data.edge_index.size(1) * 0.05),
        num_test=int(data.edge_index.size(1) * 0.1),
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=False
    )
    train_data, val_data, test_data = transform(data)
    print("데이터 분할 완료.")

    # --- 4. GCN 모델 학습 (모델 변경) ---
    # 하이퍼파라미터 설정
    embedding_dim = 64     # 임베딩 차원
    hidden_channels = 128  # 은닉층 크기
    out_channels = 64      # 출력층 크기
    # heads = 4            # [삭제] GCN은 head가 없음
    dropout_rate = 0.2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")
    
    # [수정] GCN 모델 초기화 (heads 파라미터 제거)
    model = GCN(data.num_nodes, embedding_dim, hidden_channels, out_channels, dropout=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    patience, min_delta, best_val_loss, patience_counter = 50, 1e-4, float('inf'), 0
    best_model_state = None
    
    print("\n--- GCN 모델 학습 시작 (Early Stopping 적용) ---")
    epochs = 500
    
    # 데이터 GPU 이동
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        pos_edge_index_train = train_data.edge_index
        
        # Negative Sampling
        neg_edge_index_train = negative_sampling(
            edge_index=pos_edge_index_train, 
            num_nodes=data.num_nodes, 
            num_neg_samples=pos_edge_index_train.size(1), 
            method='sparse'
        )
        
        # Forward
        z_train = model(pos_edge_index_train)
        
        pos_score_train = (z_train[pos_edge_index_train[0]] * z_train[pos_edge_index_train[1]]).sum(dim=1)
        neg_score_train = (z_train[neg_edge_index_train[0]] * z_train[neg_edge_index_train[1]]).sum(dim=1)
        
        loss_train = F.binary_cross_entropy_with_logits(pos_score_train, torch.ones_like(pos_score_train)) + \
                     F.binary_cross_entropy_with_logits(neg_score_train, torch.zeros_like(neg_score_train))
        
        loss_train.backward()
        optimizer.step()
        
        # [최적화] 검증을 10 에포크마다 수행
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                z_val = model(val_data.edge_index)
                val_pos_score = (z_val[val_data.pos_edge_label_index[0]] * z_val[val_data.pos_edge_label_index[1]]).sum(dim=1)
                val_neg_score = (z_val[val_data.neg_edge_label_index[0]] * z_val[val_data.neg_edge_label_index[1]]).sum(dim=1)
                
                loss_val = F.binary_cross_entropy_with_logits(val_pos_score, torch.ones_like(val_pos_score)) + \
                           F.binary_cross_entropy_with_logits(val_neg_score, torch.zeros_like(val_neg_score))
            
            print(f'Epoch: {epoch:03d}, Train Loss: {loss_train:.4f}, Val Loss: {loss_val:.4f}')
            
            if loss_val < best_val_loss - min_delta:
                best_val_loss = loss_val
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                    
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("최적의 GCN 모델 상태를 로드했습니다.")

    # 임베딩 추출 (CPU로 이동)
    model.eval()
    with torch.no_grad():
        # 전체 그래프 구조를 넣어 임베딩 생성
        node_embeddings = model(data.edge_index.to(device)).cpu()
    print("\nGCN 임베딩 생성 완료.")

    # --- 5. XGBoost 학습 (최적화된 피처 생성 함수 사용) ---
    percentage_of_positive_edges_for_xgb_training = 0.05
    total_original_graph_edges = graph_for_gat.number_of_edges()
    num_positive_edges_per_xgb_model = int(total_original_graph_edges * percentage_of_positive_edges_for_xgb_training)
    degrees_mapped = dict(graph_for_gat.degree())
    num_xgboost_models = 5
    xgboost_models, xgboost_metrics = [], []
    
    print(f"\n--- {num_xgboost_models}개의 XGBoost 모델 학습 시작 ---")
    all_train_pos_edges = train_data.edge_index.cpu() # CPU로 이동
    
    if num_positive_edges_per_xgb_model > 0 and all_train_pos_edges.size(1) > 0:
        for i in range(num_xgboost_models):
            print(f"  - XGBoost Model {i+1}/{num_xgboost_models}")
            num_pos_to_sample = min(num_positive_edges_per_xgb_model, all_train_pos_edges.size(1))
            perm_pos = torch.randperm(all_train_pos_edges.size(1))
            selected_pos_edges = all_train_pos_edges[:, perm_pos[:num_pos_to_sample]]
            
            # [최적화] 벡터화된 함수 사용
            X_pos = get_link_features(node_embeddings, selected_pos_edges, degrees_mapped)
            y_pos = np.ones(X_pos.shape[0])
            
            selected_neg_edges = negative_sampling(edge_index=data.edge_index.cpu(), num_nodes=data.num_nodes, num_neg_samples=num_pos_to_sample, method='sparse')
            X_neg = get_link_features(node_embeddings, selected_neg_edges, degrees_mapped)
            y_neg = np.zeros(X_neg.shape[0])
            
            X_sub = np.vstack((X_pos, X_neg))
            y_sub = np.hstack((y_pos, y_neg))
            X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = sk_train_test_split(X_sub, y_sub, test_size=0.2, random_state=i, stratify=y_sub)
            
            xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=i)
            xgb_model.fit(X_train_xgb, y_train_xgb)
            xgboost_models.append(xgb_model)
            
            y_pred_xgb = xgb_model.predict(X_test_xgb)
            accuracy, f1 = accuracy_score(y_test_xgb, y_pred_xgb), f1_score(y_test_xgb, y_pred_xgb)
            xgboost_metrics.append({'Model': f'Model {i+1}', 'Accuracy': accuracy, 'F1-Score': f1})
            print(f"    Model {i+1} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    
    # --- 6. 새로운 링크 예측 (배치 처리 적용) ---
    print("\n--- 새로운 링크 예측 수행 시작 ---")
    num_candidate_neg_edges_for_pred = 2000000
    candidate_edges_list_pyg_base = test_data.neg_edge_label_index.cpu()
    
    # 추가 샘플링이 필요한 경우
    num_additional = max(0, num_candidate_neg_edges_for_pred - candidate_edges_list_pyg_base.size(1))
    if num_additional > 0:
        additional = negative_sampling(edge_index=data.edge_index.cpu(), num_nodes=data.num_nodes, num_neg_samples=num_additional, method='sparse')
        candidate_edges_list_pyg = torch.cat([candidate_edges_list_pyg_base, additional], dim=1)
    else:
        candidate_edges_list_pyg = candidate_edges_list_pyg_base

    final_new_links = []
    # 중복 및 기존 엣지 제거 (Numpy 변환 후 처리하여 속도 향상)
    candidate_edges_np = candidate_edges_list_pyg.numpy()
    
    # Set으로 변환하여 중복 제거 (Python set이 빠름)
    candidate_set = set()
    for i in range(candidate_edges_np.shape[1]):
        u, v = candidate_edges_np[0, i], candidate_edges_np[1, i]
        if u == v: continue
        if u > v: u, v = v, u
        if not graph_for_gat.has_edge(u, v):
            candidate_set.add((u, v))

    if not candidate_set:
        print("예측할 후보 엣지가 없습니다.")
    else:
        candidate_edges_for_prediction = torch.tensor(list(candidate_set), dtype=torch.long).t()
        
        if xgboost_models:
            print(f"총 {candidate_edges_for_prediction.size(1)}개의 후보 엣지 예측 중...")
            
            # [최적화] 배치 단위로 예측하여 메모리 및 속도 최적화
            batch_size = 50000
            num_candidates = candidate_edges_for_prediction.size(1)
            
            for i in tqdm(range(0, num_candidates, batch_size), desc="Predicting Batches"):
                batch_edges = candidate_edges_for_prediction[:, i:i+batch_size]
                
                # 배치 피처 생성
                X_predict_batch = get_link_features(node_embeddings, batch_edges, degrees_mapped)
                
                # 모든 모델의 예측 확률 평균 계산
                batch_probs = np.array([model.predict_proba(X_predict_batch)[:, 1] for model in xgboost_models])
                avg_probs = np.mean(batch_probs, axis=0)
                
                # 임계값 필터링 (0.9 이상인 것만)
                # 벡터 연산으로 필터링
                mask = np.all(batch_probs >= 0.5, axis=0)
                indices = np.where(mask)[0]
                
                for idx in indices:
                    u_mapped = batch_edges[0, idx].item()
                    v_mapped = batch_edges[1, idx].item()
                    prob = avg_probs[idx]
                    
                    u_orig = reverse_node_mapping[u_mapped]
                    v_orig = reverse_node_mapping[v_mapped]
                    final_new_links.append((u_orig, v_orig, prob))
        else:
            print("XGBoost 모델이 학습되지 않았습니다.")

    # --- 7. 예측된 링크 선택 및 파일로 저장 (기존 코드 유지) --- 
    if final_new_links:
        # 1. 확률(세 번째 요소)을 기준으로 내림차순 정렬
        final_new_links.sort(key=lambda x: x[2], reverse=True)

        # 2. 노드당 5개 링크 제한 로직
        new_link_counts = defaultdict(int)
        capped_final_links = []
        
        print(f"\n--- 최종 링크 선택 (노드당 최대 5개) ---")
        # 정렬된 링크를 순회하며 조건을 만족하는 링크만 선택
        for link_info in tqdm(final_new_links, desc="Capping Links"):
            u, v, prob = link_info
            
            # 두 노드 모두 링크 수가 5개 미만일 경우에만 링크 추가
            if new_link_counts[u] < 3 and new_link_counts[v] < 3:
                capped_final_links.append(link_info)
                # 두 노드의 카운트를 1씩 증가
                new_link_counts[u] += 1
                new_link_counts[v] += 1
        
        print(f"\n최종 예측된 새로운 링크 수: {len(capped_final_links)}")
        print("예측된 새로운 링크 (상위 10개):")
        for link in capped_final_links[:10]:
            print(f"  {link[0]} <-> {link[1]} (Probability: {link[2]:.4f})")
        
        # 최종적으로 제한된 링크 리스트를 DataFrame으로 변환하여 저장
        new_links_df = pd.DataFrame(capped_final_links, columns=['node1', 'node2', 'probability'])
        output_file_name = f"{_ROOT}/data/predicted_new_links.csv"
        new_links_df.to_csv(output_file_name, index=False)
        print(f"\n제한이 적용된 예측 링크가 '{output_file_name}' 파일로 저장되었습니다. 💾")
    else:
        print("\n예측된 새로운 링크가 없으므로 파일이 저장되지 않았습니다.")