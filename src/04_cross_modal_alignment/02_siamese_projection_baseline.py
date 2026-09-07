# Source file in the working repository: code/10. Siamese Neural Networks.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import ast # CSV에서 임베딩 리스트를 파싱하기 위해 필요
from tqdm import tqdm # 진행률 바를 위해 필요
from sklearn.model_selection import train_test_split # 데이터 분할을 위해 추가

# --- 1. 데이터셋 정의 ---
class NodeEmbeddingDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        # 'text_embedding'과 'node_embedding' 컬럼이 리스트 형태의 문자열로 저장되어 있을 수 있으므로 ast.literal_eval 적용
        self.node_embeddings = torch.tensor(
            [ast.literal_eval(x) if isinstance(x, str) else x for x in df['node_embedding'].tolist()],
            dtype=torch.float32
        )
        self.text_embeddings = torch.tensor(
            [ast.literal_eval(x) if isinstance(x, str) else x for x in df['text_embedding'].tolist()],
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.node_embeddings)

    def __getitem__(self, idx):
        return self.node_embeddings[idx], self.text_embeddings[idx]

# --- 2. 전문적인 Autoencoder 모델 정의 ---
class ProfessionalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int, hidden_dims: list, dropout_rate: float = 0.2):
        super(ProfessionalAutoencoder, self).__init__()

        # Encoder
        encoder_layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim)) # Batch Normalization
            encoder_layers.append(nn.LeakyReLU(0.01))    # Leaky ReLU activation
            encoder_layers.append(nn.Dropout(dropout_rate)) # Dropout
            current_dim = h_dim
        
        # Latent layer (인코딩된 차원)
        encoder_layers.append(nn.Linear(current_dim, encoding_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        current_dim = encoding_dim
        # hidden_dims를 역순으로 사용하여 디코더 구성
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.LeakyReLU(0.01))
            decoder_layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        
        # Decoder의 출력 레이어
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# --- 3. Early Stopping 클래스 정의 ---
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path=_ROOT + '/data/best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf # np.Inf 대신 np.inf 사용
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''검증 손실이 감소하면 모델을 저장합니다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    # --- 설정 ---
    input_csv_file = _ROOT + "/data/node_embeddings_with_text_embedding.csv" # <--- 실제 파일명으로 변경!
    model_save_path = _ROOT + "/data/best_autoencoder_model.pth" # 최적 모델 저장 경로

    # 모델 파라미터
    NODE_EMB_DIM = 1024        # 기존 'node_embedding' 컬럼의 차원
    TEXT_EMB_DIM = 1536        # 'text-embedding-ada-002' 모델의 임베딩 차원
    COMMON_EMB_DIM = TEXT_EMB_DIM # 목표 투영 공간의 차원 (text-embedding-ada-002 차원과 일치시킴)

    # Autoencoder를 위한 Hidden Layer 차원
    # [인코더 첫 레이어 출력 차원, 인코더 두 번째 레이어 출력 차원, ...]
    AE_HIDDEN_DIMS = [768, 512, 256] # 예시: 1024 -> 768 -> 512 -> 256 -> 1536 (인코딩 차원)
                                     # 디코더는 역순: 1536 -> 256 -> 512 -> 768 -> 1024 (원본 차원)
    DROPOUT_RATE = 0.3 # Dropout 비율

    LEARNING_RATE = 0.0005 # 학습률 조정
    BATCH_SIZE = 64
    NUM_EPOCHS = 300 # Early Stopping을 위해 에폭 충분히 증가

    # 손실 가중치
    RECONSTRUCTION_LOSS_WEIGHT = 0.2
    PROJECTION_LOSS_WEIGHT = 0.8
    
    # Early Stopping 파라미터
    EARLY_STOPPING_PATIENCE = 20 # 검증 손실 개선이 없을 때 기다릴 에폭 수
    EARLY_STOPPING_VERBOSE = True # Early Stopping 정보 출력 여부

    # --- 1. 데이터 로드 및 준비 ---
    if not os.path.exists(input_csv_file):
        print(f"Error: Input CSV file not found at {input_csv_file}. Please ensure it exists.")
        exit()

    print(f"\n--- 입력 CSV 파일 로드 및 데이터 준비 시작 ---")
    try:
        df = pd.read_csv(input_csv_file, dtype={'node_embedding': str, 'text_embedding': str})
        print(f"파일 로드 완료. 총 {len(df)}개 노드 데이터 발견.")

        train_df_filtered = df[df['text_embedding'].notna() & (df['text_embedding'].str.strip() != '') &
                               df['node_embedding'].notna() & (df['node_embedding'].apply(lambda x: str(x).strip() != ''))].copy()

        if train_df_filtered.empty:
            print("훈련에 사용할 데이터(텍스트 임베딩과 노드 임베딩이 모두 존재하는 'W' 노드)가 없습니다. 스크립트를 종료합니다.")
            exit()
            
        print(f"전체 훈련 및 검증 대상 데이터 {len(train_df_filtered)}개 샘플 사용.")

        # 훈련 데이터를 훈련 세트와 검증 세트로 분할 (예: 80% 훈련, 20% 검증)
        train_data, val_data = train_test_split(train_df_filtered, test_size=0.2, random_state=42)
        
        print(f"훈련 세트: {len(train_data)}개 샘플")
        print(f"검증 세트: {len(val_data)}개 샘플")


    except Exception as e:
        print(f"Error loading or preparing data from {input_csv_file}: {e}")
        print("파일이 손상되었거나 필요한 컬럼이 누락되었을 수 있습니다. 스크립트를 종료합니다.")
        exit()

    # 데이터셋과 DataLoader 생성
    train_dataset = NodeEmbeddingDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = NodeEmbeddingDataset(val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # 검증 시에는 셔플하지 않음

    # --- 2. Autoencoder 모델 초기화 및 학습 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ProfessionalAutoencoder(
        input_dim=NODE_EMB_DIM,
        encoding_dim=COMMON_EMB_DIM,
        hidden_dims=AE_HIDDEN_DIMS,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    reconstruction_criterion = nn.MSELoss()
    projection_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Early Stopping 인스턴스 생성
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        verbose=EARLY_STOPPING_VERBOSE,
        path=model_save_path
    )

    print(f"\n--- Autoencoder 학습 시작 (Early Stopping 적용, Epochs: {NUM_EPOCHS}) ---")

    for epoch in range(NUM_EPOCHS):
        # 훈련 단계
        model.train()
        total_train_loss = 0
        for batch_node_emb, batch_text_emb in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)"):
            batch_node_emb, batch_text_emb = batch_node_emb.to(device), batch_text_emb.to(device)

            optimizer.zero_grad()
            encoded_node_emb, decoded_node_emb = model(batch_node_emb)

            reconstruction_loss = reconstruction_criterion(decoded_node_emb, batch_node_emb)
            projection_loss = projection_criterion(encoded_node_emb, batch_text_emb)
            loss = (RECONSTRUCTION_LOSS_WEIGHT * reconstruction_loss) + \
                   (PROJECTION_LOSS_WEIGHT * projection_loss)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)

        # 검증 단계
        model.eval() # 평가 모드
        total_val_loss = 0
        with torch.no_grad(): # 역전파 비활성화
            for batch_node_emb, batch_text_emb in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Validation)"):
                batch_node_emb, batch_text_emb = batch_node_emb.to(device), batch_text_emb.to(device)
                
                encoded_node_emb, decoded_node_emb = model(batch_node_emb)
                
                reconstruction_loss = reconstruction_criterion(decoded_node_emb, batch_node_emb)
                projection_loss = projection_criterion(encoded_node_emb, batch_text_emb)
                val_loss = (RECONSTRUCTION_LOSS_WEIGHT * reconstruction_loss) + \
                           (PROJECTION_LOSS_WEIGHT * projection_loss)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        
        print(f"Epoch {epoch+1} 완료, 훈련 손실: {avg_train_loss:.4f}, 검증 손실: {avg_val_loss:.4f}")

        # Early Stopping 체크
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping!")
            break
            
    print("\nAutoencoder 학습 완료.")

    # --- 3. 가상 노드 임베딩 투영 (최적의 모델 로드 후 사용) ---
    print("\n--- 가상 노드 임베딩을 텍스트 임베딩 공간으로 투영 시작 ---")
    
    # 최적의 모델 가중치 로드
    model.load_state_dict(torch.load(model_save_path))
    print(f"최적의 모델 가중치를 '{model_save_path}'에서 로드했습니다.")

    # 가상 노드 필터링
    prospective_nodes_df = df[df['id'].astype(str).str.startswith('prospective_node_')].copy()

    if prospective_nodes_df.empty:
        print("네트워크에서 투영할 가상 노드를 찾을 수 없습니다. 스크립트를 종료합니다.")
        exit()

    # 가상 노드의 기존 'node_embedding' 추출
    prospective_node_embeddings = torch.tensor(
        [ast.literal_eval(x) if isinstance(x, str) else x for x in prospective_nodes_df['node_embedding'].tolist()],
        dtype=torch.float32
    ).to(device)

    model.eval() # 모델을 평가 모드로 전환
    with torch.no_grad(): # 역전파 비활성화
        projected_prospective_embeddings = model.encoder(prospective_node_embeddings).cpu().numpy()

    print(f"가상 노드 {len(prospective_nodes_df)}개의 임베딩 투영 완료. 투영된 임베딩 차원: {projected_prospective_embeddings.shape[1]}")

    # --- 4. 투영된 임베딩을 CSV에 저장 ---
    print("\n--- 투영된 가상 노드 임베딩을 새로운 CSV 파일에 저장 ---")

    projected_embeddings_records = []
    for i, row_index in enumerate(prospective_nodes_df.index):
        original_prospective_node_id = df.loc[row_index, 'id']
        projected_embedding = projected_prospective_embeddings[i].tolist()

        projected_embeddings_records.append({
            'id': original_prospective_node_id,
            'projected_text_embedding': projected_embedding
        })

    projected_df = pd.DataFrame(projected_embeddings_records)
    
    # 원본 코드와 동일한 파일명 규칙 사용
    output_csv_file = f"{_ROOT}/data/prospective_node_projected_embeddings.csv"

    # 파일이 이미 존재하면 이름 뒤에 숫자를 붙여 덮어쓰기 방지
    counter = 1
    while os.path.exists(output_csv_file):
        output_csv_file = f"{_ROOT}/data/prospective_node_projected_embeddings_{counter}.csv"
        counter += 1

    try:
        projected_df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
        print(f"\n투영된 가상 노드 임베딩이 '{output_csv_file}' 파일로 저장되었습니다.")
        print(f"이 파일에는 'id', 'projected_text_embedding' 컬럼이 포함됩니다.")
    except Exception as e:
        print(f"Error saving projected embeddings to CSV: {e}")
        print("CSV 파일 저장 중 오류가 발생했습니다. 권한 또는 경로 문제일 수 있습니다.")