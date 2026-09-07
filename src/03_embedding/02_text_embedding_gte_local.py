# Source file in the working repository: code/7. text_embedding_contriever.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import pandas as pd
import os
import time
from tqdm import tqdm
from datetime import datetime

# --- Sentence-Transformers 및 PyTorch 라이브러리 임포트 ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from sentence_transformers import SentenceTransformer
import torch

# --- 로컬 임베딩 모델 로딩 ---
print("--- 로컬 임베딩 모델 로딩 시작 ---")
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device.upper()}")
model = SentenceTransformer('thenlper/gte-base', device=device)
print("모델 로딩 완료.")
print(f"임베딩 차원: {model.get_sentence_embedding_dimension()}")

BATCH_SIZE = 256


if __name__ == "__main__":
    # =========================================================================
    # 설정: 원본 메타데이터 파일 경로
    # =========================================================================
    metadata_files = [
        (_ROOT + "/data/raw/collected_data_abstract_with_citation.csv", "original_id"),
        (_ROOT + "/data/raw/Cited_patent_data.csv", "id")
    ]

    # =========================================================================
    # Step 1: 원본 CSV들에서 id + abstract 수집
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 1: 원본 메타데이터에서 id + abstract 수집")
    print("=" * 60)

    all_records = {}  # {id: abstract}

    for file_path, id_col in metadata_files:
        if not os.path.exists(file_path):
            print(f"파일 없음: {file_path}")
            continue

        print(f"읽는 중: {file_path}")
        df = pd.read_csv(file_path, dtype={id_col: str}, low_memory=False)
        df.fillna('', inplace=True)

        count = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(file_path)}"):
            current_id = str(row[id_col]).strip()
            abstract = str(row.get('abstract', '')).strip()

            if current_id and abstract and abstract.lower() not in ('', 'nan', 'none'):
                if current_id not in all_records:
                    all_records[current_id] = abstract
                    count += 1
        print(f"   -> {count}개 수집")

    print(f"\n총 {len(all_records)}개의 고유 (id, abstract) 쌍 수집됨")

    if not all_records:
        print("임베딩할 데이터가 없습니다. 스크립트를 종료합니다.")
        exit()

    # =========================================================================
    # Step 2: gte-base로 배치 임베딩 생성
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: thenlper/gte-base로 임베딩 생성")
    print("=" * 60)

    ids_list = list(all_records.keys())
    abstracts_list = list(all_records.values())

    print(f"배치 크기: {BATCH_SIZE}, 총 {len(abstracts_list)}개 텍스트")

    all_embeddings = []
    for i in tqdm(range(0, len(abstracts_list), BATCH_SIZE), desc="Generating Embeddings (batch)"):
        batch_texts = abstracts_list[i:i + BATCH_SIZE]
        batch_embs = model.encode(batch_texts, show_progress_bar=False)
        all_embeddings.extend([emb.tolist() for emb in batch_embs])

    results = []
    for idx in range(len(ids_list)):
        results.append({
            'id': ids_list[idx],
            'abstract': abstracts_list[idx],
            'text_embedding': all_embeddings[idx]
        })

    # =========================================================================
    # Step 3: 저장
    # =========================================================================
    output_file = _ROOT + "/data/text_embeddings_gte.csv"
    counter = 1
    while os.path.exists(output_file):
        output_file = f"{_ROOT}/data/text_embeddings_gte_{counter}.csv"
        counter += 1

    output_df = pd.DataFrame(results)

    success_count = output_df['text_embedding'].notna().sum()
    fail_count = output_df['text_embedding'].isna().sum()
    print(f"\n임베딩 결과: 성공 {success_count}개, 실패 {fail_count}개")

    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"저장 완료: {output_file}")
    print(f"   컬럼: {list(output_df.columns)}")
    print(f"   임베딩 차원: {len(all_embeddings[0]) if all_embeddings else 'N/A'}")