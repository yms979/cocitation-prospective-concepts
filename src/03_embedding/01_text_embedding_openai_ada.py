# Source file in the working repository: code/7. text_embedding.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import pandas as pd
from openai import OpenAI
import os
import time
from tqdm import tqdm
from datetime import datetime


openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수 OPENAI_API_KEY를 설정하거나 코드에 직접 입력해주세요.")

client = OpenAI(api_key=openai_api_key)

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_API_DELAY = 0.02


def get_embedding_with_retry(text: str, max_attempts: int = 5) -> list:
    if not text or not isinstance(text, str) or text.strip() == "":
        return None

    attempts = 0
    while attempts < max_attempts:
        try:
            response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
            time.sleep(EMBEDDING_API_DELAY)
            return response.data[0].embedding
        except Exception as e:
            attempts += 1
            tqdm.write(f"\n[Error] 임베딩 호출 실패 (시도 {attempts}/{max_attempts}): {e}")
            time.sleep(2 ** attempts)

    tqdm.write(f"\n[오류] 최대 재시도 횟수 초과: '{text[:50]}...'")
    return None


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
            print(f"⚠️ 파일 없음: {file_path}")
            continue

        print(f"📖 읽는 중: {file_path}")
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
        print(f"   → {count}개 수집")

    print(f"\n✅ 총 {len(all_records)}개의 고유 (id, abstract) 쌍 수집됨")

    if not all_records:
        print("❌ 임베딩할 데이터가 없습니다. 스크립트를 종료합니다.")
        exit()

    # =========================================================================
    # Step 2: Text embedding 생성
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: OpenAI text-embedding-ada-002로 임베딩 생성")
    print("=" * 60)

    results = []

    for node_id, abstract in tqdm(all_records.items(), desc="Generating Embeddings"):
        embedding = get_embedding_with_retry(abstract)
        results.append({
            'id': node_id,
            'abstract': abstract,
            'text_embedding': embedding  # None if failed
        })

    # =========================================================================
    # Step 3: 저장
    # =========================================================================
    output_file = _ROOT + "/data/text_embeddings_ada.csv"
    counter = 1
    while os.path.exists(output_file):
        output_file = f"{_ROOT}/data/text_embeddings_ada_{counter}.csv"
        counter += 1

    output_df = pd.DataFrame(results)

    success_count = output_df['text_embedding'].notna().sum()
    fail_count = output_df['text_embedding'].isna().sum()
    print(f"\n임베딩 결과: 성공 {success_count}개, 실패 {fail_count}개")

    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 저장 완료: {output_file}")
    print(f"   컬럼: {list(output_df.columns)}")