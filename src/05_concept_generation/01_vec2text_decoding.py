"""
Embedding Inversion: Projected Text Embedding → Natural Language Text
=====================================================================
Vec2text를 사용하여 투영된 임베딩을 자연어로 복원합니다.

주요 개선 사항 (기존 대비):
1. SEQUENCE_BEAM_WIDTH 증가 (2 → 4 이상) — 즉각적 품질 향상
2. 생성된 텍스트를 다시 임베딩하여 projected embedding과 비교 (round-trip 검증)
3. 실제 특허 DB에서 nearest neighbor를 찾아 생성 결과와 비교
4. 품질 메트릭 자동 출력 (cosine similarity, 텍스트 길이 등)
"""
# Source file in the working repository: code/11. vec2text.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 물리적 GPU 1번 사용

import pandas as pd
import numpy as np
import torch
import ast
import gc
import time
from tqdm import tqdm

try:
    import vec2text
    from openai import OpenAI
except ImportError as e:
    print(f"필수 라이브러리 누락: {e}")
    exit()


# =============================================================================
# 설정
# =============================================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# 파일 경로
INPUT_FILE = _ROOT + "/data/prospective_node_projected_embeddings_deep_mlp_ada.csv"
OUTPUT_FILE = _ROOT + "/data/final_decoded_results.csv"

# 원본 데이터 (nearest neighbor 비교용, 없어도 동작함)
REFERENCE_FILE = _ROOT + "/data/node_embeddings_with_text_embedding_ada.csv"

# GPU
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    print(f"✅ GPU: 물리적 GPU 1번 (Internal: cuda:0)")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    DEVICE = torch.device('cpu')
    print("⚠️ CPU 사용")

# =============================================================================
# ★ Vec2text 파라미터 (핵심 개선)
# =============================================================================
# SEQUENCE_BEAM_WIDTH: 클수록 탐색 공간이 넓어져 품질 향상
#   - 2 (기존): 거의 greedy, 반복적이고 애매한 텍스트
#   - 4: 적당한 개선, VRAM ~12GB
#   - 8: 큰 개선, VRAM ~20GB
#   - GPU 여유에 따라 조절하세요
NUM_STEPS = 60
SEQUENCE_BEAM_WIDTH = 4     # ★ 2 → 4 (최소 권장), GPU 여유시 8
BATCH_SIZE = 1              # beam width가 크면 메모리 때문에 1 유지

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# =============================================================================
# 유틸리티
# =============================================================================
def get_embedding(client: OpenAI, text: str, max_retries: int = 3):
    """OpenAI ada-002 임베딩 생성 (재시도 포함)"""
    text = str(text).replace("\n", " ").strip()
    if not text:
        return None
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(input=[text], model="text-embedding-ada-002")
            time.sleep(0.02)
            return resp.data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                print(f"  임베딩 실패: {e}")
                return None


def cosine_sim(v1, v2):
    """두 벡터의 cosine similarity"""
    if v1 is None or v2 is None:
        return np.nan
    v1, v2 = np.array(v1), np.array(v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(np.dot(v1, v2) / norm)


def parse_embedding(val):
    """문자열 또는 리스트를 numpy array로 변환"""
    if isinstance(val, str):
        return np.array(ast.literal_eval(val), dtype=np.float32)
    elif isinstance(val, list):
        return np.array(val, dtype=np.float32)
    return val


# =============================================================================
# Step 1: Embedding Inversion (vec2text)
# =============================================================================
def invert_embeddings(input_file: str, output_file: str) -> str:
    print("\n" + "=" * 70)
    print(" Step 1: Embedding → Text 복원 (vec2text)")
    print("=" * 70)
    print(f"  NUM_STEPS: {NUM_STEPS}")
    print(f"  SEQUENCE_BEAM_WIDTH: {SEQUENCE_BEAM_WIDTH}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")

    if not os.path.exists(input_file):
        print(f"❌ 파일 없음: {input_file}")
        return None

    df = pd.read_csv(input_file)

    if 'projected_text_embedding' not in df.columns:
        print(f"❌ 'projected_text_embedding' 컬럼 없음. 컬럼: {df.columns.tolist()}")
        return None

    # 임베딩 파싱
    if isinstance(df['projected_text_embedding'].iloc[0], str):
        df['projected_text_embedding'] = df['projected_text_embedding'].apply(ast.literal_eval)

    print(f"  대상: {len(df)}개 prospective node")

    # Corrector 로드
    print("  Corrector 모델 로딩 중...")
    try:
        corrector = vec2text.load_pretrained_corrector("text-embedding-ada-002")
        if hasattr(corrector, "model"):
            corrector.model.to(DEVICE)
            print("  ✅ Corrector → GPU 이동 완료")
    except Exception as e:
        print(f"  ❌ 모델 로드 실패: {e}")
        return None

    embeddings_tensor = torch.tensor(
        df['projected_text_embedding'].tolist(),
        dtype=torch.float32
    )

    decoded_results = []
    decode_times = []

    with torch.no_grad():
        for i in tqdm(range(0, embeddings_tensor.size(0), BATCH_SIZE), desc="Inverting"):
            batch_cpu = embeddings_tensor[i: i + BATCH_SIZE]
            batch_gpu = batch_cpu.to(DEVICE)

            start_time = time.time()

            try:
                inverted_texts = vec2text.invert_embeddings(
                    embeddings=batch_gpu,
                    corrector=corrector,
                    num_steps=NUM_STEPS,
                    sequence_beam_width=SEQUENCE_BEAM_WIDTH
                )
                decoded_results.extend(inverted_texts)
                elapsed = time.time() - start_time
                decode_times.append(elapsed)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n  ⚠️ OOM at batch {i}. beam_width를 줄여보세요.")
                    decoded_results.extend(["ERROR_OOM"] * len(batch_gpu))
                    torch.cuda.empty_cache()
                else:
                    print(f"\n  ⚠️ Runtime error: {e}")
                    decoded_results.extend(["ERROR"] * len(batch_gpu))
            except Exception as e:
                print(f"\n  ⚠️ Error: {e}")
                decoded_results.extend(["ERROR"] * len(batch_gpu))

            del batch_gpu
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

    df['decoded_text'] = decoded_results

    # 생성 통계
    success = sum(1 for t in decoded_results if t not in ("ERROR", "ERROR_OOM", ""))
    print(f"\n  결과: {success}/{len(decoded_results)} 성공")
    if decode_times:
        print(f"  평균 소요 시간: {np.mean(decode_times):.1f}초/건")

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  ✅ 저장: {output_file}")

    return output_file


# =============================================================================
# Step 2: Round-trip 검증 (생성 텍스트 → 재임베딩 → 원래 projected와 비교)
# =============================================================================
def validate_round_trip(output_file: str):
    print("\n" + "=" * 70)
    print(" Step 2: Round-trip 검증")
    print("=" * 70)
    print("  (생성된 텍스트를 다시 ada-002로 임베딩 → projected embedding과 비교)")

    df = pd.read_csv(output_file)

    valid_mask = df['decoded_text'].notna() & ~df['decoded_text'].isin(["ERROR", "ERROR_OOM", ""])
    df_valid = df[valid_mask].copy()

    if df_valid.empty:
        print("  ⚠️ 유효한 복원 텍스트 없음. 스킵.")
        return

    print(f"  검증 대상: {len(df_valid)}건")

    client = OpenAI(api_key=OPENAI_API_KEY)
    round_trip_sims = []

    for idx, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="Round-trip check"):
        decoded_text = row['decoded_text']
        projected_emb = parse_embedding(row['projected_text_embedding'])

        # 생성된 텍스트를 다시 임베딩
        re_embedded = get_embedding(client, decoded_text)
        sim = cosine_sim(projected_emb, re_embedded)
        round_trip_sims.append(sim)

    df.loc[valid_mask, 'round_trip_cosine_sim'] = round_trip_sims

    # 통계 출력
    sims_arr = np.array([s for s in round_trip_sims if not np.isnan(s)])
    if len(sims_arr) > 0:
        print(f"\n  --- Round-trip Cosine Similarity ---")
        print(f"  Mean:   {np.mean(sims_arr):.4f}")
        print(f"  Median: {np.median(sims_arr):.4f}")
        print(f"  Min:    {np.min(sims_arr):.4f}")
        print(f"  Max:    {np.max(sims_arr):.4f}")

        # 품질 등급
        high = np.sum(sims_arr >= 0.90)
        mid = np.sum((sims_arr >= 0.80) & (sims_arr < 0.90))
        low = np.sum(sims_arr < 0.80)
        print(f"\n  품질 분포: ≥0.90: {high}건 | 0.80~0.90: {mid}건 | <0.80: {low}건")

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  ✅ 업데이트 저장: {output_file}")


# =============================================================================
# Step 3: Nearest Neighbor 비교 (원본 특허와 비교)
# =============================================================================
def compare_with_reference(output_file: str, reference_file: str):
    print("\n" + "=" * 70)
    print(" Step 3: Nearest Neighbor 비교 (원본 특허 DB)")
    print("=" * 70)

    if not os.path.exists(reference_file):
        print(f"  ⚠️ 참조 파일 없음: {reference_file}. 스킵.")
        return

    df_result = pd.read_csv(output_file)
    df_ref = pd.read_csv(reference_file, dtype={'text_embedding': str})

    # 참조 데이터에서 text_embedding이 있는 것만
    ref_mask = (
        df_ref['text_embedding'].notna() &
        (df_ref['text_embedding'].str.strip() != '') &
        (df_ref['text_embedding'].str.strip() != 'None')
    )
    df_ref_valid = df_ref[ref_mask].copy()

    if df_ref_valid.empty:
        print("  ⚠️ 참조 데이터에 유효한 text_embedding 없음. 스킵.")
        return

    print(f"  참조 특허 수: {len(df_ref_valid)}개")

    # 참조 임베딩 행렬 구성
    ref_embs = np.array([
        parse_embedding(x) for x in df_ref_valid['text_embedding']
    ], dtype=np.float32)
    ref_norms = ref_embs / (np.linalg.norm(ref_embs, axis=1, keepdims=True) + 1e-8)

    # projected embedding과 nearest neighbor 매칭
    nn_records = []

    valid_mask = df_result['decoded_text'].notna() & ~df_result['decoded_text'].isin(["ERROR", "ERROR_OOM", ""])

    for idx, row in df_result[valid_mask].iterrows():
        proj_emb = parse_embedding(row['projected_text_embedding'])
        proj_norm = proj_emb / (np.linalg.norm(proj_emb) + 1e-8)

        sims = ref_norms @ proj_norm
        top_idx = np.argmax(sims)
        top_sim = sims[top_idx]

        nn_row = df_ref_valid.iloc[top_idx]
        nn_records.append({
            'node_id': row['id'],
            'nn_patent_id': nn_row.get('id', 'N/A'),
            'nn_title': str(nn_row.get('title', 'N/A'))[:120],
            'nn_abstract': str(nn_row.get('abstract', 'N/A'))[:200],
            'nn_cosine_sim': float(top_sim),
            'decoded_text': row['decoded_text']
        })

    # 결과 출력
    print(f"\n  --- Prospective Node별 가장 유사한 기존 특허 ---")
    for rec in nn_records:
        print(f"\n  [{rec['node_id']}]")
        print(f"    생성 텍스트: {rec['decoded_text'][:150]}...")
        print(f"    최근접 특허: {rec['nn_patent_id']} (sim={rec['nn_cosine_sim']:.4f})")
        print(f"    특허 제목:   {rec['nn_title']}")
        print(f"    특허 초록:   {rec['nn_abstract']}...")

    # nearest neighbor 정보도 결과 파일에 추가
    nn_df = pd.DataFrame(nn_records)
    nn_output = OUTPUT_FILE.replace('.csv', _ROOT + '/data/_with_nn.csv')
    nn_df.to_csv(nn_output, index=False, encoding='utf-8-sig')
    print(f"\n  ✅ NN 비교 결과 저장: {nn_output}")


# =============================================================================
# Step 4: 최종 요약
# =============================================================================
def print_final_summary(output_file: str):
    print("\n" + "=" * 70)
    print(" 최종 결과 요약")
    print("=" * 70)

    if not os.path.exists(output_file):
        return

    df = pd.read_csv(output_file)

    print(f"\n  총 prospective node: {len(df)}개")

    # 생성 결과 출력
    for idx, row in df.iterrows():
        node_id = row.get('id', f'node_{idx}')
        decoded = row.get('decoded_text', 'N/A')
        rt_sim = row.get('round_trip_cosine_sim', np.nan)

        print(f"\n  ┌─ [{node_id}]")
        print(f"  │  생성 텍스트: {str(decoded)[:200]}")
        if not np.isnan(rt_sim) if isinstance(rt_sim, float) else True:
            print(f"  │  Round-trip sim: {rt_sim:.4f}" if isinstance(rt_sim, float) else "")
        print(f"  └─")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("\n" + "★" * 35)
    print(" Embedding Inversion Pipeline")
    print("★" * 35)
    print(f"  Input:  {INPUT_FILE}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Beam Width: {SEQUENCE_BEAM_WIDTH}")
    print(f"  Steps: {NUM_STEPS}")

    # Step 1: 복원
    generated_file = invert_embeddings(INPUT_FILE, OUTPUT_FILE)

    if generated_file:
        # Step 2: Round-trip 검증
        validate_round_trip(generated_file)

        # Step 3: Nearest Neighbor 비교
        compare_with_reference(generated_file, REFERENCE_FILE)

        # Step 4: 요약
        print_final_summary(generated_file)

    print("\n🏁 모든 작업 완료.")