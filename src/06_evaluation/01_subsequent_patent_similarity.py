# Source file in the working repository: code/12.Validation.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast

# --- 1. 데이터 로드 ---
print("🚀 데이터 로드 중...")

df_validation = pd.read_csv(_ROOT + '/data/raw/validation_abstract.csv')
df_prospective = pd.read_csv(_ROOT + '/data/final_decoded_results_with_nn.csv')

# --- 2. 임베딩 컬럼 파싱 ---
print("🚀 임베딩 파싱 중...")

def parse_embedding(val):
    """문자열로 저장된 임베딩을 리스트로 변환합니다."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except:
            return None
    return None

df_validation['embedding'] = df_validation['embedding'].apply(parse_embedding)
df_prospective['embedding'] = df_prospective['embedding'].apply(parse_embedding)

val_null = df_validation['embedding'].isnull().sum()
pros_null = df_prospective['embedding'].isnull().sum()
if val_null > 0 or pros_null > 0:
    print(f"⚠️ 임베딩 파싱 실패: validation={val_null}건, prospective={pros_null}건")
    df_validation = df_validation.dropna(subset=['embedding'])
    df_prospective = df_prospective.dropna(subset=['embedding'])

print(f"✅ Validation 문서: {len(df_validation)}건, Prospective 노드: {len(df_prospective)}건")

# --- 3. 전체 코사인 유사도 행렬 계산 ---
print("\n🚀 전체 코사인 유사도 행렬 계산 중 (55 x 120)...")

validation_embeddings = np.array(df_validation['embedding'].tolist())
prospective_embeddings = np.array(df_prospective['embedding'].tolist())

similarity_matrix = cosine_similarity(prospective_embeddings, validation_embeddings)

print(f"✅ 유사도 행렬 크기: {similarity_matrix.shape}")

# --- 4. 각 Prospective Node별 통계 ---
print("\n🚀 각 Prospective Node별 통계 계산 중...")

node_stats = []
for i in range(len(df_prospective)):
    scores = similarity_matrix[i]
    top5_indices = np.argsort(scores)[::-1][:5]
    top5_scores = scores[top5_indices]

    top5_titles = []
    for idx in top5_indices:
        if 'title' in df_validation.columns:
            top5_titles.append(str(df_validation.iloc[idx]['title']))
        else:
            top5_titles.append(f"Doc_{idx}")

    # z-score: Top-1이 해당 노드의 유사도 분포에서 몇 표준편차 위에 있는지
    node_std = np.std(scores)
    z_score_top1 = (np.max(scores) - np.mean(scores)) / node_std if node_std > 0 else 0

    stat = {
        'decoded_text': df_prospective.iloc[i]['decoded_text'],
        'mean_similarity': np.mean(scores),
        'std_similarity': node_std,
        'median_similarity': np.median(scores),
        'max_similarity': np.max(scores),
        'min_similarity': np.min(scores),
        'z_score_top1': z_score_top1,
        'top1_score': top5_scores[0],
        'top1_title': top5_titles[0],
        'top2_score': top5_scores[1],
        'top2_title': top5_titles[1],
        'top3_score': top5_scores[2],
        'top3_title': top5_titles[2],
        'top4_score': top5_scores[3],
        'top4_title': top5_titles[3],
        'top5_score': top5_scores[4],
        'top5_title': top5_titles[4],
        'above_0.9': int(np.sum(scores >= 0.9)),
        'above_0.8': int(np.sum(scores >= 0.8)),
        'above_0.7': int(np.sum(scores >= 0.7)),
    }
    node_stats.append(stat)

df_node_stats = pd.DataFrame(node_stats)

# --- 5. 전체 통계 요약 ---
print("\n" + "=" * 60)
print("📊 전체 통계 요약")
print("=" * 60)

all_scores = similarity_matrix.flatten()
print(f"  전체 유사도 쌍 수:     {len(all_scores)}")
print(f"  전체 평균 유사도:      {np.mean(all_scores):.4f}")
print(f"  전체 표준편차:         {np.std(all_scores):.4f}")
print(f"  전체 중앙값:           {np.median(all_scores):.4f}")
print(f"  전체 최대값:           {np.max(all_scores):.4f}")
print(f"  전체 최소값:           {np.min(all_scores):.4f}")
print(f"  0.9 이상 쌍 수:        {int(np.sum(all_scores >= 0.9))}")
print(f"  0.8 이상 쌍 수:        {int(np.sum(all_scores >= 0.8))}")
print(f"  0.7 이상 쌍 수:        {int(np.sum(all_scores >= 0.7))}")

print("\n📊 Prospective Node별 평균 유사도 통계")
print("-" * 40)
node_means = df_node_stats['mean_similarity']
print(f"  노드별 평균의 평균:   {node_means.mean():.4f}")
print(f"  노드별 평균의 표준편차: {node_means.std():.4f}")
print(f"  노드별 평균의 최대:   {node_means.max():.4f}")
print(f"  노드별 평균의 최소:   {node_means.min():.4f}")

print("\n📊 Prospective Node별 최대 유사도 통계")
print("-" * 40)
node_maxs = df_node_stats['max_similarity']
print(f"  노드별 최대의 평균:   {node_maxs.mean():.4f}")
print(f"  노드별 최대의 표준편차: {node_maxs.std():.4f}")
print(f"  노드별 최대의 최대:   {node_maxs.max():.4f}")
print(f"  노드별 최대의 최소:   {node_maxs.min():.4f}")

print("\n📊 Prospective Node별 Top-1 z-score 통계")
print("-" * 40)
node_zscores = df_node_stats['z_score_top1']
print(f"  z-score 평균:         {node_zscores.mean():.4f}")
print(f"  z-score 표준편차:     {node_zscores.std():.4f}")
print(f"  z-score 최대:         {node_zscores.max():.4f}")
print(f"  z-score 최소:         {node_zscores.min():.4f}")

# --- 6. 결과 저장 ---
print("\n🚀 결과 저장 중...")

# 6-1. 노드별 통계 저장
output_stats = _ROOT + '/data/prospective_nodes_full_similarity_stats.xlsx'
df_node_stats.to_excel(output_stats, index=False)
print(f"✅ 노드별 통계: '{output_stats}'")

# 6-2. 전체 유사도 행렬 저장
val_labels = df_validation['title'].tolist() if 'title' in df_validation.columns else [f"Val_{i}" for i in range(len(df_validation))]
df_matrix = pd.DataFrame(similarity_matrix, columns=val_labels)
if 'decoded_text' in df_prospective.columns:
    df_matrix.insert(0, 'decoded_text', df_prospective['decoded_text'].values)

output_matrix = _ROOT + '/data/prospective_validation_similarity_matrix.xlsx'
df_matrix.to_excel(output_matrix, index=False)
print(f"✅ 전체 유사도 행렬: '{output_matrix}'")

# 6-3. 전체 요약 통계 저장
summary_data = {
    'metric': [
        '전체 유사도 쌍 수', '전체 평균', '전체 표준편차', '전체 중앙값',
        '전체 최대값', '전체 최소값', '0.9 이상 쌍 수', '0.8 이상 쌍 수', '0.7 이상 쌍 수',
        '노드별 평균의 평균', '노드별 평균의 표준편차', '노드별 평균의 최대', '노드별 평균의 최소',
        '노드별 최대의 평균', '노드별 최대의 표준편차', '노드별 최대의 최대', '노드별 최대의 최소',
        'Top-1 z-score 평균', 'Top-1 z-score 표준편차', 'Top-1 z-score 최대', 'Top-1 z-score 최소',
    ],
    'value': [
        len(all_scores), np.mean(all_scores), np.std(all_scores), np.median(all_scores),
        np.max(all_scores), np.min(all_scores),
        int(np.sum(all_scores >= 0.9)), int(np.sum(all_scores >= 0.8)), int(np.sum(all_scores >= 0.7)),
        node_means.mean(), node_means.std(), node_means.max(), node_means.min(),
        node_maxs.mean(), node_maxs.std(), node_maxs.max(), node_maxs.min(),
        node_zscores.mean(), node_zscores.std(), node_zscores.max(), node_zscores.min(),
    ]
}
df_summary = pd.DataFrame(summary_data)

output_summary = _ROOT + '/data/prospective_similarity_summary.xlsx'
df_summary.to_excel(output_summary, index=False)
print(f"✅ 전체 요약 통계: '{output_summary}'")

print(f"\n🎉 모든 통계 분석이 완료되었습니다!")