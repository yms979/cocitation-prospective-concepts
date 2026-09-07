# Source file in the working repository: code/1. co-citation.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import itertools
import os
from tqdm import tqdm

# --- 1. 데이터 ID 처리 함수 ---
def process_and_extract_ids(ref_string: str) -> list:
    if not isinstance(ref_string, str) or not ref_string.strip():
        return []
    raw_ids = [ref.strip() for ref in ref_string.split(';') if ref.strip()]
    return list(set(raw_ids))

# --- 2. 네트워크 구축 함수 (Association Strength) ---
def build_association_strength_network(papers_df: pd.DataFrame) -> nx.Graph:
    print(" >> 데이터 전처리 및 통계 계산 중...")
    papers_df['cited_patent_ids'] = papers_df['cited_patent_ids'].apply(process_and_extract_ids)
    
    total_docs = len(papers_df)
    used_docs_df = papers_df[papers_df['cited_patent_ids'].map(len) > 0]
    used_docs_count = len(used_docs_df)
    
    print(f"\n[문서 통계]")
    print(f" - 전체 문서 수 (Total Documents): {total_docs:,}")
    print(f" - 네트워크 구축에 사용된 문서 수 (Documents with Citations): {used_docs_count:,}")
    print(f" - 제외된 문서 수 (No Citations): {total_docs - used_docs_count:,}")

    # 개별 인용 빈도수(S_i) 계산
    print(" >> 개별 인용 빈도(Citation Frequency) 계산 중...")
    citation_frequency = Counter()
    for refs in tqdm(used_docs_df['cited_patent_ids'], desc="Counting Frequencies"):
        citation_frequency.update(refs)
        
    # 공동 인용 횟수(C_ij) 카운팅
    co_citation_counts = defaultdict(int)
    source_to_references = used_docs_df['cited_patent_ids'].tolist()
    
    print(" >> 공동 인용(Co-citation) 카운팅 중...")
    for references in tqdm(source_to_references, desc="Counting Pairs"):
        if len(references) >= 2:
            for ref1, ref2 in itertools.combinations(sorted(references), 2):
                edge_key = tuple(sorted((ref1, ref2)))
                co_citation_counts[edge_key] += 1
                
    # Association Strength 계산
    edge_list = []
    print(" >> Association Strength 계산 중...")
    for (ref1, ref2), co_count in tqdm(co_citation_counts.items(), desc="Calculating Weights"):
        freq1 = citation_frequency[ref1]
        freq2 = citation_frequency[ref2]
        
        if freq1 > 0 and freq2 > 0:
            weight = co_count / (freq1 * freq2)
            edge_list.append((ref1, ref2, weight))
    
    G = nx.Graph()
    G.add_weighted_edges_from(edge_list)
    print(f" -> 초기 그래프 생성 완료: 노드 {G.number_of_nodes():,}, 엣지 {G.number_of_edges():,}")
    return G

# --- 3. 임계값 시뮬레이션 함수 (LCC 기반) ---
def analyze_thresholds_lcc(graph: nx.Graph):
    print(f"\n{'='*75}")
    print(f" [LCC 기준] Association Strength 임계값 시뮬레이션")
    print(f"{'='*75}")
    print(f"{'Threshold (AS)':<15} | {'LCC Nodes':<10} | {'LCC Edges':<10} | {'LCC Density':<10}")
    print(f"{'-'*75}")
    
    edges_data = [(u, v, d['weight']) for u, v, d in graph.edges(data=True)]
    
    # 0.0부터 1.0까지 0.1 단위 + (중요) 낮은 구간 상세 확인용 값 추가
    # Association Strength는 값이 작으므로 0.01단위도 확인하는 것이 좋습니다.
    # 여기서는 요청하신 대로 0.1 단위를 기본으로 하되, 앞부분에 미세 구간을 추가했습니다.
    thresholds = [0.0, 0.001, 0.005, 0.01, 0.05] + [round(i * 0.1, 1) for i in range(1, 11)]
    
    # 중복 제거 및 정렬
    thresholds = sorted(list(set(thresholds)))

    for th in thresholds:
        # 1. 임계값 필터링
        filtered_edges = [(u, v, w) for u, v, w in edges_data if w >= th]
        
        if not filtered_edges:
            print(f"{th:<15.4f} | {'0':<10} | {'0':<10} | {'0.0000':<10}")
            continue
            
        # 2. 임시 그래프 생성
        temp_G = nx.Graph()
        temp_G.add_weighted_edges_from(filtered_edges)
        
        # 3. LCC(가장 큰 연결 요소) 추출
        if temp_G.number_of_nodes() > 0:
            largest_cc_nodes = max(nx.connected_components(temp_G), key=len)
            lcc_G = temp_G.subgraph(largest_cc_nodes).copy()
            
            num_nodes = lcc_G.number_of_nodes()
            num_edges = lcc_G.number_of_edges()
            density = nx.density(lcc_G)
        else:
            num_nodes = 0
            num_edges = 0
            density = 0.0
        
        print(f"{th:<15.4f} | {num_nodes:<10,} | {num_edges:<10,} | {density:.6f}")

    print(f"{'='*75}")

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    file_path = _ROOT + "/data/raw/collected_data_abstract_with_citation.csv"
    
    if os.path.exists(file_path):
        print(f"Loading {file_path}...")
        papers_df = pd.read_csv(file_path, dtype={'cited_patent_ids': str, 'original_id': str}, low_memory=False)
    else:
        print(f"Error: 파일을 찾을 수 없습니다: {file_path}")
        exit()
    
    # 1. 네트워크 구축
    print("\n--- Association Strength 기반 네트워크 구축 ---")
    full_graph = build_association_strength_network(papers_df)
    
    # 2. 임계값 시뮬레이션 (LCC 기준)
    # LCC만을 남겼을 때 노드가 몇 개나 남는지 보여줍니다.
    analyze_thresholds_lcc(full_graph)

    # 3. 저장
    try:
        user_input = input("\n저장할 임계값(Threshold)을 입력하세요 (예: 0.05): ")
        target_th = float(user_input)
    except ValueError:
        target_th = 0.0
        print("잘못된 입력입니다. 0.0으로 진행합니다.")
        
    print(f"\n>> 임계값 {target_th} 적용 및 LCC 추출 중...")
    
    # 최종 필터링 및 저장
    final_edges = [(u, v, d['weight']) for u, v, d in full_graph.edges(data=True) if d['weight'] >= target_th]
    final_G = nx.Graph()
    final_G.add_weighted_edges_from(final_edges)
    
    if final_G.number_of_nodes() > 0:
        largest_cc_nodes = max(nx.connected_components(final_G), key=len)
        lcc_final_G = final_G.subgraph(largest_cc_nodes).copy()
        
        output_file = f"{_ROOT}/data/networks/patent_co_citation_network_filtered.gexf"
        nx.write_gexf(lcc_final_G, output_file)
        print(f"\n✅ 저장 완료: {output_file}")
        print(f"   (최종 저장된 LCC 정보: 노드 {lcc_final_G.number_of_nodes()}, 엣지 {lcc_final_G.number_of_edges()})")
    else:
        print("\n⚠️ 해당 임계값에서는 남은 노드가 없어 저장하지 않았습니다.")