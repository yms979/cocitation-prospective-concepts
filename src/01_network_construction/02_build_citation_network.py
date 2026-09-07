# Source file in the working repository: code/5. Citation_net.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import pandas as pd
import networkx as nx
import os
from tqdm import tqdm

# --- 1. 데이터 형식을 처리하기 위한 헬퍼 함수 ---
def process_and_extract_ids(ref_string: str) -> list:
    """
    'US-319296-A; US-668879-A; ...' 형태의 문자열을 파싱하여 리스트로 반환합니다.
    """
    if not isinstance(ref_string, str) or not ref_string.strip():
        return []
    id_list = [ref.strip() for ref in ref_string.split(';') if ref.strip()]
    return id_list

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    # 2. 파일 로드
    file_path = _ROOT + "/data/raw/collected_data_abstract_with_citation.csv"
    print(f"--- 데이터 파일 로드 시작: {file_path} ---")
    
    if os.path.exists(file_path):
        papers_df = pd.read_csv(file_path, dtype={'original_id': str, 'cited_patent_ids': str}, low_memory=False)
        print(f"총 {len(papers_df)} 개의 데이터를 로드했습니다.")
    else:
        print(f"오류: 파일 '{file_path}'을(를) 찾을 수 없습니다.")
        exit()

    # 데이터 전처리
    print("\n'cited_patent_ids' 컬럼 파싱 중...")
    tqdm.pandas()
    papers_df['cited_patent_ids'] = papers_df['cited_patent_ids'].progress_apply(process_and_extract_ids)
    print("파싱 완료.")

    # 3. 방향성 인용 네트워크 구축 (Cited -> Citing)
    print("\n--- 방향성 인용 네트워크(Citation Network) 구축 시작 ---")
    citation_graph = nx.DiGraph()
    
    for index, row in tqdm(papers_df.iterrows(), total=len(papers_df), desc="네트워크 구축 중"):
        citing_id = row['original_id']
        cited_ids = row['cited_patent_ids']
        
        if cited_ids:
            for cited_id in cited_ids:
                # Cited(과거) -> Citing(현재): 지식의 흐름 방향
                # 가중치 정규화 없이 단순 연결 (기본 weight=1로 간주되거나 unweighted)
                citation_graph.add_edge(cited_id, citing_id)
            
    print(f"전체 원본 노드 수: {citation_graph.number_of_nodes()}")
    print(f"전체 원본 엣지 수: {citation_graph.number_of_edges()}")

    # 4. 가장 큰 연결 요소(LWCC) 추출
    # 정규화 및 임계값 탐색 과정을 제거하고, 가장 큰 덩어리만 남깁니다.
    print("\n--- 가장 큰 연결 요소(Largest Weakly Connected Component) 추출 중 ---")
    
    if citation_graph.number_of_nodes() > 0:
        # 모든 약한 연결 요소 가져오기
        wcc_list = list(nx.weakly_connected_components(citation_graph))
        
        if wcc_list:
            # 가장 노드 수가 많은 컴포넌트(LWCC) 선택
            largest_component_nodes = max(wcc_list, key=len)
            
            # 원본 그래프에서 해당 노드들만 서브그래프로 추출
            final_network = citation_graph.subgraph(largest_component_nodes).copy()
            
            # 결과 출력
            print(f"선택된 LWCC 노드 수: {final_network.number_of_nodes()}")
            print(f"선택된 LWCC 엣지 수: {final_network.number_of_edges()}")
            
            # 원본 대비 비율 확인
            node_ratio = (final_network.number_of_nodes() / citation_graph.number_of_nodes()) * 100
            print(f"전체 데이터 대비 유지율: {node_ratio:.2f}%")

            # 5. 파일 저장 (기존 요청 파일명 유지)
            output_file = _ROOT + "/data/networks/directed_citation_network_filtered.gexf"
            print(f"\n--- 최종 저장 시작 ---")
            try:
                nx.write_gexf(final_network, output_file)
                print(f"✅ 저장 완료: {output_file}")
            except Exception as e:
                print(f"파일 저장 오류: {e}")
        else:
            print("연결된 컴포넌트가 존재하지 않습니다.")
    else:
        print("그래프에 노드가 없습니다.")