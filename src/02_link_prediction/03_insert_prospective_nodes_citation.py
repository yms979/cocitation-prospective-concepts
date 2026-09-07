# Source file in the working repository: code/6-1.insert_prospective _node_in_citation_network.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import networkx as nx
import pandas as pd
import os
from tqdm import tqdm 

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    # --- 1. 파일 경로 설정 ---
    gexf_file = _ROOT + "/data/networks/directed_citation_network_filtered.gexf"
    csv_file = _ROOT + "/data/predicted_new_links.csv"
    output_gexf_file = _ROOT + "/data/networks/network_with_prospective_nodes.gexf"

    # --- 2. 네트워크 파일 로드 ---
    if not os.path.exists(gexf_file):
        print(f"오류: 네트워크 파일 '{gexf_file}'을(를) 찾을 수 없습니다.")
        print("참고: 이전 단계에서 저장된 파일명이 맞는지 확인해주세요.")
        exit()
        
    print(f"--- 네트워크 로드 중: {gexf_file} ---")
    try:
        G = nx.read_gexf(gexf_file)
        print(f"네트워크 로드 완료. (노드: {G.number_of_nodes()}, 엣지: {G.number_of_edges()})")
    except Exception as e:
        print(f"네트워크 파일 로드 중 오류 발생: {e}")
        exit()

    # ✨ [추가] 기존에 존재하는 모든 노드에게 초기 값 1 부여
    # 속성 이름: 'prospective_value' (기존=1, 신규=10)
    print("--- 기존 노드 속성 초기화 중 (value=1) ---")
    nx.set_node_attributes(G, 1, name='prospective_value')

    # --- 3. 예측된 링크 CSV 파일 로드 ---
    if not os.path.exists(csv_file):
        print(f"오류: 예측된 링크 파일 '{csv_file}'을(를) 찾을 수 없습니다.")
        exit()

    print(f"--- 예측된 링크 로드 중: {csv_file} ---")
    try:
        predicted_links_df = pd.read_csv(csv_file)
        # 노드 ID를 문자열로 통일
        predicted_links_df['node1'] = predicted_links_df['node1'].astype(str)
        predicted_links_df['node2'] = predicted_links_df['node2'].astype(str)
        print(f"총 {len(predicted_links_df)}개의 예측된 링크를 로드했습니다.")
    except Exception as e:
        print(f"CSV 파일 로드 중 오류 발생: {e}")
        exit()

    # --- 4. 새로운 'Prospective Node' 생성 및 연결 ---
    print("\n--- Prospective Node 생성 및 연결 시작 (방향: Existing -> New) ---")
    
    prospective_node_counter = 1
    nodes_not_found = set()

    for index, row in tqdm(predicted_links_df.iterrows(), total=predicted_links_df.shape[0], desc="Processing Links"):
        node1 = row['node1']
        node2 = row['node2']

        node1_exists = G.has_node(node1)
        node2_exists = G.has_node(node2)

        if node1_exists and node2_exists:
            new_node_name = f"prospective node {prospective_node_counter} (row {index})"
            
            # ✨ [수정] 새로운 노드 추가 시 'prospective_value'를 10으로 설정
            G.add_node(
                new_node_name, 
                type='prospective', 
                label='Prospective Invention', 
                source_row=index,
                prospective_value=10  # 여기에 값 10 할당
            )
            
            # 엣지 방향: 기존 노드 -> 새로운 노드
            G.add_edge(node1, new_node_name)
            G.add_edge(node2, new_node_name)
            
            prospective_node_counter += 1
        else:
            if not node1_exists:
                nodes_not_found.add(node1)
            if not node2_exists:
                nodes_not_found.add(node2)

    print(f"\n총 {prospective_node_counter - 1}개의 Prospective Node를 생성하고 연결했습니다.")

    # --- 5. 결과 보고 ---
    if nodes_not_found:
        print(f"\n경고: 예측된 링크 파일에 있었지만 네트워크에서 찾을 수 없는 노드가 총 {len(nodes_not_found)}개 발견되었습니다.")
    else:
        print("\n모든 예측 링크의 노드가 네트워크에 존재하여 성공적으로 처리되었습니다.")

    # --- 6. 수정된 네트워크 파일 저장 ---
    try:
        nx.write_gexf(G, output_gexf_file)
        print(f"\nProspective Node가 추가된 새로운 네트워크가 '{output_gexf_file}' 파일로 저장되었습니다.")
        print(f"최종 네트워크 정보: (노드: {G.number_of_nodes()}, 엣지: {G.number_of_edges()})")
        print("참고: 'prospective_value' 속성을 사용하여 노드 크기나 색상을 구분할 수 있습니다.")
    except Exception as e:
        print(f"\n수정된 네트워크 파일 저장 중 오류 발생: {e}")