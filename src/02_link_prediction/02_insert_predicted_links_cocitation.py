# Source file in the working repository: code/Inserting_New_link.py
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
    base_gexf_file = _ROOT + "/data/networks/patent_co_citation_network_filtered.gexf"
    predicted_links_csv = _ROOT + "/data/predicted_new_links.csv"
    output_gexf_file = _ROOT + "/data/networks/co_citation_network_with_predicted_links.gexf"

    # --- 2. 원본 GEXF 네트워크 파일 로드 ---
    if not os.path.exists(base_gexf_file):
        print(f"오류: 원본 네트워크 파일 '{base_gexf_file}'을(를) 찾을 수 없습니다.")
        exit()
        
    print(f"--- 원본 네트워크 로드 중: {base_gexf_file} ---")
    try:
        G = nx.read_gexf(base_gexf_file)
        print("네트워크 로드 완료.")
        print(f"  초기 노드 수: {G.number_of_nodes()}")
        print(f"  초기 엣지 수: {G.number_of_edges()}")
    except Exception as e:
        print(f"네트워크 파일 로드 중 오류 발생: {e}")
        exit()

    # ✨ [수정 1] 기존 엣지에 'link_status' 속성을 'Unpredicted'로 초기화
    # (0/1 대신 문자열을 쓰면 Gephi에서 색상 구분하기가 훨씬 편합니다)
    nx.set_edge_attributes(G, "Unpredicted", "link_status")
    print("\n모든 기존 엣지에 'link_status: Unpredicted' 속성을 부여했습니다.")

    # --- 3. 예측된 링크 CSV 파일 로드 ---
    if not os.path.exists(predicted_links_csv):
        print(f"오류: 예측된 링크 파일 '{predicted_links_csv}'을(를) 찾을 수 없습니다.")
        exit()

    print(f"--- 예측된 링크 로드 중: {predicted_links_csv} ---")
    try:
        # ID가 숫자로만 되어 있을 경우를 대비해 문자열로 강제 변환
        predicted_links_df = pd.read_csv(predicted_links_csv, dtype={'node1': str, 'node2': str})
        print(f"총 {len(predicted_links_df)}개의 예측된 링크를 로드했습니다.")
    except Exception as e:
        print(f"CSV 파일 로드 중 오류 발생: {e}")
        exit()

    # --- 4. 예측된 링크를 네트워크에 엣지로 추가 ---
    print("\n--- 예측된 링크 처리 중 ---")
    
    added_new_edges = 0
    updated_existing_edges = 0
    skipped_count = 0

    for index, row in tqdm(predicted_links_df.iterrows(), total=predicted_links_df.shape[0], desc="Processing Links"):
        # ✨ [수정 2] 노드 이름 자르기(rsplit) 제거!
        # 원본 네트워크가 전체 ID를 쓰므로, 여기서도 그대로 써야 매칭이 됩니다.
        node1 = row['node1'].strip()
        node2 = row['node2'].strip()

        # 네트워크에 두 노드가 모두 존재하는지 확인
        if G.has_node(node1) and G.has_node(node2):
            if G.has_edge(node1, node2):
                # [CASE A] 이미 존재하는 엣지라면 -> 속성만 'Predicted'로 변경 (모델이 기존 관계를 맞춤)
                G[node1][node2]['link_status'] = "Predicted"
                updated_existing_edges += 1
            else:
                # [CASE B] 존재하지 않는 엣지라면 -> 새로 추가 (새로운 발견)
                G.add_edge(node1, node2, weight=0.5, link_status="Predicted")
                added_new_edges += 1
        else:
            # 매칭 실패 (네트워크에 없는 노드인 경우)
            skipped_count += 1
            # 디버깅용: 왜 매칭이 안 되는지 처음 3개만 출력
            if skipped_count <= 3:
                print(f"  [Skip] 매칭 실패: '{node1}' 또는 '{node2}' 가 그래프에 없음.")
    
    print("\n--- 처리 결과 ---")
    print(f"1. 새로 추가된 엣지 (New Discovery): {added_new_edges}개")
    print(f"2. 기존 엣지 속성 업데이트 (Re-discovery): {updated_existing_edges}개")
    print(f"3. 매칭 실패로 건너뜀: {skipped_count}개")

    # --- 5. 수정된 네트워크를 새로운 GEXF 파일로 저장 ---
    try:
        nx.write_gexf(G, output_gexf_file)
        print(f"\n작업 완료! '{output_gexf_file}' 파일로 저장되었습니다. ✨")
        print("\n--- 최종 네트워크 정보 ---")
        print(f"  최종 노드 수: {G.number_of_nodes()}")
        print(f"  최종 엣지 수: {G.number_of_edges()}")
    except Exception as e:
        print(f"\n수정된 네트워크 파일 저장 중 오류 발생: {e}")