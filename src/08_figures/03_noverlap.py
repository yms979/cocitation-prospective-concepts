"""레이아웃 후처리: 노드 겹침 제거 (Gephi 의 Noverlap 에 해당).

각 노드를 반지름 r 의 원으로 보고, 겹치는 쌍을 반복적으로 밀어낸다.
전체 클러스터 구조는 유지하면서 뭉친 노드만 분리되어
"어떤 노드가 어떤 노드와 연결되는지"가 보이게 된다.
"""
# Source file in the working repository: Networks/figures/noverlap.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import numpy as np
from scipy.spatial import cKDTree

def remove_overlap(P, r, iters=300, strength=.55, tol=1e-5):
    P = P.copy().astype(float)
    n = len(P)
    for it in range(iters):
        pairs = cKDTree(P).query_pairs(2*r, output_type="ndarray")
        if len(pairs) == 0:
            print(f"    {it}회에서 겹침 해소"); break
        i, j = pairs[:, 0], pairs[:, 1]
        d = P[i] - P[j]
        dist = np.linalg.norm(d, axis=1)
        # 완전히 같은 좌표면 임의 방향으로 분리
        z = dist < 1e-12
        if z.any():
            th = np.random.default_rng(0).uniform(0, 2*np.pi, z.sum())
            d[z] = np.c_[np.cos(th), np.sin(th)] * 1e-6
            dist[z] = 1e-6
        push = (d / dist[:, None]) * ((2*r - dist) * strength / 2)[:, None]
        disp = np.zeros_like(P)
        for c in range(2):
            disp[:, c] += np.bincount(i, push[:, c], minlength=n)
            disp[:, c] -= np.bincount(j, push[:, c], minlength=n)
        P += disp
        if np.abs(disp).max() < tol:
            print(f"    {it}회에서 수렴"); break
    else:
        print(f"    {iters}회 종료 (잔여 겹침 {len(pairs):,}쌍)")
    return P

if __name__ == "__main__":
    for src, dst, r in [(_ROOT + "/figures/layouts/ig_citation.npz", _ROOT + "/figures/layouts/nov_citation.npz", .0042),
                        (_ROOT + "/figures/layouts/igfr_cocitation.npz", _ROOT + "/figures/layouts/nov_cocitation.npz", .0042)]:
        z = np.load(src, allow_pickle=True); P0 = z["pos"]
        t0 = cKDTree(P0).query_pairs(2*r, output_type="ndarray")
        print(f"  {src}: 초기 겹침 {len(t0):,}쌍 → 제거중")
        P = remove_overlap(P0, r)
        sc = np.percentile(np.abs(P - np.median(P, 0)), 99)
        np.savez_compressed(dst, pos=(P - np.median(P, 0)) / sc, nodes=z["nodes"])
        mv = np.linalg.norm(P - P0, axis=1)
        print(f"  → {dst} | 이동거리 중앙값 {np.median(mv):.4f} 최대 {mv.max():.4f}\n")
