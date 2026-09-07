"""igraph 기반 레이아웃 재계산 (before/after 공통 좌표)."""
# Source file in the working repository: Networks/figures/layout_ig.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import igraph as ig, numpy as np, pandas as pd, os, time
D = _ROOT + "/data/networks"
F = _ROOT + "/figures"

def run(kind, ef, algo):
    dst = f"{F}/layouts/ig_{kind}.npz"
    if os.path.exists(dst):
        print(f"  {kind}: 기존 결과 사용", flush=True); return
    df = pd.read_csv(f"{D}/{ef}", dtype={"source": str, "target": str})
    nodes = pd.Index(pd.unique(pd.concat([df.source, df.target])))
    idx = {v: i for i, v in enumerate(nodes)}
    el = list(zip(df.source.map(idx), df.target.map(idx)))
    g = ig.Graph(n=len(nodes), edges=el, directed=False)
    print(f"  {kind}: 노드 {g.vcount():,} 엣지 {g.ecount():,} → {algo} 계산중...", flush=True)
    t0 = time.time()
    if algo == "drl":
        L = g.layout_drl(seed=None)
    else:
        L = g.layout_fruchterman_reingold(niter=800)
    P = np.array(L.coords, dtype=float)
    P -= np.median(P, 0)
    s = np.percentile(np.abs(P), 99)
    P /= s
    np.savez_compressed(dst, pos=P, nodes=nodes.to_numpy())
    print(f"{_ROOT}/data/  {kind}: 완료 ({time.time()-t0:.0f}s) → ig_{kind}.npz", flush=True)

run("citation",   _ROOT + "/data/networks/citation_after_edges.csv",   "fr")
run("cocitation", _ROOT + "/data/networks/cocitation_after_edges.csv", "drl")
