# Source file in the working repository: Networks/figures/layout_fr_coc.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import igraph as ig, numpy as np, pandas as pd, time
D=_ROOT + "/data/networks"; F=_ROOT + "/figures"
df=pd.read_csv(f"{D}/cocitation_after_edges.csv",dtype={"source":str,"target":str})
nodes=pd.Index(pd.unique(pd.concat([df.source,df.target]))); idx={v:i for i,v in enumerate(nodes)}
g=ig.Graph(n=len(nodes),edges=list(zip(df.source.map(idx),df.target.map(idx))),directed=False)
print(f"노드 {g.vcount():,} 엣지 {g.ecount():,} → FR 계산중...",flush=True)
t0=time.time(); L=g.layout_fruchterman_reingold(niter=500)
P=np.array(L.coords,float); P-=np.median(P,0); P/=np.percentile(np.abs(P),99)
np.savez_compressed(f"{F}/layouts/igfr_cocitation.npz",pos=P,nodes=nodes.to_numpy())
r=np.linalg.norm(P-np.median(P,0),axis=1)
print(f"완료 ({time.time()-t0:.0f}s) | 중심거리 중앙값 {np.median(r):.3f} 최대 {r.max():.3f}",flush=True)
