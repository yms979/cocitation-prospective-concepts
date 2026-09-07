# -*- coding: utf-8 -*-
"""예측 링크 양 끝노드 + 1-hop 국소 서브그래프 추출 & 레이아웃 (실제 데이터)."""
# Source file in the working repository: Networks/figures/sub_layout.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import pandas as pd, numpy as np, igraph as ig, collections
D=_ROOT + "/data/networks"
F=_ROOT + "/figures"
A_ID,B_ID = "US-9861271-B2","EP-3679851-A1"; PR="prospective node 22 (row 21)"

e =pd.read_csv(f"{D}/citation_after_edges.csv",dtype=str)
nd=pd.read_csv(f"{D}/citation_after_nodes.csv",dtype=str)
pros=set(nd.loc[nd.type=="prospective","id"])

adj=collections.defaultdict(set)
for s,t in zip(e.source,e.target): adj[s].add(t); adj[t].add(s)
keep = {A_ID,B_ID} | adj[A_ID] | adj[B_ID]                 # 1-hop
sub  = e[e.source.isin(keep)&e.target.isin(keep)].reset_index(drop=True)
nodes= sorted(keep); idx={v:i for i,v in enumerate(nodes)}

g=ig.Graph(n=len(nodes), edges=list(zip(sub.source.map(idx),sub.target.map(idx))),
           directed=False)
np.random.seed(3)
P=np.array(g.layout_kamada_kawai().coords,float)
P-=P.mean(0); P/=np.abs(P).max()
np.savez_compressed(f"{F}/layouts/sub1_citation.npz", pos=P, nodes=np.array(nodes),
                    src=sub.source.to_numpy(), tgt=sub.target.to_numpy())

print(f"  1-hop 서브그래프 : 노드 {len(nodes)}  엣지 {len(sub)}")
print(f"  prospective 노드 : {sorted(pros & keep)}")
print("\n  엣지 (source → target = cited → citing):")
for _,r in sub.iterrows(): print(f"    {r.source:32s} → {r.target}")
print("\n  좌표:")
for n in nodes: print(f"    {n:32s} {P[idx[n]].round(3)}")
