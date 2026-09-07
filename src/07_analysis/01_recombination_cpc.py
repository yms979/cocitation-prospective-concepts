# -*- coding: utf-8 -*-
"""분석 A — 예측 링크 55건의 기술영역 재조합 집계 (Table 2용).

대표 CPC 규칙
  각 특허의 CPC 목록에서 '첫 번째로 등장하는 분류코드'의 main group 을 취한다.
  단, 2000번대 코드(A61B2034/301 등)는 CPC 상 indexing code(보조 색인)이며
  독립적인 main group 이 아니므로 제외하고, Y 섹션(교차분류 태그)도 제외한다.
  예) A61B 34/20 → A61B 34   /   A61B2034/301 은 건너뛰고 다음 분류코드 사용
"""
# Source file in the working repository: analysis/recombination.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import pandas as pd, re
from collections import Counter
D=_ROOT
PAT=re.compile(r'^([A-HY])(\d{2})([A-Z])(\d+)/')

def main_group(cpc_str):
    """첫 번째 유효 분류코드의 main group 반환 (indexing/Y 코드 제외)"""
    for c in str(cpc_str).split(';'):
        c=c.strip()
        mt=PAT.match(c)
        if not mt: continue
        sec,cls,sub,grp=mt.groups()
        if sec=="Y" or int(grp)>=2000: continue      # 색인코드 제외
        return f"{sec}{cls}{sub} {int(grp)}"
    return None

meta=pd.read_csv(f"{D}/data/raw/Cited_patent_data.csv",dtype=str,low_memory=False)\
       .drop_duplicates('id').set_index('id')
mg={i:main_group(c) for i,c in meta['cpc_codes'].items()}

def pairs_of(df,c1,c2):
    out=[]
    for a,b in zip(df[c1],df[c2]):
        ga,gb=mg.get(a),mg.get(b)
        if not ga or not gb: out.append(None); continue
        out.append(tuple(sorted((ga,gb))))
    return out

# ── 예측 링크 55건 ────────────────────────────────────────────
L=pd.read_csv(f"{D}/data/predicted_new_links.csv",dtype=str)
L['pair']=pairs_of(L,'node1','node2')
L['cross']=[p is not None and p[0]!=p[1] for p in L.pair]
print(f"예측 링크 {len(L)}건 | 대표 CPC 결정 실패 {L.pair.isna().sum()}건")
print(f"  cross-area {L.cross.sum()}건 ({L.cross.mean()*100:.1f}%) | "
      f"same-area {(~L.cross).sum()}건 ({(~L.cross).mean()*100:.1f}%)")
cnt=Counter(p for p in L.pair if p)
print(f"\n=== 쌍별 빈도 (고유 {len(cnt)}쌍) ===")
for p,n in cnt.most_common():
    print(f"  {n:2d}  {p[0]:10s} × {p[1]:10s} {'(same)' if p[0]==p[1] else ''}")

# ── 기준선: 기존 동시인용 링크 전체 ───────────────────────────
E=pd.read_csv(f"{D}/data/networks/cocitation_after_edges.csv",
              dtype={"source":str,"target":str})
E=E[E.link_status=="Unpredicted"]
E['pair']=pairs_of(E,'source','target')
ok=E.pair.notna()
cross=pd.Series([p[0]!=p[1] for p in E.pair[ok]])
print(f"\n=== 기준선: 기존 동시인용 링크 ===")
print(f"  대상 {ok.sum():,}건 (CPC 결정 실패 {(~ok).sum():,}건 제외)")
print(f"  cross-area {cross.sum():,}건 ({cross.mean()*100:.1f}%) | "
      f"same-area {(~cross).sum():,}건 ({(~cross).mean()*100:.1f}%)")
pd.to_pickle({"L":L,"cnt":cnt,"base_cross":cross.mean(),
              "base_cnt":Counter(p for p in E.pair[ok])},_ROOT + '/data/recomb_intermediate.pkl')
