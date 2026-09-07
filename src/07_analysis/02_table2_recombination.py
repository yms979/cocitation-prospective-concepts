# -*- coding: utf-8 -*-
"""Table 2 (최종) — Cross-area 블록 / Same-area 블록 / Total 구조."""
# Source file in the working repository: analysis/table2_final.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import pandas as pd
R=pd.read_pickle(_ROOT + '/data/recomb_intermediate.pkl')
L,cnt,base_cnt = R["L"],R["cnt"],R["base_cnt"]
BASE_N=sum(base_cnt.values()); N=len(L)

SHORT={"A61B 1":"Endoscopes","A61B 5":"Diagnostic measurement","A61B 6":"Radiation imaging",
 "A61B 17":"Surgical instruments","A61B 19":"Surgical navigation (legacy)",
 "A61B 34":"Computer-aided surgery","A61B 90":"Surgical guidance accessories",
 "A61F 2":"Implants and prostheses","A61N 1":"Electrotherapy",
 "B25J 9":"Programme-controlled manipulators","G01C 11":"Photogrammetry",
 "G06F 3":"User-input interfaces","G06F 30":"CAD and simulation","G06N 3":"Neural networks",
 "G06V 10":"Image recognition","G06V 20":"Scene understanding","G06V 40":"Biometric recognition",
 "G09B 9":"Training simulators","G09B 23":"Medical teaching models",
 "G16Z 99":"Health informatics (other)","H04N 13":"Stereoscopic imaging",
 "H04N 23":"Cameras and camera control","H10H 20":"Semiconductor light-emitting devices"}

bshare=lambda p: base_cnt.get(p,0)/BASE_N*100
cross=[(p,n) for p,n in cnt.items() if p[0]!=p[1]]
same =[(p,n) for p,n in cnt.items() if p[0]==p[1]]
cross.sort(key=lambda kv:(-kv[1],kv[0])); same.sort(key=lambda kv:(-kv[1],kv[0]))
CX_BASE=sum(bshare(p) for p,_ in base_cnt.items() if p[0]!=p[1])
SM_BASE=100-CX_BASE

def ratio(a,b): return None if b==0 else round(a/b,1)
def R_(name,desc,n,be,note=None):
    sh=n/N*100
    return dict(Recombined_areas=name,Description=desc,Links_n=n,
                Share_pct=round(sh,1),Share_existing_pct=round(be,2),
                Ratio=("n/a" if be==0 else f"{sh/be:.1f}") if n else "")

rows=[]
LISTED=9                                    # cross-area 상위 9쌍 (n>=2)
top_cx=[(p,n) for p,n in cross if n>=2][:LISTED]
oth_cx=[(p,n) for p,n in cross if (p,n) not in top_cx]
rows.append(dict(Recombined_areas="Cross-area recombinations",Description="",Links_n=None,
                 Share_pct=None,Share_existing_pct=None,Ratio=""))
for p,n in top_cx:
    rows.append(R_(f"{p[0]} × {p[1]}",f"{SHORT[p[0]]} × {SHORT[p[1]]}",n,bshare(p)))
rows.append(R_(f"Other cross-area pairs ({len(oth_cx)} pairs)","",
               sum(n for _,n in oth_cx),sum(bshare(p) for p,_ in oth_cx)))
rows.append(R_("Subtotal — cross-area","",sum(n for _,n in cross),CX_BASE))
rows.append(dict(Recombined_areas="Same-area recombinations",Description="",Links_n=None,
                 Share_pct=None,Share_existing_pct=None,Ratio=""))
for p,n in same:
    rows.append(R_(f"{p[0]} × {p[1]}",f"{SHORT[p[0]]} × {SHORT[p[1]]}",n,bshare(p)))
rows.append(R_("Subtotal — same-area","",sum(n for _,n in same),SM_BASE))
rows.append(dict(Recombined_areas="Total",Description="",Links_n=N,Share_pct=100.0,
                 Share_existing_pct=100.0,Ratio="1.0"))
T=pd.DataFrame(rows)
T.to_excel(_ROOT + '/tables/table2_recombination.xlsx',index=False)
T.to_csv(_ROOT + '/tables/table2_recombination.csv',index=False,encoding='utf-8-sig')
pd.set_option("display.width",240)
print(T.fillna("").to_string(index=False))
print(f"\n기존 링크 N={BASE_N:,} | cross {CX_BASE:.1f}% / same {SM_BASE:.1f}%")
