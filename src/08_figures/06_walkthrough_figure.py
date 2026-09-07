# -*- coding: utf-8 -*-
"""End-to-end walkthrough — 동시인용의 예측 링크에서 출발해 생성 텍스트까지.

(a) 동시인용 네트워크 전체 (예측 후)        ─확대─▶
(b) 예측 링크 A–B  [동시인용, 엣지로 존재]   ─삽입─▶
(c) prospective node P [인용 네트워크, 노드로 존재]  ─복원─▶
(d) 생성된 기술 개념 텍스트
(a)(b) 는 동시인용 좌표, (c) 는 인용 좌표 — 서로 다른 네트워크임을 명시한다.
"""
# Source file in the working repository: Networks/figures/walkthrough.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np, pandas as pd, textwrap

D=_ROOT + "/data/networks"
F=_ROOT + "/figures"
RED,PINK,GRAY,NODE,MUTE,INK,BLUE = \
    "#d62728","#f0a3a8","#c2c6cd","#5b6472","#8a8f98","#1f2933","#2f6f9f"
A_ID,B_ID="US-9861271-B2","EP-3679851-A1"; PR="prospective node 22 (row 21)"
CONCEPT=("One method relates to a surgical imaging system comprising a manipulating "
         "device, an endoscope, and a camera. The manipulating device generates an image "
         "of the anatomical position of a patient in a surgical procedure. The imaging "
         "system is augmented to provide a virtual image of the anatomical position of a "
         "patient in a surgical procedure. In addition, the image is analyzed for accuracy "
         "by a data acquisition ...")
seg=lambda P,i,j: np.stack([P[i],P[j]],1)


def load(kind, layout):
    z=np.load(f"{F}/{layout}",allow_pickle=True); P,nodes=z["pos"],list(z["nodes"])
    idx={v:i for i,v in enumerate(nodes)}
    if kind=="cocitation":
        df=pd.read_csv(f"{D}/cocitation_after_edges.csv",dtype={"source":str,"target":str})
        new=df.link_status=="Predicted"; style=(.16,.012,2.0*.6,.68); pm=None
        tset={A_ID,B_ID}
    else:
        df=pd.read_csv(f"{D}/citation_after_edges.csv",dtype={"source":str,"target":str})
        nd=pd.read_csv(f"{D}/citation_after_nodes.csv",dtype={"id":str})
        pros=set(nd.loc[nd.type=="prospective","id"])
        new=df.source.isin(pros)|df.target.isin(pros); style=(.30,.45,1.6,.66)
        pm=np.array([str(n).startswith("prospective") for n in nodes])
        tset={A_ID,B_ID,PR}
    g=(df.loc[~new,"source"].map(idx).to_numpy(), df.loc[~new,"target"].map(idx).to_numpy())
    r=(df.loc[ new,"source"].map(idx).to_numpy(), df.loc[ new,"target"].map(idx).to_numpy())
    tm=df.loc[new].apply(lambda x:x.source in tset and x.target in tset,axis=1).to_numpy()
    return dict(P=P,idx=idx,g=g,r=r,t=(r[0][tm],r[1][tm]),style=style,pm=pm)


def draw(ax,S,zoom):
    P=S["P"]; ew,ea,nw,rw=S["style"]; gi,gj=S["g"]; ri,rj=S["r"]; ti,tj=S["t"]
    if zoom: e_w,e_a,n_s,r_w,t_w,t_s = ew*1.5,min(ea*5,.30),nw*2.6,rw*.85,rw*1.7,nw*11
    else:    e_w,e_a,n_s,r_w,t_w,t_s = ew,ea,nw,rw,rw*1.4,nw*5
    ax.add_collection(LineCollection(seg(P,gi,gj),colors=GRAY,linewidths=e_w,
                                     alpha=e_a,rasterized=True,zorder=1))
    ax.scatter(P[:,0],P[:,1],s=n_s,c=NODE,linewidths=0,alpha=.62,rasterized=True,zorder=2)
    ax.add_collection(LineCollection(seg(P,ri,rj),colors=PINK if zoom else RED,
                                     linewidths=r_w,alpha=.5 if zoom else .9,zorder=3))
    ep=np.concatenate([ri,rj])
    ax.scatter(P[ep,0],P[ep,1],s=n_s*(1.6 if zoom else 3.2),c=PINK if zoom else RED,
               linewidths=0,alpha=.55 if zoom else 1,zorder=4)
    ax.add_collection(LineCollection(seg(P,ti,tj),colors=RED,linewidths=t_w,alpha=1,zorder=6))
    ep=np.unique(np.concatenate([ti,tj])); lw=.6 if zoom else .35
    isp = S["pm"][ep] if S["pm"] is not None else np.zeros(len(ep),bool)
    ax.scatter(P[ep[~isp],0],P[ep[~isp],1],s=t_s*1.7,c=BLUE,edgecolors="w",
               linewidths=lw,zorder=7)
    if isp.any():
        ax.scatter(P[ep[isp],0],P[ep[isp],1],s=t_s*5.0,c=RED,alpha=.16,linewidths=0,zorder=7)
        ax.scatter(P[ep[isp],0],P[ep[isp],1],s=t_s*2.6,c=RED,marker="D",
                   edgecolors="w",linewidths=lw*1.8,zorder=9)
    ax.set_xticks([]); ax.set_yticks([])


def window(S,names,m=2.1):
    pts=np.array([S["P"][S["idx"][n]] for n in names]); c=pts.mean(0)
    h=max(np.abs(pts-c).max()*m,.05)
    n_in=((np.abs(S["P"][:,0]-c[0])<h)&(np.abs(S["P"][:,1]-c[1])<h)).sum()
    return (c[0]-h,c[0]+h,c[1]-h,c[1]+h), n_in


def lead(ax,S,n,xy,ha,va,rad,col,txt,fs=8.2):
    p=S["P"][S["idx"][n]]
    if n!=PR:
        ax.scatter(*p,s=300,facecolors="none",edgecolors=INK,linewidths=1.0,zorder=8)
    ax.annotate(txt,xy=p,xycoords="data",xytext=xy,textcoords="axes fraction",
                ha=ha,va=va,fontsize=fs,color=col,zorder=10,fontweight="bold",
                bbox=dict(fc="white",ec=col,lw=.8,alpha=.95,boxstyle="round,pad=.3"),
                arrowprops=dict(arrowstyle="-",color=INK,lw=1.0,alpha=.95,
                                linestyle=(0,(2.6,1.9)),shrinkA=2,shrinkB=12,
                                connectionstyle=f"arc3,rad={rad}"))


CO = load("cocitation",_ROOT + "/figures/layouts/novlight_cocitation.npz")
CI = load("citation",_ROOT + "/figures/layouts/nov_citation.npz")
win_co,n_co = window(CO,[A_ID,B_ID],3.0)
win_ci,n_ci = window(CI,[A_ID,B_ID,PR],2.1)
lo=np.percentile(CO["P"],.7,axis=0); hi=np.percentile(CO["P"],99.3,axis=0)
mc=(lo+hi)/2; mh=(hi-lo).max()/2*1.04
main=(mc[0]-mh,mc[0]+mh,mc[1]-mh,mc[1]+mh)

# ── 4개 패널을 동일 크기 정사각으로 배치 (figsize 정사각 → PW==PH 이면 물리적 정사각) ──
FW=FH=12.0
PW=PH=.32          # 패널 한 변
GAP=.17            # 화살표가 놓이는 간격 = 모든 화살표의 길이(세 개 동일)
TGAP=.045          # 아래 행 제목이 들어갈 여유
COL=[.10,.10+PW+GAP]                 # 좌/우 열의 x0
ROW_T=.60; ROW_B=ROW_T-PH-GAP-TGAP   # 위/아래 행의 y0

fig=plt.figure(figsize=(FW,FH))
ax1=fig.add_axes([COL[0],ROW_T,PW,PH])   # 좌상
ax2=fig.add_axes([COL[1],ROW_T,PW,PH])   # 우상
ax3=fig.add_axes([COL[1],ROW_B,PW,PH])   # 우하
ax4=fig.add_axes([COL[0],ROW_B,PW,PH])   # 좌하

# ── (a) 동시인용 전체 ─────────────────────────────────────────
draw(ax1,CO,False); ax1.set_xlim(main[0],main[1]); ax1.set_ylim(main[2],main[3])
for s in ax1.spines.values(): s.set_edgecolor("#d5d5d5")

# ── (b) 동시인용 확대: 예측 링크 ──────────────────────────────
ax2.set_facecolor("#fcfcfc"); draw(ax2,CO,True)
ax2.set_xlim(win_co[0],win_co[1]); ax2.set_ylim(win_co[2],win_co[3])
for s in ax2.spines.values(): s.set_edgecolor("#d5d5d5")
lead(ax2,CO,A_ID,(.03,.97),"left","top",-.14,BLUE,"US-9861271-B2",10.5)
lead(ax2,CO,B_ID,(.97,.03),"right","bottom",-.14,BLUE,"EP-3679851-A1",10.5)

# ── (c) 인용 확대: prospective node ───────────────────────────
ax3.set_facecolor("#fcfcfc"); draw(ax3,CI,True)
ax3.set_xlim(win_ci[0],win_ci[1]); ax3.set_ylim(win_ci[2],win_ci[3])
for s in ax3.spines.values(): s.set_edgecolor("#d5d5d5")
lead(ax3,CI,PR,(.03,.97),"left","top",.14,INK,"inserted prospective node",10.5)
lead(ax3,CI,B_ID,(.03,.03),"left","bottom",-.10,BLUE,"EP-3679851-A1",10.5)
lead(ax3,CI,A_ID,(.97,.03),"right","bottom",.16,BLUE,"US-9861271-B2",10.5)

# ── (d) 생성된 개념 ──────────────────────────────────────────
ax4.set_xticks([]); ax4.set_yticks([]); ax4.set_facecolor("#fcfcfc")
for s in ax4.spines.values(): s.set_edgecolor("#d5d5d5")
ax4.text(.07,.885,textwrap.fill(CONCEPT,38),transform=ax4.transAxes,
         ha="left",va="top",fontsize=11,color=INK,linespacing=1.72)

# ── 확대 관계 + 동일 길이 화살표 ─────────────────────────────
fig.canvas.draw()
ind=ax1.indicate_inset([win_co[0],win_co[2],win_co[1]-win_co[0],win_co[3]-win_co[2]],
                       inset_ax=ax2,edgecolor=RED,linewidth=1.4,alpha=.95)
conns=getattr(ind,"connectors",None) or ind[1]
for cl in conns:
    cl.set_visible(True); cl.set_linewidth(1.0); cl.set_color(RED)
    cl.set_alpha(.55); cl.set_linestyle((0,(4,2.4)))

M=.008          # 양 끝 여백 — 세 화살표에 동일 적용
def AR(p0,p1):
    fig.add_artist(FancyArrowPatch(p0,p1,transform=fig.transFigure,
                   arrowstyle="-|>",mutation_scale=20,color=RED,lw=2.2))
ycen_t, ycen_b = ROW_T+PH/2, ROW_B+PH/2
xcen_r = COL[1]+PW/2
AR((COL[0]+PW+M,ycen_t),(COL[1]-M,ycen_t))                       # 좌상 → 우상
AR((xcen_r,ROW_T-M),(xcen_r,ROW_T-GAP+M))                        # 우상 → 우하
AR((COL[1]-M,ycen_b),(COL[0]+PW+M,ycen_b))                       # 우하 → 좌하
ARROW_FS=9.5   # 세 화살표 라벨의 글씨 크기(동일)
fig.text((COL[0]+PW+COL[1])/2,ycen_t+.012,"Enlargement",ha="center",va="bottom",
         fontsize=ARROW_FS,color=INK,fontweight="bold")
fig.text(xcen_r-.013,ROW_T-GAP/2,"Prospective node insertion\nin citation network",
         ha="right",va="center",fontsize=ARROW_FS,color=INK,fontweight="bold",linespacing=1.4)
fig.text((COL[0]+PW+COL[1])/2,ycen_b+.016,
         "Node embedding +\ncross-modal alignment\n+ embedding inversion",
         ha="center",va="bottom",fontsize=ARROW_FS,color=INK,fontweight="bold",linespacing=1.4)

for ax,ttl in ((ax1,"Predicted co-citation network"),
               (ax2,"Example of predicted link  (co-citation network)"),
               (ax3,"Inserted prospective node  (citation network)"),
               (ax4,"Prospective technology concept")):
    bb=ax.get_position()
    fig.text((bb.x0+bb.x1)/2,bb.y1+.010,ttl,ha="center",va="bottom",
             fontsize=12.5,color=INK,fontweight="bold")

fig.legend(handles=[Line2D([],[],color=GRAY,lw=2.0,label="Existing link"),
                    Line2D([],[],marker="o",color="w",markerfacecolor=BLUE,markersize=11,
                           label="Endpoint patents"),
                    Line2D([],[],marker="D",color="w",markerfacecolor=RED,markersize=12,
                           label="Inserted prospective node"),
                    Line2D([],[],color=RED,lw=2.6,label="Target predicted link"),
                    Line2D([],[],color=PINK,lw=2.0,label="Other predicted links")],
           loc="upper center",ncol=5,frameon=False,fontsize=11.5,
           bbox_to_anchor=(.5,ROW_B-.022))
for ext,dpi in [("pdf",600),("png",400)]:
    fig.savefig(f"{F}/fig_walkthrough.{ext}",dpi=dpi,bbox_inches="tight")
print(f"  생성 완료 | (b) 동시인용 {n_co}개 | (c) 인용 {n_ci}개")
