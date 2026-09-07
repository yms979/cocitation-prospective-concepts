"""논문용 before/after 네트워크 그림.

설계
 - before/after 는 동일 좌표(after 그래프에서 계산)를 사용해야 비교가 성립한다.
 - 예측 링크는 붉은색. inset 은 지정한 대상 링크(predicted_new_links.csv 11번째 줄)를 확대한다.
 - inset 은 양 패널에 동일 영역으로 넣어, 추가된 붉은 링크가 대비되어 보이게 한다.
"""
# Source file in the working repository: Networks/figures/make_figures.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np, pandas as pd

D = _ROOT + "/data/networks"
F = _ROOT + "/figures"
RED, PINK, GRAY, NODE = "#d62728", "#f0a3a8", "#c2c6cd", "#5b6472"
INK = "#1f2933"   # 주석(라벨·리더선) 전용 색 — 엣지/예측링크와 구분
BLUE = "#2f6f9f"  # 끝점 특허 노드 (prospective node 와 구분)
TARGET = ("US-9861271-B2", "EP-3679851-A1")   # 컨셉 No.11 = prospective node 22 (row 21)
TARGET_PROS = "prospective node 22 (row 21)"

def seg(P, i, j): return np.stack([P[i], P[j]], axis=1)

def crop(P, q=99.3, pad=1.04):
    """이상치를 제외한 데이터 경계상자를 정사각으로 감싼다 (여백 최소화)."""
    lo = np.percentile(P, 100-q, axis=0)
    hi = np.percentile(P, q, axis=0)
    c = (lo + hi) / 2
    h = (hi - lo).max() / 2 * pad
    return (c[0]-h, c[0]+h, c[1]-h, c[1]+h)


def draw(ax, P, gi, gj, ri, rj, ti, tj, style, zoom=False, pros_mask=None):
    """zoom=True 이면 하단 확대 패널. 확대 패널은 면적이 크므로
    본 패널보다 요소를 키워야 하지만, 과하면 연결 관계가 가려진다."""
    ew, ea, nw, rw = style
    if zoom:   # (엣지폭, 엣지알파, 노드크기, 예측폭, 대상폭, 대상노드)
        e_w, e_a, n_s, r_w, t_w, t_s = ew*1.5, min(ea*5, .30), nw*2.6, rw*.85, rw*1.7, nw*11
    else:
        e_w, e_a, n_s, r_w, t_w, t_s = ew, ea, nw, rw, rw*1.4, nw*5
    ax.add_collection(LineCollection(seg(P, gi, gj), colors=GRAY, linewidths=e_w,
                                     alpha=e_a, rasterized=True, zorder=1))
    ax.scatter(P[:,0], P[:,1], s=n_s, c=NODE, linewidths=0, alpha=.62,
               rasterized=True, zorder=2)
    if len(ri):
        ax.add_collection(LineCollection(seg(P, ri, rj), colors=PINK if zoom else RED,
                                         linewidths=r_w, alpha=.5 if zoom else .9, zorder=3))
        ep = np.concatenate([ri, rj])
        ax.scatter(P[ep,0], P[ep,1], s=n_s*(1.6 if zoom else 3.2),
                   c=PINK if zoom else RED, linewidths=0,
                   alpha=.55 if zoom else 1, zorder=4)
    if len(ti) and len(ri):
        ax.add_collection(LineCollection(seg(P, ti, tj), colors=RED,
                                         linewidths=t_w, alpha=1, zorder=6))
        ep = np.unique(np.concatenate([ti, tj]))
        lw = .6 if zoom else .35
        if pros_mask is None:
            ax.scatter(P[ep,0], P[ep,1], s=t_s, c=RED,
                       edgecolors="w", linewidths=lw, zorder=7)
        else:
            isp = pros_mask[ep]
            # 끝점 특허는 파란 원, prospective node 는 후광 + 마름모로 뚜렷하게
            ax.scatter(P[ep[~isp],0], P[ep[~isp],1], s=t_s*1.7, c=BLUE,
                       edgecolors="w", linewidths=lw, zorder=7)
            ax.scatter(P[ep[isp],0], P[ep[isp],1], s=t_s*5.0, c=RED,
                       alpha=.16, linewidths=0, zorder=7)
            ax.scatter(P[ep[isp],0], P[ep[isp],1], s=t_s*2.6, c=RED, marker="D",
                       edgecolors="w", linewidths=lw*1.8, zorder=9)


def build(kind, layout, suffix="", node_scale=1.0):
    z = np.load(f"{F}/{layout}", allow_pickle=True)
    P, nodes = z["pos"], z["nodes"]
    idx = {v: i for i, v in enumerate(nodes)}

    if kind == "cocitation":
        df = pd.read_csv(f"{D}/cocitation_after_edges.csv", dtype={"source":str,"target":str})
        new_ = df.link_status == "Predicted"
        net = "co-citation"
        cap = [f"{(~new_).sum():,} existing {net} links",
               f"+ {new_.sum()} predicted links (red)",
               f"predicted link {TARGET[0]} – {TARGET[1]}"]
        style = (.16, .012, 2.0, .68)
        tnodes = [n for n in TARGET if n in idx]
    else:
        df = pd.read_csv(f"{D}/citation_after_edges.csv", dtype={"source":str,"target":str})
        nd = pd.read_csv(f"{D}/citation_after_nodes.csv", dtype={"id":str})
        pros = set(nd.loc[nd.type=="prospective","id"])
        new_ = df.source.isin(pros) | df.target.isin(pros)
        net = "citation"
        cap = [f"{(~new_).sum():,} existing {net} links",
               f"+ {len(pros)} prospective nodes (red)",
               f"prospective node inserted for {TARGET[0]} – {TARGET[1]}"]
        style = (.30, .45, 1.6, .66)
        tnodes = [n for n in (*TARGET, TARGET_PROS) if n in idx]
    style = (style[0], style[1], style[2]*node_scale, style[3])

    pmask = np.array([str(n).startswith("prospective") for n in nodes]) \
            if kind != "cocitation" else None

    gi = df.loc[~new_,"source"].map(idx).to_numpy(); gj = df.loc[~new_,"target"].map(idx).to_numpy()
    ri = df.loc[ new_,"source"].map(idx).to_numpy(); rj = df.loc[ new_,"target"].map(idx).to_numpy()
    tset = set(tnodes)
    tm = df.loc[new_].apply(lambda r: r.source in tset and r.target in tset, axis=1).to_numpy()
    ti, tj = ri[tm], rj[tm]
    NONE = np.array([], int)

    tp = np.array([P[idx[n]] for n in tnodes])
    c = tp.mean(0); h = max(np.abs(tp - c).max()*2.1, .05)
    win = (c[0]-h, c[0]+h, c[1]-h, c[1]+h)
    main = crop(P)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15.6, 5.9))
    # 네트워크별로 실제 일어나는 조작이 다르다:
    #   동시인용 = 링크 예측 / 인용 = prospective node 삽입
    if kind == "cocitation":
        titles = ["(a) Before link prediction",
                  "(b) After link prediction",
                  "(c) Enlarged view"]
    else:
        titles = ["(a) Before prospective node insertion",
                  "(b) After prospective node insertion",
                  "(c) Enlarged view"]

    # ── (a) 예측 전 / (b) 예측 후 ────────────────────────────────
    for k, ax in enumerate([ax0, ax1]):
        r1, r2 = (ri, rj) if k else (NONE, NONE)
        t1, t2 = (ti, tj) if k else (NONE, NONE)
        draw(ax, P, gi, gj, r1, r2, t1, t2, style, pros_mask=pmask)
        ax.set_xlim(main[0], main[1]); ax.set_ylim(main[2], main[3])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titles[k], fontsize=14.5, pad=9)
        ax.text(.5, -.032, cap[k], transform=ax.transAxes, ha="center", va="top",
                fontsize=11, color="#444")
        for s in ax.spines.values(): s.set_edgecolor("#d5d5d5")

    # ── (c) 확대 패널 ────────────────────────────────────────────
    ax2.set_facecolor("#fcfcfc")
    draw(ax2, P, gi, gj, ri, rj, ti, tj, style, zoom=True, pros_mask=pmask)
    ax2.set_xlim(win[0], win[1]); ax2.set_ylim(win[2], win[3])
    ax2.set_aspect("equal"); ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title(titles[2], fontsize=14.5, pad=9)
    ax2.text(.5, -.032, cap[2], transform=ax2.transAxes, ha="center", va="top",
             fontsize=11, color="#444")
    for s in ax2.spines.values(): s.set_edgecolor(RED); s.set_linewidth(1.6)

    # (b) 의 확대영역 → (c) 로 이어지는 확대 효과 (연결선 유지)
    ind = ax1.indicate_inset([win[0], win[2], win[1]-win[0], win[3]-win[2]],
                             inset_ax=ax2, edgecolor=RED, linewidth=1.4, alpha=.95)
    conns = getattr(ind, "connectors", None)
    if conns is None:
        conns = ind[1]
    for cl in conns:
        cl.set_visible(True); cl.set_linewidth(1.0)
        cl.set_color(RED); cl.set_alpha(.55); cl.set_linestyle((0, (4, 2.4)))

    # 확대 패널의 특허번호 라벨
    pts = {n: P[idx[n]] for n in TARGET if n in idx}
    order = sorted(pts, key=lambda n: -pts[n][1])
    slots = [((.03, .97), "left", "top", -.16), ((.97, .03), "right", "bottom", .16)]
    for n, (xy, ha, va, rad) in zip(order, slots):
        px, py = pts[n]
        ax2.scatter([px], [py], s=300, facecolors="none", edgecolors=INK,
                    linewidths=1.0, zorder=8)
        ax2.annotate(n, xy=(px, py), xycoords="data",
                     xytext=xy, textcoords="axes fraction",
                     ha=ha, va=va, fontsize=11.5, color=INK, zorder=9,
                     bbox=dict(fc="white", ec=INK, lw=.7, alpha=.94,
                               boxstyle="round,pad=.3"),
                     arrowprops=dict(arrowstyle="-", color=INK, lw=1.0,
                                     linestyle=(0, (2.6, 1.9)), alpha=.95,
                                     shrinkA=2, shrinkB=12,
                                     connectionstyle=f"arc3,rad={rad}"))

    # 동시인용에서 붉은 선은 예측 링크 55개, 인용에서는 prospective node 가
    # 양 끝점과 이어진 엣지 110개이므로 범례 문구를 네트워크별로 구분한다.
    red_label = "Predicted link (n=55)" if pmask is None else "Inserted edge (n=110)"
    handles = [Line2D([],[],color=GRAY,lw=2.0,label="Existing link"),
               Line2D([],[],color=RED,lw=2.2,label=red_label)]
    if pmask is not None:      # 인용 네트워크: 노드 기호 설명 추가
        handles += [Line2D([],[],marker="o",color="w",markerfacecolor=BLUE,
                           markersize=10,label="Endpoint patents"),
                    Line2D([],[],marker="D",color="w",markerfacecolor=RED,
                           markersize=11,label="Inserted prospective node")]
    handles += [Line2D([],[],color=RED,lw=3.0,
                       label=f"Highlighted: {TARGET[0]} – {TARGET[1]}")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=11.5, bbox_to_anchor=(.5,-.004))
    fig.tight_layout(rect=[0,.075,1,1])
    for ext, dpi in [("pdf",600), ("png",400)]:
        fig.savefig(f"{F}/fig_{kind}_3panel{suffix}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig_{kind}_3panel{suffix}  | 확대창 ±{h:.3f} | 노드 {style[2]:.2f} | 대상 엣지 {tm.sum()}개", flush=True)
