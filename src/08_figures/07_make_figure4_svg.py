#!/usr/bin/env python3
"""
Build Figure 4 -- end-to-end walkthrough of one prospective node.

Left-to-right, four stages:
  (a) the two endpoint patents of the predicted link
  (b) their local citation subgraph as collected  -> two separate components
  (c) the same subgraph with the prospective node inserted -> one component
  (d) the concept text decoded from that node's projected embedding

Node positions in (b) and (c) are a Kamada-Kawai layout of the real induced
subgraph, computed once on the graph that includes the prospective nodes and
reused in both panels so the two panels are directly comparable.

Output: figure4_walkthrough.svg / .pdf / .png
Geometry: 1 user unit = 1 pt; viewBox width 510 pt = 180 mm (full journal width).
"""
# Source file in the working repository: make_figure4.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------

import math
import networkx as nx

# ----------------------------------------------------------------- real data
GEXF = _ROOT + '/data/networks/network_with_prospective_nodes.gexf'
A, B = 'US-10383765-B2', 'US-2022101745-A1'
P27 = 'prospective node 27 (row 26)'

G = nx.read_gexf(GEXF)
U = G.to_undirected()
H = U.subgraph(set([A, B]) | set(U[A]) | set(U[B])).copy()

SHORT = {A: 'A', B: 'B',
         'US-12268460-B2': 'C1', 'US-11957421-B2': 'C2', 'US-12290319-B2': 'C3',
         'prospective node 25 (row 24)': 'P25',
         'prospective node 26 (row 25)': 'P26',
         P27: 'P27'}
PROSPECTIVE = {n for n in H if n.startswith('prospective')}
OBSERVED = [n for n in H if n not in PROSPECTIVE]

# one layout, reused by both graph panels
pos = nx.kamada_kawai_layout(H)

Ho = H.subgraph(OBSERVED)
COMPONENTS = [sorted(c) for c in nx.connected_components(Ho)]

# Rotate so the A-side component sits left of the B-side component, putting the
# subgraph's own axis along the figure's left-to-right reading direction.
_cA = [c for c in COMPONENTS if A in c][0]
_cB = [c for c in COMPONENTS if B in c][0]


def _centroid(ns):
    return (sum(pos[n][0] for n in ns) / len(ns),
            sum(pos[n][1] for n in ns) / len(ns))


_ax, _ay = _centroid(_cA)
_bx, _by = _centroid(_cB)
_th = -math.atan2(_by - _ay, _bx - _ax)
_c, _s = math.cos(_th), math.sin(_th)
pos = {n: (x * _c - y * _s, x * _s + y * _c) for n, (x, y) in pos.items()}
assert len(COMPONENTS) == 2 and nx.number_connected_components(
    H.subgraph(OBSERVED + [P27])) == 1

# ------------------------------------------------------------------- palette
INK, RULE, MUTED, ACC, SOFT = '#1A1A1A', '#B8B8B8', '#666666', '#17557A', '#EAF1F6'
FILL, HULL = '#F5F5F3', '#EDEDEA'

# ------------------------------------------------------------------ geometry
W, Hh = 510, 196
C1X, C1W = 8, 96            # (a) endpoint patents
C2X, C2W = 116, 102         # (b) subgraph as collected
C3X, C3W = 230, 102         # (c) subgraph with prospective node
C4X, C4W = 344, 158         # (d) decoded concept
TOP, PANH = 28, 138         # content band top, panel height

o = []
add = o.append


def esc(t):
    return t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def txt(x, y, s, size=6, weight=None, fill=INK, anchor=None, ls=None, style=None):
    a = ['x="%g"' % x, 'y="%g"' % y, 'font-size="%g"' % size]
    if weight:
        a.append('font-weight="%s"' % weight)
    if fill != INK:
        a.append('fill="%s"' % fill)
    if anchor:
        a.append('text-anchor="%s"' % anchor)
    if ls:
        a.append('letter-spacing="%g"' % ls)
    if style:
        a.append('font-style="%s"' % style)
    add('  <text %s>%s</text>' % (' '.join(a), s))


def rect(x, y, w, h, fill='none', stroke=RULE, sw=0.6, dash=None):
    a = ['x="%g"' % x, 'y="%g"' % y, 'width="%g"' % w, 'height="%g"' % h,
         'fill="%s"' % fill, 'stroke="%s"' % stroke, 'stroke-width="%g"' % sw]
    if dash:
        a.append('stroke-dasharray="%s"' % dash)
    add('  <rect %s/>' % ' '.join(a))


def line(x1, y1, x2, y2, stroke=INK, sw=0.6, dash=None, marker=None):
    a = ['x1="%g"' % x1, 'y1="%g"' % y1, 'x2="%g"' % x2, 'y2="%g"' % y2,
         'stroke="%s"' % stroke, 'stroke-width="%g"' % sw]
    if dash:
        a.append('stroke-dasharray="%s"' % dash)
    if marker:
        a.append('marker-end="url(#%s)"' % marker)
    add('  <line %s/>' % ' '.join(a))


def circ(cx, cy, r, fill='#FFFFFF', stroke=INK, sw=0.6, dash=None):
    a = ['cx="%g"' % cx, 'cy="%g"' % cy, 'r="%g"' % r,
         'fill="%s"' % fill, 'stroke="%s"' % stroke, 'stroke-width="%g"' % sw]
    if dash:
        a.append('stroke-dasharray="%s"' % dash)
    add('  <circle %s/>' % ' '.join(a))


# ------------------------------------------------- node radii from real degree
def radius(n):
    if n in PROSPECTIVE:
        return 7.5 if n == P27 else 3.2      # emphasised, not degree-scaled
    return 2.2 + 2.2 * math.log10(G.degree(n) + 1)


# --------------------------------------------------- map layout into a panel
GTOP, GBOX_H = 36, 74                         # drawing box inside a panel


def keepout(n):
    """Radius that must stay clear: the node, its halo, and its label."""
    r = radius(n)
    if n == P27:
        r += 6.0                             # halo plus breathing room
    if n in PROSPECTIVE and n != P27:
        r += 4.5                             # label sits outside these
    return r + 1.6


def place(px, pw):
    """Kamada-Kawai positions fitted to the panel, then relaxed so that no two
    nodes (or their labels) overlap. Identical positions are reused by both
    graph panels, so the only difference between them is the inserted node."""
    x0, y0 = px + 3, GTOP
    x1, y1 = px + pw - 3, GTOP + GBOX_H
    xs = [q[0] for q in pos.values()]
    ys = [q[1] for q in pos.values()]
    pad = 12
    s = min((pw - 2 * pad) / (max(xs) - min(xs)), GBOX_H / (max(ys) - min(ys)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    mx, my = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    p = {n: [cx + (x - mx) * s, cy - (y - my) * s] for n, (x, y) in pos.items()}

    ko = {n: keepout(n) for n in p}
    keys = list(p)
    for _ in range(400):
        moved = False
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                dx, dy = p[b][0] - p[a][0], p[b][1] - p[a][1]
                d = math.hypot(dx, dy) or 0.01
                need = ko[a] + ko[b]
                if d < need:
                    push, ux, uy = (need - d) / 2, dx / d, dy / d
                    p[a][0] -= ux * push
                    p[a][1] -= uy * push
                    p[b][0] += ux * push
                    p[b][1] += uy * push
                    moved = True
        for n in p:                                   # keep inside the box
            p[n][0] = min(max(p[n][0], x0 + ko[n]), x1 - ko[n])
            p[n][1] = min(max(p[n][1], y0 + ko[n]), y1 - ko[n])
        if not moved:
            break
    return {n: tuple(v) for n, v in p.items()}


def draw_graph(px, pw, show_prospective):
    p = place(px, pw)

    if not show_prospective:                  # shade the two components
        for comp in COMPONENTS:
            xs = [p[n][0] for n in comp]
            ys = [p[n][1] for n in comp]
            r = max(radius(n) for n in comp)
            add('  <rect x="%g" y="%g" width="%g" height="%g" rx="%g" '
                'fill="%s" stroke="none"/>'
                % (min(xs) - r - 3.5, min(ys) - r - 3.5,
                   max(xs) - min(xs) + 2 * r + 7, max(ys) - min(ys) + 2 * r + 7,
                   r + 3.5, HULL))

    nodes = list(H.nodes()) if show_prospective else OBSERVED
    nodeset = set(nodes)
    for u, v in H.edges():
        if u not in nodes or v not in nodes:
            continue
        subject = P27 in (u, v)
        new = u in PROSPECTIVE or v in PROSPECTIVE
        line(p[u][0], p[u][1], p[v][0], p[v][1],
             stroke=ACC if new else INK,
             sw=1.2 if subject else (0.5 if new else 0.55))

    if show_prospective:                      # halo marks the inserted node
        circ(p[P27][0], p[P27][1], radius(P27) + 3.6, fill='none',
             stroke=ACC, sw=0.5, dash='2 2')

    for n in nodes:                           # all circles first ...
        r = radius(n)
        pro = n in PROSPECTIVE
        circ(p[n][0], p[n][1], r,
             fill=SOFT if pro else (INK if n in (A, B) else '#FFFFFF'),
             stroke=ACC if pro else INK,
             sw=1.5 if n == P27 else (0.7 if pro else 0.8))

    for n in nodes:                           # ... then every label on top
        r = radius(n)
        pro = n in PROSPECTIVE
        lab = SHORT[n]
        if n in (A, B):                       # single letter, reversed out
            txt(p[n][0], p[n][1] + 1.6, lab, 4.6, 'bold', '#FFFFFF', 'middle')
        elif r >= 6:                          # fits inside the node
            txt(p[n][0], p[n][1] + 1.7, lab, 5, 'bold',
                ACC if pro else INK, 'middle')
        else:                                 # outside, pushed away from
            nb = [q for q in H[n] if q in nodeset]   # its own neighbours
            dx = sum(p[n][0] - p[q][0] for q in nb) or 0.0
            dy = sum(p[n][1] - p[q][1] for q in nb) or -1.0
            d = math.hypot(dx, dy) or 1.0
            lx = p[n][0] + dx / d * (r + 4.6)
            ly = p[n][1] + dy / d * (r + 4.6) + 1.5
            w = len(lab) * 3.0 + 1.6
            add('  <rect x="%g" y="%g" width="%g" height="%g" fill="#FFFFFF" '
                'stroke="none"/>' % (lx - w / 2, ly - 4.4, w, 5.6))
            txt(lx, ly, lab, 4.6, 'bold', ACC if pro else INK, 'middle')
    return p


# =============================================================== emit the SVG
add('<?xml version="1.0" encoding="UTF-8"?>')
add('<!-- Figure 4. Generated by make_figure4.py. 1 user unit = 1 pt; '
    '510 pt = 180 mm. Black = present in the collected data, blue = '
    'synthesised by the method. -->')
add('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
    'width="%dpt" height="%dpt" '
    'font-family="Helvetica, \'Nimbus Sans\', Arimo, Arial, sans-serif" '
    'fill="%s">' % (W, Hh, W, Hh, INK))
add('  <defs>')
for mid, col in (('aInk', INK), ('aAcc', ACC)):
    add('    <marker id="%s" viewBox="0 0 10 10" refX="9.5" refY="5" '
        'markerWidth="4.5" markerHeight="4.5" orient="auto-start-reverse">'
        '<path d="M0 0 L10 5 L0 10 Z" fill="%s"/></marker>' % (mid, col))
add('  </defs>')
add('  <rect x="0" y="0" width="%d" height="%d" fill="#FFFFFF"/>' % (W, Hh))

# ---- stage headings
for x, letter, lines in (
        (C1X, 'a', ('Endpoint patents of the', 'predicted co-citation link')),
        (C2X, 'b', ('Local citation subgraph,', 'as collected')),
        (C3X, 'c', ('Prospective node', 'inserted')),
        (C4X, 'd', ('Concept decoded from', 'the inserted node'))):
    txt(x, 12, '(%s)' % letter, 7.5, 'bold')
    for i, ln in enumerate(lines):
        txt(x + 13, 12 + i * 7.6, esc(ln), 6.5, 'bold')
line(C1X, 23.5, 502, 23.5, RULE, 0.6)

# ---- flow arrows between stages
for x0 in (C1X + C1W, C2X + C2W, C3X + C3W):
    line(x0 + 1, 97, x0 + 10, 97, INK, 0.8, marker='aInk')

# ================================================================= stage (a)
rect(C1X, TOP, C1W, 52, FILL)
circ(C1X + C1W - 7, TOP + 7, 5.5, '#FFFFFF', INK, 0.7)
txt(C1X + C1W - 7, TOP + 8.9, 'A', 5.5, 'bold', INK, 'middle')
txt(C1X + 5, 36, 'US-10383765-B2', 6.5, 'bold', ls=0.1)
txt(C1X + 5, 43, 'granted 2019-08-20', 5, fill=MUTED)
for i, ln in enumerate(('Apparatus and method for a',
                        'global coordinate system for',
                        'use in robotic surgery')):
    txt(C1X + 5, 52.5 + i * 7.4, ln, 6)
txt(C1X + 5, 76.5, 'co-citation degree 848', 5, fill=MUTED)

txt(C1X + C1W / 2, 87, 'PREDICTED CO-CITATION', 4.8, 'bold', ACC, 'middle', 0.35)
txt(C1X + C1W / 2, 96.5, 'p = 0.963', 8.5, 'bold', ACC, 'middle')
line(C1X + 16, 101, C1X + C1W - 16, 101, ACC, 0.9, dash='2.5 2')
txt(C1X + C1W / 2, 107, '0 co-citations observed', 5, fill=MUTED, anchor='middle')
txt(C1X + C1W / 2, 112.5, '189 neighbours shared', 5, fill=MUTED, anchor='middle')

rect(C1X, 114, C1W, 52, FILL)
circ(C1X + C1W - 7, 121, 5.5, '#FFFFFF', INK, 0.7)
txt(C1X + C1W - 7, 122.9, 'B', 5.5, 'bold', INK, 'middle')
txt(C1X + 5, 122, 'US-2022101745-A1', 6.5, 'bold', ls=0.1)
txt(C1X + 5, 129, 'published 2022-03-31', 5, fill=MUTED)
for i, ln in enumerate(('Virtual reality system for',
                        'simulating a robotic',
                        'surgical environment')):
    txt(C1X + 5, 138.5 + i * 7.4, ln, 6)
txt(C1X + 5, 162.5, 'co-citation degree 220', 5, fill=MUTED)

# ============================================================ stages (b), (c)
rect(C2X, TOP, C2W, PANH)
draw_graph(C2X, C2W, show_prospective=False)
txt(C2X + C2W / 2, 120, '2 components', 6, 'bold', INK, 'middle')
txt(C2X + C2W / 2, 128, '5 nodes  \u00b7  3 edges', 5, fill=MUTED, anchor='middle')
for i, ln in enumerate(('A and B sit in separate',
                        'components. No seed',
                        'patent cites both, so the',
                        'pair is never co-cited.')):
    txt(C2X + C2W / 2, 139 + i * 6.6, ln, 5, fill=MUTED, anchor='middle')

rect(C3X, TOP, C3W, PANH)
draw_graph(C3X, C3W, show_prospective=True)
txt(C3X + C3W / 2, 120, '1 component', 6, 'bold', ACC, 'middle')
txt(C3X + C3W / 2, 128, '8 nodes  \u00b7  7 edges', 5, fill=MUTED, anchor='middle')
for i, ln in enumerate(('P27 carries the predicted',
                        'link and is the only path',
                        'from A to B. The network',
                        'grows to 5,265 nodes.')):
    txt(C3X + C3W / 2, 139 + i * 6.6, ln, 5, fill=MUTED, anchor='middle')

# ================================================================= stage (d)
rect(C4X, TOP, C4W, PANH, SOFT, ACC, 0.8)
txt(C4X + 6, 37, 'VEC2TEXT INVERSION \u00b7 60 STEPS \u00b7 BEAM 4',
    4.8, 'bold', ACC, ls=0.3)
for i, ln in enumerate(('Sensor-synchronized', 'visualization of the',
                        'surgical scene')):
    txt(C4X + 6, 48 + i * 8, ln, 7, 'bold')
BODY = ('One or more methods of executing a surgical',
        'system are described, including the',
        'manipulation of a virtual object by a',
        'computer-aided robotic device to generate an',
        'anatomical image of a surgical scene. The',
        'system generates an image of the surgical',
        'scene based on the user\u2019s perception of the',
        'location and orientation of the surgical')
for i, ln in enumerate(BODY):
    txt(C4X + 6, 78 + i * 8.6, ln, 6.5)
add('  <text x="%g" y="%g" font-size="6.5">scene. <tspan fill="%s">'
    '[\u2026]</tspan></text>' % (C4X + 6, 78 + len(BODY) * 8.6, MUTED))
txt(C4X + 6, 160, 'opening 2 of 4 sentences, verbatim', 5, fill=MUTED)

# =================================================================== footnote
txt(C1X, 176,
    'A, B = link endpoints (solid)  \u00b7  C1 = US-12268460-B2 (cites 610 '
    'patents)  \u00b7  C2 = US-11957421-B2 (246)  \u00b7  C3 = US-12290319-B2 '
    '(221)  \u00b7  P25, P26 = two further predicted links on A.',
    5, fill=MUTED)
txt(C1X, 183.5,
    'Positions: Kamada\u2013Kawai layout of the induced subgraph, shared by (b) '
    'and (c). Node size scales with degree in the full citation network; the '
    'prospective node is drawn enlarged.', 5, fill=MUTED)
txt(C1X, 191,
    'The decoded text is nearest to US-10058393-B2 (2018) within the corpus, '
    'cosine 0.951, and to US-2025288363-A1 (2025) among patents published after '
    'the collection window, cosine 0.929.', 5, fill=MUTED)
add('</svg>')

svg = '\n'.join(o) + '\n'
open(_ROOT + '/figures/figure4_walkthrough.svg', 'w').write(svg)

import cairosvg
cairosvg.svg2pdf(url=_ROOT + '/figures/figure4_walkthrough.svg', write_to=_ROOT + '/figures/figure4_walkthrough.pdf')
cairosvg.svg2png(url=_ROOT + '/figures/figure4_walkthrough.svg', write_to=_ROOT + '/figures/figure4_walkthrough.png',
                 scale=6.67, background_color='white')
print('figure4_walkthrough.{svg,pdf,png}  %d x %d pt  = %.1f x %.1f mm'
      % (W, Hh, W * 25.4 / 72, Hh * 25.4 / 72))
print('observed components=%d  with P27=1' % len(COMPONENTS))
