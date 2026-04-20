"""Exporta el grafo 23-F a Gephi (.gexf) con atributos visuales:
color por comunidad y tamaño por betweenness.
"""
from __future__ import annotations

from pathlib import Path
import colorsys
import networkx as nx


def _palette(n: int) -> list[tuple[int, int, int]]:
    out = []
    k = max(n, 1)
    for i in range(k):
        r, g, b = colorsys.hsv_to_rgb(i / k, 0.65, 0.9)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def export_gexf(G: nx.Graph, out_path: str | Path) -> Path:
    H = G.copy()

    communities = sorted({d.get("community", 0) for _, d in H.nodes(data=True)})
    palette = _palette(len(communities))
    colors = {c: palette[i] for i, c in enumerate(communities)}

    betws = [d.get("betweenness", 0.0) for _, d in H.nodes(data=True)]
    bmin, bmax = (min(betws), max(betws)) if betws else (0.0, 1.0)
    span = (bmax - bmin) or 1.0

    for _, d in H.nodes(data=True):
        cid = d.get("community", 0)
        r, g, b = colors[cid]
        size = 5.0 + 25.0 * ((d.get("betweenness", 0.0) - bmin) / span)
        d["viz"] = {
            "color": {"r": r, "g": g, "b": b, "a": 1.0},
            "size": float(size),
        }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(H, out)
    return out
