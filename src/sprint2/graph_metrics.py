"""23-F graph metrics: degree, betweenness, Louvain communities,
and ranking of inter-community brokers.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities


def compute_metrics(G: nx.Graph, seed: int = 42) -> pd.DataFrame:
    """Compute degree, degree_centrality, betweenness, and community_id per node.

    Returns a DataFrame sorted by descending betweenness and also writes those
    values as graph node attributes (for Gephi export).
    """
    degree = dict(G.degree(weight="weight"))
    deg_cent = nx.degree_centrality(G)
    betw = nx.betweenness_centrality(G, weight="weight", seed=seed)

    communities = louvain_communities(G, weight="weight", seed=seed)
    comm_map = {n: cid for cid, comm in enumerate(communities) for n in comm}
    for n in G.nodes():
        G.nodes[n]["community"] = comm_map.get(n, -1)
        G.nodes[n]["betweenness"] = betw[n]
        G.nodes[n]["degree"] = degree[n]

    rows = [
        {
            "entity": n,
            "type": G.nodes[n].get("type", "UNK"),
            "degree": degree[n],
            "degree_centrality": deg_cent[n],
            "betweenness": betw[n],
            "community_id": comm_map[n],
        }
        for n in G.nodes()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("betweenness", ascending=False)
        .reset_index(drop=True)
    )


def top_brokers(metrics: pd.DataFrame, G: nx.Graph, top_n: int = 10) -> pd.DataFrame:
    """Top-N brokers: high betweenness and links to >= 2 different communities."""
    comm_by_entity = dict(zip(metrics["entity"], metrics["community_id"]))
    rows = []
    for _, r in metrics.iterrows():
        node = r["entity"]
        if node not in G:
            continue
        neigh_comms = {comm_by_entity[nb] for nb in G.neighbors(node)
                       if nb in comm_by_entity}
        if len(neigh_comms) >= 2:
            rows.append({**r.to_dict(), "n_communities_bridged": len(neigh_comms)})
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows)


def save_metrics(metrics: pd.DataFrame, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out, index=False)
    return out
