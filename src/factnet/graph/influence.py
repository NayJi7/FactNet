"""Account-influence scoring on propagation graphs (the module's contribution).

Combines structural measures — k-core coreness and PageRank — with a degree
(activity) proxy into a single influence score, and ranks the top spreaders.
On UPFD the nodes are anonymised per cascade; on real collected data (Bluesky)
the same ranking maps to identifiable accounts.

    uv run python -m factnet.graph.influence
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
from torch_geometric.datasets import UPFD
from torch_geometric.utils import to_networkx

ROOT = str(Path(__file__).resolve().parents[3] / "data" / "raw" / "upfd")


def _normalise(d: dict) -> dict:
    if not d:
        return d
    lo, hi = min(d.values()), max(d.values())
    span = (hi - lo) or 1.0
    return {k: (v - lo) / span for k, v in d.items()}


def build_graph(name: str = "politifact", feature: str = "profile", split: str = "train"):
    """Union of all cascades into one directed graph, nodes tagged by cascade."""
    ds = UPFD(ROOT, name, feature, split=split)
    g = nx.DiGraph()
    for i, data in enumerate(ds):
        cascade = to_networkx(data, to_undirected=False)
        g = nx.union(g, nx.relabel_nodes(cascade, {n: f"c{i}:n{n}" for n in cascade}))
    return g


def influence_ranking(g: nx.DiGraph, top_k: int = 15):
    # Downstream reach (subtree size) = spreading power on a cascade; PageRank and
    # k-core are the structural measures (k-core is trivial on trees, but carries
    # signal on the denser reshare networks of real collected data).
    pagerank = _normalise(nx.pagerank(g))
    coreness = _normalise(nx.core_number(g.to_undirected()))
    reach = _normalise({n: len(nx.descendants(g, n)) for n in g})
    score = {n: 0.5 * reach[n] + 0.3 * pagerank[n] + 0.2 * coreness[n] for n in g}
    ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k], reach, pagerank, coreness


def main():
    g = build_graph()
    print(f"Propagation graph  |  nodes {g.number_of_nodes()}  edges {g.number_of_edges()}\n")
    top, reach, pr, core = influence_ranking(g)
    print(f"{'rank':>4s}  {'node':<10s} {'influence':>9s} {'reach':>7s} "
          f"{'pagerank':>9s} {'k-core':>7s}")
    print("-" * 50)
    for rank, (node, s) in enumerate(top, 1):
        print(f"{rank:>4d}  {node:<10s} {s:>9.3f} {reach[node]:>7.3f} "
              f"{pr[node]:>9.3f} {core[node]:>7.3f}")


if __name__ == "__main__":
    main()
