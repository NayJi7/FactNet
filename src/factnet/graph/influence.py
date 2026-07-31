"""Account-influence scoring on propagation graphs (the module's contribution).

Combines structural measures — k-core coreness and PageRank — with a degree
(activity) proxy into a single influence score, and ranks the top spreaders.
On UPFD the nodes are anonymised per cascade; on real collected data (Bluesky)
the same ranking maps to identifiable accounts.

    uv run python -m factnet.graph.influence
"""

from __future__ import annotations

import argparse
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


def build_graph_from_collected(path: str | Path):
    """Union of collected cascades, keyed by account so the same handle recurs.

    Unlike the benchmark, where accounts are anonymised per cascade, a real
    account appears in every cascade it took part in; merging on the account
    identity is what turns the ranking into a statement about people rather than
    about graph positions, and it is what makes k-core meaningful, since the
    union is no longer a forest of disjoint trees.
    """
    from factnet.ingestion.to_graph import read_cascades

    g = nx.DiGraph()
    for cascade in read_cascades(path):
        handles = {n["did"]: n.get("handle", n["did"]) for n in cascade["nodes"]}
        for did, handle in handles.items():
            if did not in g:
                g.add_node(did, handle=handle, cascades=0)
            g.nodes[did]["cascades"] += 1
        for edge in cascade["edges"]:
            source, target = edge["source"], edge["target"]
            # authors often reply within their own thread: that is not a share
            if source != target and source in handles and target in handles:
                g.add_edge(source, target)
    return g


def influence_ranking(g: nx.DiGraph, top_k: int = 15):
    # Downstream reach (subtree size) = spreading power on a cascade; PageRank and
    # k-core are the structural measures (k-core is trivial on trees, but carries
    # signal on the denser reshare networks of real collected data).
    pagerank = _normalise(nx.pagerank(g))
    undirected = nx.Graph(g)  # drops direction and any self loop k-core rejects
    undirected.remove_edges_from(nx.selfloop_edges(undirected))
    coreness = _normalise(nx.core_number(undirected))
    reach = _normalise({n: len(nx.descendants(g, n)) for n in g})
    score = {n: 0.5 * reach[n] + 0.3 * pagerank[n] + 0.2 * coreness[n] for n in g}
    ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k], reach, pagerank, coreness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected", default=None,
                        help="rank accounts over a collected file instead of the benchmark")
    args = parser.parse_args()

    if args.collected:
        g = build_graph_from_collected(args.collected)
        print(f"Collected reshare network  |  accounts {g.number_of_nodes()}  "
              f"shares {g.number_of_edges()}\n")
        top, reach, pr, core = influence_ranking(g)
        print(f"{'rank':>4s}  {'account':<28s} {'influence':>9s} {'reach':>7s} "
              f"{'pagerank':>9s} {'k-core':>7s}")
        print("-" * 68)
        for rank, (node, s) in enumerate(top, 1):
            print(f"{rank:>4d}  {g.nodes[node].get('handle', node)[:28]:<28s} {s:>9.3f} "
                  f"{reach[node]:>7.3f} {pr[node]:>9.3f} {core[node]:>7.3f}")
        return

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
