"""Plot / ranking helpers for the streamlit viewer (kept separate so tests don't need streamlit)."""

from __future__ import annotations

from collections import deque
from typing import Any

from factnet.graph.influence import influence_ranking

PALETTE = {"source": "#0FA297", "repost": "#14284D", "reply": "#6B7C93"}


def from_pyg(data: Any, label: int | None = None) -> Any:
    """PyG Data -> nx.DiGraph"""
    import networkx as nx

    graph = nx.DiGraph(label=label if label is not None else int(getattr(data, "y", 0)))
    for node in range(data.num_nodes):
        graph.add_node(node, handle=f"account {node}" if node else "news source",
                       kind="source" if node == 0 else "repost")
    for source, target in data.edge_index.t().tolist():
        graph.add_edge(source, target, kind="repost")
    return graph


def annotate_influence(graph: Any) -> list[tuple[Any, float]]:
    if graph.number_of_nodes() == 0:
        return []
    ranked, reach, pagerank, coreness = influence_ranking(graph, top_k=graph.number_of_nodes())
    for node, score in ranked:
        graph.nodes[node].update(influence=score, reach=reach[node],
                                 pagerank=pagerank[node], coreness=coreness[node])
    return ranked


def hierarchy_layout(graph: Any, root: Any = None) -> dict[Any, tuple[float, float]]:
    """one row per hop, source on top"""
    if graph.number_of_nodes() == 0:
        return {}
    root = root if root is not None else next(iter(graph.nodes))
    levels: dict[Any, int] = {root: 0}
    queue = deque([root])
    undirected = graph.to_undirected(as_view=True)
    while queue:
        node = queue.popleft()
        for neighbour in undirected.neighbors(node):
            if neighbour not in levels:
                levels[neighbour] = levels[node] + 1
                queue.append(neighbour)
    for node in graph.nodes:  # unreachable -> last row
        levels.setdefault(node, max(levels.values(), default=0) + 1)

    rows: dict[int, list[Any]] = {}
    for node, depth in levels.items():
        rows.setdefault(depth, []).append(node)

    positions = {}
    for depth, nodes in rows.items():
        span = max(1, len(nodes) - 1)
        for i, node in enumerate(sorted(nodes, key=str)):
            x = 0.5 if len(nodes) == 1 else i / span
            positions[node] = (x, -float(depth))
    return positions


def draw(graph: Any, highlight: int = 5, title: str = "") -> Any:
    import matplotlib.pyplot as plt
    import networkx as nx

    ranked = annotate_influence(graph)
    top = {node for node, _ in ranked[:highlight]}
    positions = hierarchy_layout(graph)

    figure, axes = plt.subplots(figsize=(9, 5.2))
    nx.draw_networkx_edges(graph, positions, ax=axes, edge_color="#C7CED8",
                           arrows=False, width=0.8)
    sizes = [40 + 900 * graph.nodes[n].get("influence", 0.0) for n in graph.nodes]
    colours = [PALETTE.get(graph.nodes[n].get("kind", "repost"), PALETTE["repost"])
               for n in graph.nodes]
    edge_colours = ["#D94F30" if n in top else "none" for n in graph.nodes]
    nx.draw_networkx_nodes(graph, positions, ax=axes, node_size=sizes,
                           node_color=colours, edgecolors=edge_colours, linewidths=1.6)
    axes.set_title(title or "Propagation cascade", loc="left", fontsize=11)
    axes.axis("off")
    figure.tight_layout()
    return figure


def spreader_table(graph: Any, top_k: int = 10) -> list[dict[str, Any]]:
    ranked = annotate_influence(graph)[:top_k]
    return [{"rank": i, "account": graph.nodes[node].get("handle", str(node)),
             "influence": round(score, 3),
             "reach": round(graph.nodes[node].get("reach", 0.0), 3),
             "pagerank": round(graph.nodes[node].get("pagerank", 0.0), 3),
             "k-core": round(graph.nodes[node].get("coreness", 0.0), 3)}
            for i, (node, score) in enumerate(ranked, 1)]
