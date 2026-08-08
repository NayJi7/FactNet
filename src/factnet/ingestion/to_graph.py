"""Turn collected Bluesky cascades into the graph objects the models expect.

The benchmark cascades reach the models as PyTorch Geometric graphs whose nodes
carry the ``profile`` feature vector; the records produced by the collector are
converted to exactly that shape so a model trained on the benchmark can be
evaluated on collected data without any change to its input format.

Counter features are compressed with ``log1p`` and scaled, mirroring how the
benchmark normalises its own account statistics: raw follower counts span
several orders of magnitude and would otherwise dominate every other feature.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

COUNTER_SLOTS = (2, 3, 5)   # followers, follows, posts
AGE_SLOT = 6
LENGTH_SLOTS = (7, 8, 9)    # handle, display name, description


def read_cascades(path: str | Path) -> list[dict]:
    """Read a JSONL file produced by the collector."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


BLANK_PROFILE = [0.0] * 10


def normalise_profile(profile: list[float] | None) -> list[float]:
    """Scale raw account statistics into a comparable range.

    A node may arrive without a profile: a cascade pasted by hand carries a
    topology and rarely the ten counters. The vector is then all zeros, which
    the pipeline detects as constant features and answers by reading the shape
    alone rather than by pretending to know the accounts.
    """
    out = list(profile) if profile else list(BLANK_PROFILE)
    if len(out) != len(BLANK_PROFILE):
        raise ValueError(f"a profile holds {len(BLANK_PROFILE)} numbers, got {len(out)}")
    for slot in COUNTER_SLOTS:
        out[slot] = math.log1p(max(0.0, out[slot])) / 15.0
    out[AGE_SLOT] = min(1.0, out[AGE_SLOT] / 3650.0)  # ten years saturates
    for slot in LENGTH_SLOTS:
        out[slot] = min(1.0, out[slot] / 300.0)
    return out


def cascade_to_networkx(cascade: dict) -> Any:
    """Directed graph of one cascade, source first, with normalised features."""
    import networkx as nx

    graph = nx.DiGraph(uri=cascade.get("uri", ""), text=cascade.get("text", ""),
                       label=cascade.get("label"))
    for node in cascade["nodes"]:
        graph.add_node(node["did"], handle=node.get("handle", ""),
                       kind=node.get("kind", ""),
                       features=normalise_profile(node.get("profile")))
    for edge in cascade["edges"]:
        if edge["source"] in graph and edge["target"] in graph:
            graph.add_edge(edge["source"], edge["target"], kind=edge.get("kind", ""))
    return graph


def cascade_to_pyg(cascade: dict) -> Any:
    """One cascade as a PyTorch Geometric ``Data`` object (root node first)."""
    import torch
    from torch_geometric.data import Data

    nodes = cascade["nodes"]
    order = sorted(range(len(nodes)), key=lambda i: nodes[i].get("kind") != "source")
    index = {nodes[i]["did"]: rank for rank, i in enumerate(order)}
    x = torch.tensor([normalise_profile(nodes[i].get("profile")) for i in order],
                     dtype=torch.float)

    pairs = [(index[e["source"]], index[e["target"]]) for e in cascade["edges"]
             if e["source"] in index and e["target"] in index]
    edge_index = (torch.tensor(pairs, dtype=torch.long).t().contiguous()
                  if pairs else torch.empty((2, 0), dtype=torch.long))

    label = cascade.get("label")
    data = Data(x=x, edge_index=edge_index)
    if label is not None:
        data.y = torch.tensor([int(label)])
    return data

