"""Fake vs real cascades on UPFD: size, depth, breadth, structural virality
(Goel et al. 2016), per class.

    uv run python -m factnet.graph.propagation_stats
"""

from __future__ import annotations

import statistics
from collections import deque
from pathlib import Path

from torch_geometric.datasets import UPFD

ROOT = str(Path(__file__).resolve().parents[3] / "data" / "raw" / "upfd")


def _adjacency(data) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(data.num_nodes)]
    for s, t in data.edge_index.t().tolist():
        adj[s].append(t)
        adj[t].append(s)
    return adj


def cascade_shape(data) -> dict[str, float]:
    adj = _adjacency(data)
    depth = {0: 0}
    queue, order = deque([0]), [0]
    while queue:
        node = queue.popleft()
        for nxt in adj[node]:
            if nxt not in depth:
                depth[nxt] = depth[node] + 1
                order.append(nxt)
                queue.append(nxt)
    levels: dict[int, int] = {}
    for d in depth.values():
        levels[d] = levels.get(d, 0) + 1
    return {
        "size": float(data.num_nodes),
        "depth": float(max(depth.values())),
        "breadth": float(max(levels.values())),
        "direct_shares": float(sum(1 for d in depth.values() if d == 1)),
        # not really the pairwise version from Goel et al, just mean distance to the root
        "virality": float(statistics.fmean(depth.values())),
    }


def compare(name: str = "politifact", feature: str = "profile") -> dict[str, dict[str, float]]:
    """mean shape per class (UPFD: 0 = fake, 1 = real)"""
    shapes: dict[int, list[dict[str, float]]] = {0: [], 1: []}
    for split in ("train", "val", "test"):
        for data in UPFD(ROOT, name, feature, split=split):
            shapes[int(data.y)].append(cascade_shape(data))
    return {
        "fake": {k: statistics.fmean(s[k] for s in shapes[0]) for k in shapes[0][0]},
        "real": {k: statistics.fmean(s[k] for s in shapes[1]) for k in shapes[1][0]},
        "counts": {"fake": float(len(shapes[0])), "real": float(len(shapes[1]))},
    }


def main():
    for name in ("politifact", "gossipcop"):
        stats = compare(name)
        fake, real = stats["fake"], stats["real"]
        counts = stats["counts"]
        print(f"\nCascade shape on UPFD {name}  "
              f"({int(counts['fake'])} fake / {int(counts['real'])} real)\n")
        print(f"{'measure':16s} {'fake':>9s} {'real':>9s} {'ratio':>8s}")
        print("-" * 45)
        for key in ("size", "depth", "breadth", "direct_shares", "virality"):
            ratio = fake[key] / real[key] if real[key] else float("nan")
            print(f"{key:16s} {fake[key]:>9.2f} {real[key]:>9.2f} {ratio:>8.2f}")


if __name__ == "__main__":
    main()
