"""Early detection: Bi-GCN trained on full cascades, tested on the first 20..80%.

No timestamps in UPFD so we use BFS order from the root as arrival order.

    uv run python -m factnet.graph.early_detection
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph

from factnet.graph.models import BiGCN

ROOT = str(Path(__file__).resolve().parents[3] / "data" / "raw" / "upfd")
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def _bfs_order(data: Data) -> list[int]:
    """BFS from node 0, ties by index"""
    adj: list[list[int]] = [[] for _ in range(data.num_nodes)]
    for s, t in data.edge_index.t().tolist():
        adj[s].append(t)
        adj[t].append(s)
    seen, order, queue = {0}, [], deque([0])
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in sorted(adj[node]):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    order += sorted(set(range(data.num_nodes)) - set(order))  # unreachable ones at the end
    return order


def truncate(data: Data, fraction: float) -> Data:
    """first `fraction` of nodes in BFS order, root always kept"""
    if fraction >= 1.0:
        return data
    order = _bfs_order(data)
    keep = torch.tensor(sorted(order[: max(1, math.ceil(fraction * len(order)))]))
    edge_index, _ = subgraph(keep, data.edge_index, relabel_nodes=True,
                             num_nodes=data.num_nodes)
    return Data(x=data.x[keep], edge_index=edge_index, y=data.y)


def train_bigcn(train_ds, epochs: int = 60, hidden: int = 64, lr: float = 0.01,
                seed: int = 0) -> BiGCN:
    torch.manual_seed(seed)
    model = BiGCN(train_ds.num_features, hidden, train_ds.num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    for _ in range(epochs):
        model.train()
        for batch in loader:
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            F.cross_entropy(out, batch.y).backward()
            opt.step()
    return model


def evaluate(model: BiGCN, dataset, fraction: float) -> float:
    loader = DataLoader([truncate(d, fraction) for d in dataset], batch_size=128)
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch.x, batch.edge_index, batch.batch).argmax(dim=1)
            ps += pred.tolist()
            ys += batch.y.tolist()
    return f1_score(ys, ps, average="macro")


def main():
    train_ds = UPFD(ROOT, "politifact", "profile", split="train")
    test_ds = UPFD(ROOT, "politifact", "profile", split="test")
    model = train_bigcn(train_ds)
    print("Early detection  |  Bi-GCN trained on full cascades, "
          "evaluated on truncated ones\n")
    print(f"{'observed':>9s} {'macro-F1':>9s}")
    print("-" * 20)
    for fraction in FRACTIONS:
        f1 = evaluate(model, test_ds, fraction)
        print(f"{fraction:>8.0%} {f1:>9.3f}")


if __name__ == "__main__":
    main()
