"""Integration point: the content score goes in an extra column on the root node
(0 elsewhere).

Dry run: Bi-GCN with no score / random score / noisy label as score. Random
shouldn't help, the noisy oracle should.

    uv run python -m factnet.graph.integration
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

from factnet.graph.models import BiGCN

ROOT = str(Path(__file__).resolve().parents[3] / "data" / "raw" / "upfd")


def attach_scores(dataset, scores) -> list[Data]:
    """+1 column, score on the root. strict zip, a short list must not silently drop graphs."""
    scores = list(scores)
    if len(scores) != len(dataset):
        raise ValueError(f"{len(dataset)} graphs but {len(scores)} scores")
    out = []
    for data, score in zip(dataset, scores, strict=True):
        col = torch.zeros(data.num_nodes, 1)
        col[0, 0] = float(score)
        out.append(Data(x=torch.cat([data.x, col], dim=1),
                        edge_index=data.edge_index, y=data.y))
    return out


def _scores(kind: str, dataset, seed: int = 0) -> torch.Tensor:
    """fake scores for the dry run"""
    gen = torch.Generator().manual_seed(seed)
    y = torch.tensor([int(d.y) for d in dataset], dtype=torch.float)
    if kind == "none":
        return torch.zeros(len(y))
    if kind == "random":
        return torch.rand(len(y), generator=gen)
    if kind == "informative":  # label + noise
        noisy = 0.1 + 0.8 * y + 0.15 * torch.randn(len(y), generator=gen)
        return noisy.clamp(0.0, 1.0)
    raise ValueError(kind)


def _train_eval(train_list, test_list, in_dim: int, epochs: int = 60,
                hidden: int = 64, lr: float = 0.01, seed: int = 0) -> float:
    torch.manual_seed(seed)
    model = BiGCN(in_dim, hidden, 2)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    train_loader = DataLoader(train_list, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_list, batch_size=128)
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            F.cross_entropy(out, batch.y).backward()
            opt.step()
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch.x, batch.edge_index, batch.batch).argmax(dim=1)
            ps += pred.tolist()
            ys += batch.y.tolist()
    return f1_score(ys, ps, average="macro")


def main():
    train_ds = UPFD(ROOT, "politifact", "profile", split="train")
    test_ds = UPFD(ROOT, "politifact", "profile", split="test")
    in_dim = train_ds.num_features + 1
    print("Integration interface dry run  |  Bi-GCN, score column on the news root\n")
    print(f"{'story scores':14s} {'macro-F1':>9s}")
    print("-" * 25)
    for kind in ("none", "random", "informative"):
        train_list = attach_scores(train_ds, _scores(kind, train_ds))
        test_list = attach_scores(test_ds, _scores(kind, test_ds, seed=1))
        f1 = _train_eval(train_list, test_list, in_dim)
        print(f"{kind:14s} {f1:>9.3f}")


if __name__ == "__main__":
    main()
