"""Train and evaluate propagation-graph detectors on UPFD.

Baseline (GCN/GAT) and the main model (Bi-GCN), reported with accuracy and
macro-F1 on the test split. Small dataset, runs on CPU.

    uv run python -m factnet.graph.train_upfd
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

from factnet.graph.models import MODELS

ROOT = str(Path(__file__).resolve().parents[3] / "data" / "raw" / "upfd")


def load(name: str = "politifact", feature: str = "profile"):
    splits = [UPFD(ROOT, name, feature, split=s) for s in ("train", "val", "test")]
    return splits  # train, val, test


def run(model_cls, train_ds, test_ds, epochs: int = 60, hidden: int = 64,
        lr: float = 0.01, seed: int = 0):
    torch.manual_seed(seed)
    model = model_cls(train_ds.num_features, hidden, train_ds.num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)

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
    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro")


def main():
    train_ds, _val, test_ds = load()
    print(f"UPFD politifact  |  train {len(train_ds)}  test {len(test_ds)}  "
          f"feat {train_ds.num_features}\n")
    print(f"{'model':10s} {'accuracy':>10s} {'macro-F1':>10s}")
    print("-" * 32)
    for name, cls in MODELS.items():
        acc, f1 = run(cls, train_ds, test_ds)
        tag = "  (main)" if name == "Bi-GCN" else ""
        print(f"{name:10s} {acc:>10.3f} {f1:>10.3f}{tag}")


if __name__ == "__main__":
    main()
