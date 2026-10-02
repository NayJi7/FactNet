"""How good is the injected score on its own?

It's NOT the LIAR model (the papers first said it was), it's a small MLP on the UPFD
root embedding, 5-fold cross-fitted on train. Reproduced here and scored on test.

    uv run python -m factnet.graph.injected_score
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.nn import Linear
from torch_geometric.datasets import UPFD

from factnet.graph.train_upfd import ROOT

RESULTS = Path(__file__).resolve().parents[3] / "data" / "results"
SEEDS = (0, 1, 2)


class ContentMLP(torch.nn.Module):
    """root embedding only"""

    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.l1, self.l2 = Linear(in_dim, hidden), Linear(hidden, 2)

    def forward(self, x):
        return self.l2(F.dropout(F.relu(self.l1(x)), p=0.3, training=self.training))


def _roots(dataset):
    x = torch.stack([d.x[0] for d in dataset])
    y = torch.tensor([int(d.y) for d in dataset])
    return x, y


def _fit(x, y, seed: int, epochs: int = 120, lr: float = 1e-3) -> ContentMLP:
    torch.manual_seed(seed)
    model = ContentMLP(x.size(1))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    return model


@torch.no_grad()
def _score(model: ContentMLP, x) -> torch.Tensor:
    model.eval()
    return model(x).argmax(dim=1)


def measure(name: str) -> dict:
    train = UPFD(ROOT, name, "bert", split="train")
    test = UPFD(ROOT, name, "bert", split="test")
    x_tr, y_tr = _roots(train)
    x_te, y_te = _roots(test)
    scores = [f1_score(y_te, _score(_fit(x_tr, y_tr, seed), x_te), average="macro")
              for seed in SEEDS]
    return {"macro_f1": round(st.fmean(scores), 4),
            "std": round(st.pstdev(scores), 4),
            "train": len(train), "test": len(test)}


def main() -> None:
    report = {name: measure(name) for name in ("politifact", "gossipcop")}
    print("The content model whose score the integration ablation injects,\n"
          "measured on its own on the held-out split\n")
    for name, cell in report.items():
        print(f"  {name:11s} macro-F1 {cell['macro_f1']:.3f} "
              f"±{cell['std']:.3f}  on {cell['test']} test stories")
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "injected_score.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
