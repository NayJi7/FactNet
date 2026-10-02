"""Root ablation. In UPFD node 0 is the news item, and with `bert` it holds the
article embedding, so a "graph" model is partly reading text.

  full     as is
  masked   same graph, root features = 0   <- the one that matters
  removed  root and its edges deleted

    uv run python -m factnet.graph.root_ablation
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch_geometric.data import Data

from factnet.graph.models import BiGCN
from factnet.graph.train_upfd import ROOT, run

RESULTS = Path(__file__).resolve().parents[3] / "data" / "results"
CONDITIONS = ("full", "masked", "removed")


def _mask_root(g: Data) -> Data:
    x = g.x.clone()
    x[0] = 0.0
    return Data(x=x, edge_index=g.edge_index, y=g.y)


def _remove_root(g: Data) -> Data:
    keep = torch.arange(1, g.num_nodes)
    if keep.numel() == 0:
        return Data(x=torch.zeros(1, g.num_features), y=g.y,
                    edge_index=torch.empty(2, 0, dtype=torch.long))
    src, dst = g.edge_index
    alive = (src != 0) & (dst != 0)
    edges = g.edge_index[:, alive] - 1
    return Data(x=g.x[keep], edge_index=edges, y=g.y)


TRANSFORMS = {"full": lambda g: g, "masked": _mask_root, "removed": _remove_root}


def _materialise(dataset, condition: str) -> list[Data]:
    """PyG datasets are lazy, we need real Data objects to modify"""
    fn = TRANSFORMS[condition]
    return [fn(g) for g in dataset]


class _Wrapped(list):
    """hack: list with the 2 attributes run() reads off a dataset"""

    def __init__(self, graphs, num_features, num_classes):
        super().__init__(graphs)
        self.num_features = num_features
        self.num_classes = num_classes


def ablate(name: str, feature: str, seeds=(0, 1, 2)) -> dict[str, dict]:
    from torch_geometric.datasets import UPFD

    train_raw = UPFD(ROOT, name, feature, split="train")
    test_raw = UPFD(ROOT, name, feature, split="test")
    nf, nc = train_raw.num_features, train_raw.num_classes

    out = {}
    for condition in CONDITIONS:
        tr = _Wrapped(_materialise(train_raw, condition), nf, nc)
        te = _Wrapped(_materialise(test_raw, condition), nf, nc)
        f1s = [run(BiGCN, tr, te, seed=s)[1] for s in seeds]
        out[condition] = {
            "mean": round(statistics.mean(f1s), 4),
            "std": round(statistics.pstdev(f1s), 4),
            "seeds": [round(f, 4) for f in f1s],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))

    table = {}
    print(f"{'dataset':11s} {'feat':8s} {'full':>16s} {'masked':>16s} "
          f"{'removed':>16s} {'root worth':>11s}")
    print("-" * 84)
    for name in ("politifact", "gossipcop"):
        for feature in ("profile", "bert"):
            res = ablate(name, feature, seeds)
            table[f"{name}/{feature}"] = res
            cells = [f"{res[c]['mean']:.3f} ±{res[c]['std']:.3f}" for c in CONDITIONS]
            delta = res["full"]["mean"] - res["masked"]["mean"]
            print(f"{name:11s} {feature:8s} " + " ".join(f"{c:>16s}" for c in cells)
                  + f" {delta:>+11.3f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "root_ablation.json"
    path.write_text(json.dumps(table, indent=2) + "\n")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
