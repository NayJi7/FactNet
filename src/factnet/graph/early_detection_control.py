"""Control: truncation always keeps the root, and with `bert` the root holds the
article embedding. Mask the root features and redo the sweep. Still flat -> the
cascade does the work, drops -> it was the root.

    uv run python -m factnet.graph.early_detection_control
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from torch_geometric.data import Data
from torch_geometric.datasets import UPFD

from factnet.graph.early_detection import FRACTIONS, evaluate, train_bigcn
from factnet.graph.root_ablation import _Wrapped
from factnet.graph.train_upfd import ROOT

RESULTS = Path(__file__).resolve().parents[3] / "data" / "results"
SEEDS = (0, 1, 2)


def masked(dataset) -> list[Data]:
    """root features = 0"""
    out = []
    for g in dataset:
        x = g.x.clone()
        x[0] = 0.0
        out.append(Data(x=x, edge_index=g.edge_index, y=g.y))
    return out


def sweep(name: str, feature: str) -> dict[str, dict]:
    train_raw = UPFD(ROOT, name, feature, split="train")
    test_raw = UPFD(ROOT, name, feature, split="test")
    nf, nc = train_raw.num_features, train_raw.num_classes

    conditions = {
        "full": (list(train_raw), list(test_raw)),
        "masked": (masked(train_raw), masked(test_raw)),
    }
    out: dict[str, dict] = {}
    for label, (tr, te) in conditions.items():
        train_ds = _Wrapped(tr, nf, nc)
        per_fraction: dict[str, list[float]] = {f"{f:.0%}": [] for f in FRACTIONS}
        for seed in SEEDS:
            model = train_bigcn(train_ds, seed=seed)
            for fraction in FRACTIONS:
                per_fraction[f"{fraction:.0%}"].append(evaluate(model, te, fraction))
        out[label] = {k: {"mean": round(st.fmean(v), 4),
                          "std": round(st.pstdev(v), 4)}
                      for k, v in per_fraction.items()}
    return out


def main() -> None:
    table = {}
    for name in ("politifact", "gossipcop"):
        for feature in ("profile", "bert"):
            res = sweep(name, feature)
            table[f"{name}/{feature}"] = res
            print(f"\n=== {name} / {feature}")
            cols = " ".join(f"{f:.0%}".rjust(14) for f in FRACTIONS)
            print(f"{'condition':10s} {cols}")
            for label in ("full", "masked"):
                cells = []
                for f in FRACTIONS:
                    cell = res[label][f"{f:.0%}"]
                    cells.append(f"{cell['mean']:.3f} ±{cell['std']:.3f}".rjust(14))
                print(f"{label:10s} " + " ".join(cells))
            spread = max(res["masked"][f"{f:.0%}"]["mean"] for f in FRACTIONS) - \
                min(res["masked"][f"{f:.0%}"]["mean"] for f in FRACTIONS)
            print(f"{'':10s} spread of the masked curve: {spread:.3f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "early_detection_control.json"
    path.write_text(json.dumps(table, indent=2) + "\n")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
