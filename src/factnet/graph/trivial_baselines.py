"""Edgeless baselines on UPFD: same features, no edges, logreg.

  root       root feature vector
  users      mean of the account vectors (no root)
  mean-pool  mean over all nodes + size
  size       number of accounts only (our bluesky data leans on this a lot)

CIs by bootstrapping the test split (only 221 graphs on politifact).
class_weight="balanced" everywhere, it's the strongest setting for a baseline.
Same setup as nlp.article_level, so the `root` column = the article-level number.

    uv run python -m factnet.graph.trivial_baselines
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch_geometric.datasets import UPFD

from factnet.graph.train_upfd import ROOT

RESULTS = Path(__file__).resolve().parents[3] / "data" / "results"
ROUNDS = 2000
BASELINES = ("root", "users", "mean-pool", "size")


def represent(graph, kind: str) -> np.ndarray:
    """cascade -> one vector, no edges"""
    x = graph.x.numpy()
    if kind == "root":
        return x[0]
    if kind == "users":
        return x[1:].mean(0) if len(x) > 1 else np.zeros_like(x[0])
    if kind == "mean-pool":
        return np.concatenate([x.mean(0), [len(x)]])
    if kind == "size":
        return np.array([len(x)], dtype=float)
    raise ValueError(kind)


def design(dataset, kind: str):
    return (np.array([represent(g, kind) for g in dataset]),
            np.array([int(g.y) for g in dataset]))


def interval(truth, pred, rounds: int = ROUNDS, seed: int = 0):
    """bootstrap CI over the test split"""
    rng = np.random.default_rng(seed)
    n = len(truth)
    scores = np.empty(rounds)
    for i in range(rounds):
        idx = rng.integers(0, n, n)
        scores[i] = (f1_score(truth[idx], pred[idx], average="macro")
                     if len(np.unique(truth[idx])) > 1 else np.nan)
    return np.nanpercentile(scores, [2.5, 97.5])


def evaluate(name: str, feature: str) -> dict[str, dict]:
    train = UPFD(ROOT, name, feature, split="train")
    test = UPFD(ROOT, name, feature, split="test")
    out = {}
    for kind in BASELINES:
        Xtr, ytr = design(train, kind)
        Xte, yte = design(test, kind)
        pred = (LogisticRegression(max_iter=5000, class_weight="balanced")
                .fit(Xtr, ytr).predict(Xte))
        low, high = interval(yte, pred)
        out[kind] = {"macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
                     "ci95": [round(float(low), 4), round(float(high), 4)]}
    return out


def main() -> None:
    table = {}
    header = " ".join(f"{k:>22s}" for k in BASELINES)
    print(f"{'dataset':11s} {'feat':8s} {header}")
    print("-" * (20 + 23 * len(BASELINES)))
    for name in ("politifact", "gossipcop"):
        for feature in ("profile", "bert"):
            scores = evaluate(name, feature)
            table[f"{name}/{feature}"] = scores
            cells = []
            for k in BASELINES:
                low, high = scores[k]["ci95"]
                cells.append(f"{scores[k]['macro_f1']:.3f} [{low:.3f},{high:.3f}]".rjust(22))
            print(f"{name:11s} {feature:8s} " + " ".join(cells))

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "trivial_baselines.json"
    path.write_text(json.dumps(table, indent=2) + "\n")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
