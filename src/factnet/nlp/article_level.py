"""Is the ~0.63 LIAR ceiling about short claims? Try full articles instead.

UPFD has a 768-d embedding of the whole article on the root node. Logreg on just
that vector (no graph). Same protocol as graph.trivial_baselines so both papers
report the same number. CIs by bootstrapping the test split.

    uv run python -m factnet.nlp.article_level
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
ROUNDS = 5000


def root_vectors(dataset) -> tuple[np.ndarray, np.ndarray]:
    """(root embeddings, labels)"""
    return (np.array([g.x[0].numpy() for g in dataset]),
            np.array([int(g.y) for g in dataset]))


def interval(truth: np.ndarray, pred: np.ndarray, rounds: int = ROUNDS, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(truth)
    scores = np.empty(rounds)
    for i in range(rounds):
        idx = rng.integers(0, n, n)
        scores[i] = (f1_score(truth[idx], pred[idx], average="macro")
                     if len(np.unique(truth[idx])) > 1 else np.nan)
    return np.nanpercentile(scores, [2.5, 97.5])


def evaluate(name: str) -> dict:
    train = UPFD(ROOT, name, "bert", split="train")
    test = UPFD(ROOT, name, "bert", split="test")
    Xtr, ytr = root_vectors(train)
    Xte, yte = root_vectors(test)
    model = LogisticRegression(max_iter=5000, class_weight="balanced").fit(Xtr, ytr)
    pred = model.predict(Xte)
    point = f1_score(yte, pred, average="macro")
    low, high = interval(yte, pred)
    return {"macro_f1": round(float(point), 4),
            "ci95": [round(float(low), 4), round(float(high), 4)],
            "train": len(train), "test": len(test)}


def main() -> None:
    table = {}
    print("Logistic regression on the root article embedding, nothing else\n")
    print(f"{'corpus':12s} {'train':>7s} {'test':>7s} {'macro-F1':>10s} {'95% interval':>20s}")
    print("-" * 60)
    for name in ("politifact", "gossipcop"):
        res = evaluate(name)
        table[name] = res
        low, high = res["ci95"]
        span = f"[{low:.3f}, {high:.3f}]"
        print(f"{name:12s} {res['train']:>7d} {res['test']:>7d} "
              f"{res['macro_f1']:>10.3f} {span:>20s}")

    print("\nLIAR, every model in the comparison, sits at 0.633 or below.")
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "article_level.json"
    path.write_text(json.dumps(table, indent=2) + "\n")
    print(f"written to {path}")


if __name__ == "__main__":
    main()
