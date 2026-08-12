"""The same question asked of an article instead of a sentence.

Every model on LIAR stops near 0.63 macro-F1. The claim this module supports is
that the bound belongs to the unit of text being judged and not to content
analysis as such, and the cheapest way to test it is to change only the unit.

UPFD stores, at the root of each propagation cascade, a 768-dimensional
embedding of the news article the cascade carries. Logistic regression on that
vector alone reads a whole article and nothing else: no accounts, no edges, no
cascade. If it clears 0.63 by a wide margin on both corpora, the sentence was
the binding constraint.

The protocol is the published train split with balanced class weights and no
other tuning, matching ``graph.trivial_baselines`` exactly so that the two
articles report one number for this experiment rather than two. Intervals come from resampling the
test split, which is the only uncertainty a fixed fit has.

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
    """One article embedding per cascade, and the cascade's label."""
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
