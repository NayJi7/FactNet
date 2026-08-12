"""Is early detection a property of the cascade, or of averaging a sample?

The early-detection result is that a model trained on complete cascades keeps
its score when shown only the first fraction of one, and the article reads that
as the shape of a cascade being legible from its opening.

There is a duller reading, and masking the root did not test it. The detector
mean-pools its node representations. The mean of a feature over the first 20 per
cent of the nodes is an estimator of the mean over all of them, and it converges
quickly. A model whose decision is a function of that mean would therefore hold
its score under truncation for a reason that has nothing to do with propagation:
a subsample estimates a mean.

The test removes the graph entirely. Logistic regression is fitted on the mean
of the account features of complete training cascades, exactly as in
``trivial_baselines``, and evaluated on test cascades truncated to the same
breadth-first fractions the graph model is evaluated on. No edge is read at any
point, so nothing here can be a property of the structure. If this curve is flat
too, flatness is what averaging does and not what a cascade is.

    uv run python -m factnet.graph.early_detection_edgeless
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch_geometric.datasets import UPFD

from factnet.graph.early_detection import FRACTIONS, truncate
from factnet.graph.train_upfd import ROOT

RESULTS = Path(__file__).resolve().parents[3] / "data" / "results"
ROUNDS = 2000


def accounts_mean(graph) -> np.ndarray:
    """Mean of the account features, the root excluded, as the baseline uses."""
    x = graph.x.numpy()
    return x[1:].mean(0) if len(x) > 1 else np.zeros_like(x[0])


def design(dataset, fraction: float):
    rows, labels = [], []
    for graph in dataset:
        cut = graph if fraction >= 1.0 else truncate(graph, fraction)
        rows.append(accounts_mean(cut))
        labels.append(int(graph.y))
    return np.array(rows), np.array(labels)


def interval(truth, pred, rounds=ROUNDS, seed=0):
    rng = np.random.default_rng(seed)
    n = len(truth)
    scores = np.empty(rounds)
    for i in range(rounds):
        idx = rng.integers(0, n, n)
        scores[i] = (f1_score(truth[idx], pred[idx], average="macro")
                     if len(np.unique(truth[idx])) > 1 else np.nan)
    return np.nanpercentile(scores, [2.5, 97.5])


def sweep(name: str, feature: str) -> dict:
    train = UPFD(ROOT, name, feature, split="train")
    test = UPFD(ROOT, name, feature, split="test")

    Xtr, ytr = design(train, 1.0)                      # fitted on complete cascades
    model = LogisticRegression(max_iter=5000, class_weight="balanced").fit(Xtr, ytr)

    out = {}
    for fraction in FRACTIONS:
        Xte, yte = design(test, fraction)
        pred = model.predict(Xte)
        low, high = interval(yte, pred)
        out[f"{fraction:.0%}"] = {"macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
                                  "ci95": [round(float(low), 4), round(float(high), 4)]}
    return out


def main() -> None:
    table = {}
    for name, feature in (("politifact", "bert"), ("gossipcop", "bert"),
                          ("politifact", "profile"), ("gossipcop", "profile")):
        res = sweep(name, feature)
        key = f"{name}/{feature}"
        table[key] = res
        cells = " ".join(f"{res[f'{f:.0%}']['macro_f1']:.3f}".rjust(8) for f in FRACTIONS)
        values = [res[f"{f:.0%}"]["macro_f1"] for f in FRACTIONS]
        print(f"{key:20s} {cells}   spread {max(values) - min(values):.3f}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "early_detection_edgeless.json"
    path.write_text(json.dumps(table, indent=2) + "\n")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
