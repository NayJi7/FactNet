"""What can be had on UPFD without reading the graph at all?

A propagation model is only worth its complexity if the propagation is what it
is using. The honest way to establish that is to hand the same node features to
a model that cannot see a single edge, and check that it does worse.

Four such models are fitted here, all of them linear and none of them aware of
the topology:

``root``        the news node's own feature vector,
``users``       the mean of every account feature vector, root excluded,
``mean-pool``   the mean over all nodes, plus the cascade's size,
``size``        the number of accounts, and nothing else.

``size`` is included because it is the crudest shortcut available on these
benchmarks and the one our own collected data turned out to lean on. If a
single integer scores well here, any structural reading of the same dataset has
to be reported against it.

Two things about the protocol, both of which an earlier version got wrong.
The fit is deterministic on a fixed split, but that removes only seed variance
and not the uncertainty of the split itself, which on PolitiFact's 221 test
graphs is what separates most of these numbers. Intervals therefore come from
resampling the test split. And every fit balances the class weights. That is the
setting under which a trivial baseline is strongest, which is the setting this
table has to use: the claim being tested is that the graph adds nothing a simple
model could not get, and a baseline crippled by PolitiFact's 62 training graphs
would understate it. ``nlp.article_level`` uses the same setting, so the
``root`` column here and the article-level figures reported on the content side
are one experiment rather than two.

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
    """Collapse one cascade to a fixed vector, using no edge information."""
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
    """The uncertainty a deterministic fit still carries: the test split."""
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
