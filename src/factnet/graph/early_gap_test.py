"""GossipCop bert at 20%: graph 0.903 vs edgeless logreg 0.885. Real gap or noise?

Before we just compared seed std (0.009) with the split CI (0.010), which isn't
a test. Here: paired bootstrap on the difference, repeated for each seed and
pooled. Also redone with the root masked, since at 20% the root (article
embedding) could explain the gap.

    uv run python -m factnet.graph.early_gap_test
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader

from factnet.graph.early_detection import train_bigcn, truncate
from factnet.graph.early_detection_edgeless import accounts_mean
from factnet.graph.root_ablation import _Wrapped
from factnet.graph.train_upfd import ROOT

RESULTS = Path(__file__).resolve().parents[3] / "data" / "results"
DATASET, FEATURE, FRACTION = "gossipcop", "bert", 0.2
SEEDS = (0, 1, 2)
ROUNDS = 2000


def masked(dataset) -> list[Data]:
    out = []
    for g in dataset:
        x = g.x.clone()
        x[0] = 0.0
        out.append(Data(x=x, edge_index=g.edge_index, y=g.y))
    return out


@torch.no_grad()
def predict(model, graphs) -> np.ndarray:
    model.eval()
    out = []
    for batch in DataLoader(graphs, batch_size=128):
        out += model(batch.x, batch.edge_index, batch.batch).argmax(dim=1).tolist()
    return np.array(out)


def paired_gap(truth, a, b, rounds=ROUNDS, seed=0) -> np.ndarray:
    """macro-F1(a) - macro-F1(b), same resamples"""
    rng = np.random.default_rng(seed)
    n = len(truth)
    gaps = np.empty(rounds)
    for i in range(rounds):
        idx = rng.integers(0, n, n)
        if len(np.unique(truth[idx])) < 2:
            gaps[i] = 0.0
            continue
        gaps[i] = (f1_score(truth[idx], a[idx], average="macro")
                   - f1_score(truth[idx], b[idx], average="macro"))
    return gaps


def main() -> None:
    train = UPFD(ROOT, DATASET, FEATURE, split="train")
    test = UPFD(ROOT, DATASET, FEATURE, split="test")
    nf, nc = train.num_features, train.num_classes
    cut = [truncate(g, FRACTION) for g in test]
    truth = np.array([int(g.y) for g in test])

    edgeless = LogisticRegression(max_iter=5000, class_weight="balanced").fit(
        np.array([accounts_mean(g) for g in train]),
        np.array([int(g.y) for g in train]))
    edge_pred = edgeless.predict(np.array([accounts_mean(g) for g in cut]))
    print(f"{DATASET}/{FEATURE} at {FRACTION:.0%}, {len(test)} test cascades")
    print(f"  edgeless          {f1_score(truth, edge_pred, average='macro'):.4f}\n")

    report = {"edgeless": round(float(f1_score(truth, edge_pred, average="macro")), 4)}
    for label, build in (("full", lambda d: list(d)), ("root masked", masked)):
        scores, pooled = [], []
        for seed in SEEDS:
            model = train_bigcn(_Wrapped(build(train), nf, nc), seed=seed)
            graph_pred = predict(model, [truncate(g, FRACTION) for g in build(test)])
            scores.append(f1_score(truth, graph_pred, average="macro"))
            pooled.append(paired_gap(truth, graph_pred, edge_pred, seed=seed))
        gaps = np.concatenate(pooled)
        low, high = np.percentile(gaps, [2.5, 97.5])
        excludes = bool(low > 0 or high < 0)
        report[label] = {
            "macro_f1": round(st.fmean(scores), 4),
            "seed_std": round(st.pstdev(scores), 4),
            "gap_vs_edgeless": round(float(gaps.mean()), 4),
            "gap_ci95": [round(float(low), 4), round(float(high), 4)],
            "excludes_zero": excludes,
            "p_ahead": round(float((gaps > 0).mean()), 3),
        }
        r = report[label]
        print(f"  Bi-GCN, {label:11s} {r['macro_f1']:.4f} (seed sd {r['seed_std']:.4f})")
        print(f"    gap vs edgeless {r['gap_vs_edgeless']:+.4f} "
              f"[{r['gap_ci95'][0]:+.4f}, {r['gap_ci95'][1]:+.4f}]  "
              f"P(ahead) {r['p_ahead']:.3f}  "
              f"{'SEPARATES' if excludes else 'does not separate'}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "early_gap_test.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
