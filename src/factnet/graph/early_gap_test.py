"""Does the graph model actually separate from an edgeless one at 20 per cent?

After the mean-pooling control, one cell carries every structural claim left in
this article: GossipCop ``bert`` at 20 per cent of the cascade, where the graph
model reads 0.903 and a logistic regression with no edges reads 0.885. That was
reported as a separation by putting a seed deviation of 0.009 next to a split
interval of 0.010 and observing that the numbers did not overlap. They very
nearly do, and in any case the two quantities measure different things: one is
variation across seeds on a fixed split, the other variation across splits at a
fixed fit. Setting them side by side is not a test.

The test is the one the content side of this project already uses. Both models
are scored on the same bootstrap resample of the test split, the difference is
recorded, and the interval is taken on that difference. Seed variance is folded
in by repeating the whole procedure for each of the graph model's seeds and
pooling. If the interval excludes zero the claim stands. If it does not, the
graph model is nowhere distinguishable from a model that cannot see an edge, and
that is the sentence this article has to carry.

The second question is why a gap would exist at 20 per cent and close later.
Truncation always keeps the root, and under ``bert`` the root holds the article
embedding while the edgeless baseline averages account features only. At 20 per
cent that average is at its noisiest and the root is intact, which is the shape
of an advantage that fades as the mean converges. Masking the root at the same
truncation level separates the two explanations.

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
    """macro-F1(a) - macro-F1(b) over resamples both models are scored on."""
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
