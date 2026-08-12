"""The paired test applied to every row it should have been applied to.

Table 2 of the graph article asserts three separations between the detector and
an edgeless baseline, and the article's abstract opens on one of them. All three
were read off by putting a mean beside an interval, which is the procedure the
early-detection section retracts two pages later. This runs the retraction's own
test on all four rows.

For each configuration the graph model is trained on three seeds, the edgeless
baseline is fitted once, and both are scored on the same bootstrap resample of
the test split. The interval is taken on the difference, and the whole procedure
is repeated per seed so that seed variance and split variance enter together.

The second question is the same mechanism at the other end of the sweep. Masking
the article embedding out of the root is worth +0.022 to the graph model at
20 per cent of a GossipCop cascade. Whether it is worth anything at 100 per cent
decides the article's most quotable line, since the edgeless baseline leads there
by 0.019.

    uv run python -m factnet.graph.table2_gap_test
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch_geometric.datasets import UPFD

from factnet.graph.early_detection import FRACTIONS, train_bigcn, truncate
from factnet.graph.early_gap_test import masked, paired_gap, predict
from factnet.graph.root_ablation import _Wrapped
from factnet.graph.train_upfd import ROOT
from factnet.graph.trivial_baselines import design

RESULTS = Path(__file__).resolve().parents[3] / "data" / "results"
SETUPS = [("politifact", "profile"), ("politifact", "bert"),
          ("gossipcop", "profile"), ("gossipcop", "bert")]
SEEDS = (0, 1, 2)
BASELINES = ("root", "users", "mean-pool")


def best_baseline(train, test, truth) -> tuple[str, np.ndarray, float]:
    """The strongest edgeless model on this row, which is what to beat."""
    best = (None, None, -1.0)
    for kind in BASELINES:
        Xtr, ytr = design(train, kind)
        Xte, _ = design(test, kind)
        pred = LogisticRegression(max_iter=5000, class_weight="balanced").fit(
            Xtr, ytr).predict(Xte)
        score = f1_score(truth, pred, average="macro")
        if score > best[2]:
            best = (kind, pred, score)
    return best


def row(name: str, feature: str) -> dict:
    train = UPFD(ROOT, name, feature, split="train")
    test = UPFD(ROOT, name, feature, split="test")
    truth = np.array([int(g.y) for g in test])
    kind, edge_pred, edge_f1 = best_baseline(train, test, truth)

    scores, pooled = [], []
    for seed in SEEDS:
        model = train_bigcn(_Wrapped(list(train), train.num_features,
                                     train.num_classes), seed=seed)
        graph_pred = predict(model, list(test))
        scores.append(f1_score(truth, graph_pred, average="macro"))
        pooled.append(paired_gap(truth, graph_pred, edge_pred, seed=seed))

    gaps = np.concatenate(pooled)
    low, high = np.percentile(gaps, [2.5, 97.5])
    return {"baseline": kind, "baseline_f1": round(float(edge_f1), 4),
            "graph_f1": round(st.fmean(scores), 4),
            "graph_seed_std": round(st.pstdev(scores), 4),
            "gap": round(float(gaps.mean()), 4),
            "ci95": [round(float(low), 4), round(float(high), 4)],
            "separates": bool(low > 0 or high < 0)}


def masked_sweep() -> dict:
    """The root-masking gain at every truncation level, not just at 20 per cent."""
    train = UPFD(ROOT, "gossipcop", "bert", split="train")
    test = UPFD(ROOT, "gossipcop", "bert", split="test")
    nf, nc = train.num_features, train.num_classes
    truth = np.array([int(g.y) for g in test])
    out = {}
    for label, build in (("full", lambda d: list(d)), ("masked", masked)):
        per = {f"{f:.0%}": [] for f in FRACTIONS}
        for seed in SEEDS:
            model = train_bigcn(_Wrapped(build(train), nf, nc), seed=seed)
            cut = build(test)
            for f in FRACTIONS:
                pred = predict(model, [truncate(g, f) for g in cut])
                per[f"{f:.0%}"].append(f1_score(truth, pred, average="macro"))
        out[label] = {k: {"mean": round(st.fmean(v), 4),
                          "std": round(st.pstdev(v), 4)} for k, v in per.items()}
    return out


def main() -> None:
    report = {"table2": {}}
    print("Paired bootstrap, graph model against the best edgeless baseline\n")
    print(f"{'configuration':22s} {'baseline':11s} {'edgeless':>9s} {'graph':>8s} "
          f"{'gap':>8s} {'95% interval':>20s}  verdict")
    for name, feature in SETUPS:
        r = row(name, feature)
        report["table2"][f"{name}/{feature}"] = r
        span = f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"
        print(f"{name + '/' + feature:22s} {r['baseline']:11s} {r['baseline_f1']:>9.3f} "
              f"{r['graph_f1']:>8.3f} {r['gap']:>+8.3f} {span:>20s}  "
              f"{'SEPARATES' if r['separates'] else 'no'}", flush=True)

    print("\nRoot masking across the whole sweep, GossipCop bert")
    sweep = masked_sweep()
    report["masked_sweep"] = sweep
    print(f"{'level':8s} {'full':>16s} {'masked':>16s} {'gain':>9s}")
    for f in FRACTIONS:
        k = f"{f:.0%}"
        a, b = sweep["full"][k], sweep["masked"][k]
        print(f"{k:8s} {a['mean']:.4f} ±{a['std']:.4f}  {b['mean']:.4f} ±{b['std']:.4f} "
              f"{b['mean'] - a['mean']:>+9.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "table2_gap_test.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
