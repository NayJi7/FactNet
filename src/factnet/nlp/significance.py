"""Paired bootstrap on the LIAR test split (1267 items): are the content models
actually different? (0.647 vs 0.633 looks like a gap but probably isn't)

Same resample for both models so the sample noise cancels out. Doesn't cover
seed variance, so it's a lower bound on the uncertainty.

    uv run python -m factnet.nlp.significance
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

from factnet.nlp.data import load_liar

MODELS = Path(__file__).resolve().parents[3] / "models"
RESULTS = Path(__file__).resolve().parents[3] / "data" / "results"
CACHED = {
    "RoBERTa (fine-tuned)": "roberta-liar",
    "RoBERTa (optimised)": "roberta-opt-liar",
    "BERT (fine-tuned)": "bert-liar",
    "RoBERTa (weight decay)": "roberta-liar-wd",
}
ROUNDS = 5000
BASELINE = "TF-IDF + logistic regression"


def tfidf_predictions() -> tuple[list[int], list[int]]:
    """refit tfidf on the same split"""
    train, test = load_liar("train"), load_liar("test")
    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    clf.fit(train.text, train.y)
    return list(test.y), list(clf.predict(test.text))


def frozen_predictions() -> tuple[list[int], list[int]]:
    """Recompute frozen distilbert preds (the original run only saved metrics)."""
    from factnet.nlp.baseline import embed

    train, test = load_liar("train"), load_liar("test")
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(embed(list(train.text)), train.y)
    return list(test.y), list(model.predict(embed(list(test.text))))


def collect() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    truth, pred = tfidf_predictions()
    out[BASELINE] = (np.array(truth), np.array(pred))
    cache = MODELS / "distilbert-frozen-predictions.json"
    if cache.exists():
        blob = json.loads(cache.read_text())
    else:
        y, p = frozen_predictions()
        blob = {"y_true": [int(v) for v in y], "y_pred": [int(v) for v in p]}
        cache.write_text(json.dumps(blob) + "\n")
    out["DistilBERT (frozen) + LR"] = (np.array(blob["y_true"]), np.array(blob["y_pred"]))
    for name, folder in CACHED.items():
        path = MODELS / folder / "test_predictions.json"
        if not path.exists():
            continue
        blob = json.loads(path.read_text())
        out[name] = (np.array(blob["y_true"]), np.array(blob["y_pred"]))
    return out


def paired_bootstrap(truth, a, b, rounds=ROUNDS, seed=0):
    """macro-F1(a) - macro-F1(b) over resamples"""
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


def interval(truth, pred, rounds=ROUNDS, seed=0):
    rng = np.random.default_rng(seed)
    n = len(truth)
    scores = np.empty(rounds)
    for i in range(rounds):
        idx = rng.integers(0, n, n)
        scores[i] = (f1_score(truth[idx], pred[idx], average="macro")
                     if len(np.unique(truth[idx])) > 1 else np.nan)
    return np.nanpercentile(scores, [2.5, 97.5])


def main() -> None:
    models = collect()
    truth = models[BASELINE][0]
    print(f"LIAR test split: {len(truth)} items, {ROUNDS} bootstrap resamples\n")

    print(f"{'model':30s} {'macro-F1':>9s} {'95% interval':>20s}")
    print("-" * 62)
    table = {}
    for name, (y, pred) in models.items():
        point = f1_score(y, pred, average="macro")
        low, high = interval(y, pred)
        table[name] = {"macro_f1": round(point, 4),
                       "ci95": [round(low, 4), round(high, 4)]}
        print(f"{name:30s} {point:>9.3f} {f'[{low:.3f}, {high:.3f}]':>20s}")

    print(f"\nPaired against {BASELINE}, positive means the model is ahead\n")
    print(f"{'model':30s} {'gap':>8s} {'95% interval':>20s} {'P(ahead)':>10s}")
    print("-" * 72)
    base = models[BASELINE][1]
    for name, (_truth, pred) in models.items():
        if name == BASELINE:
            continue
        gaps = paired_bootstrap(truth, pred, base)
        low, high = np.percentile(gaps, [2.5, 97.5])
        ahead = float((gaps > 0).mean())
        table[name]["gap_vs_tfidf"] = {
            "point": round(float(gaps.mean()), 4),
            "ci95": [round(low, 4), round(high, 4)],
            "p_ahead": round(ahead, 3),
        }
        print(f"{name:30s} {gaps.mean():>+8.3f} "
              f"{f'[{low:+.3f}, {high:+.3f}]':>20s} {ahead:>10.3f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "liar_significance.json"
    path.write_text(json.dumps(table, indent=2) + "\n")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
