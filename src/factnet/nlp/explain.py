"""Explanations for TF-IDF + logreg: contribution = tfidf * coef, exact, no LIME needed.

Prints top cues per class and a few test statements. Transformer side is in
the kaggle notebook.

    uv run python -m factnet.nlp.explain
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from factnet.nlp.data import load_liar

TOP_GLOBAL = 12
TOP_LOCAL = 4


def fit(train):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000)
    x = vec.fit_transform(train.text)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x, train.y)
    return vec, clf


def global_cues(vec, clf, k: int = TOP_GLOBAL):
    """Strongest n-grams per direction (positive coef -> reliable)."""
    coef = clf.coef_[0]
    terms = np.array(vec.get_feature_names_out())
    order = np.argsort(coef)
    return terms[order[:k]], terms[order[-k:]][::-1]  # misleading, reliable


def explain_statement(vec, clf, text: str, k: int = TOP_LOCAL):
    """Exact per-term contributions for one statement."""
    row = vec.transform([text])
    contrib = row.multiply(clf.coef_[0]).toarray()[0]
    terms = vec.get_feature_names_out()
    idx = np.argsort(np.abs(contrib))[::-1]
    idx = [i for i in idx if contrib[i] != 0][:k]
    verdict = "reliable" if clf.predict(row)[0] == 1 else "misleading"
    return verdict, [(terms[i], float(contrib[i])) for i in idx]


def main():
    train, test = load_liar("train"), load_liar("test")
    vec, clf = fit(train)

    misleading, reliable = global_cues(vec, clf)
    print("Strongest global cues (TF-IDF + LogReg, exact coefficients)\n")
    print(f"{'toward misleading':30s} {'toward reliable':30s}")
    print("-" * 60)
    for m, r in zip(misleading, reliable, strict=True):
        print(f"{m:30s} {r:30s}")

    print("\nExample statements, decomposed")
    print("-" * 60)
    for text in test.text.iloc[[0, 3, 11]]:
        verdict, cues = explain_statement(vec, clf, text)
        pretty = ", ".join(f"{t} ({c:+.2f})" for t, c in cues)
        print(f"\n  \"{text[:90]}\"\n  verdict: {verdict}  |  cues: {pretty}")


if __name__ == "__main__":
    main()
