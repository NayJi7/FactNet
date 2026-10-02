"""LIAR held-out topic test: train without topic X, eval in-domain vs on X.

    uv run python -m factnet.nlp.generalisation
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

from factnet.nlp.data import load_liar


def _fit(train):
    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    clf.fit(train.text, train.y)
    return clf


def cross_domain(topic: str = "health"):
    df = pd.concat([load_liar("train"), load_liar("test")], ignore_index=True)
    is_topic = df.subject.fillna("").str.contains(topic)
    source = df[~is_topic]
    out_of_topic = source.sample(frac=1.0, random_state=0)
    split = int(0.85 * len(out_of_topic))
    train, in_domain = out_of_topic.iloc[:split], out_of_topic.iloc[split:]
    cross = df[is_topic]                          # OOD

    clf = _fit(train)
    f1_in = f1_score(in_domain.y, clf.predict(in_domain.text), average="macro")
    f1_cross = f1_score(cross.y, clf.predict(cross.text), average="macro")
    return f1_in, f1_cross, len(train), len(cross)


TOPICS = ("health", "economy", "immigration", "elections", "taxes")


def main():
    print("Cross-domain generalisation  |  train outside the topic, "
          "test on the held-out topic\n")
    print(f"{'held-out topic':14s} {'in-domain':>10s} {'cross':>7s} "
          f"{'gap':>7s} {'n OOD':>6s}")
    print("-" * 48)
    for topic in TOPICS:
        f1_in, f1_cross, _n_train, n_cross = cross_domain(topic)
        print(f"{topic:14s} {f1_in:>10.3f} {f1_cross:>7.3f} "
              f"{f1_in - f1_cross:>+7.3f} {n_cross:>6d}")


if __name__ == "__main__":
    main()
