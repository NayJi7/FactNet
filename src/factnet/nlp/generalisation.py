"""Cross-domain generalisation test on LIAR (honest generalisation gap).

Trains on statements outside a topic and reports macro-F1 in-domain vs on the
held-out topic, exposing the drop when the subject shifts.

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
    source = df[~is_topic]                       # everything outside the topic
    out_of_topic = source.sample(frac=1.0, random_state=0)
    split = int(0.85 * len(out_of_topic))
    train, in_domain = out_of_topic.iloc[:split], out_of_topic.iloc[split:]
    cross = df[is_topic]                          # the held-out topic (OOD)

    clf = _fit(train)
    f1_in = f1_score(in_domain.y, clf.predict(in_domain.text), average="macro")
    f1_cross = f1_score(cross.y, clf.predict(cross.text), average="macro")
    return f1_in, f1_cross, len(train), len(cross)


def main():
    topic = "health"
    f1_in, f1_cross, n_train, n_cross = cross_domain(topic)
    print(f"Cross-domain generalisation  |  held-out topic: '{topic}'  "
          f"(train {n_train}, OOD {n_cross})\n")
    print(f"  in-domain  macro-F1 : {f1_in:.3f}")
    print(f"  cross-topic macro-F1: {f1_cross:.3f}")
    print(f"  generalisation gap  : {f1_in - f1_cross:+.3f}")


if __name__ == "__main__":
    main()
