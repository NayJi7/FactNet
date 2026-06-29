"""Content-based verdict baselines on LIAR (reliable / misleading).

- Classical: TF-IDF + logistic regression (the transparent baseline to beat).
- Transformer (frozen): mean-pooled DistilBERT embeddings + logistic regression
  (a first transformer result without fine-tuning; the full fine-tuned RoBERTa
  runs on GPU/Colab later).

    uv run python -m factnet.nlp.baseline
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from factnet.nlp.data import load_liar


def tfidf_logreg(train, test):
    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    clf.fit(train.text, train.y)
    pred = clf.predict(test.text)
    return accuracy_score(test.y, pred), f1_score(test.y, pred, average="macro")


def embed(texts, model_name: str = "distilbert-base-uncased", batch: int = 32):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=64, return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)  # mean pooling
            chunks.append(pooled.numpy())
    return np.vstack(chunks)


def frozen_transformer(train, test):
    x_train = embed(train.text.tolist())
    x_test = embed(test.text.tolist())
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_train, train.y)
    pred = clf.predict(x_test)
    return accuracy_score(test.y, pred), f1_score(test.y, pred, average="macro")


def main():
    train, test = load_liar("train"), load_liar("test")
    print(f"LIAR binary  |  train {len(train)}  test {len(test)}  "
          f"(1 = reliable, 0 = misleading)\n")
    print(f"{'model':26s} {'accuracy':>9s} {'macro-F1':>9s}")
    print("-" * 46)
    acc, f1 = tfidf_logreg(train, test)
    print(f"{'TF-IDF + LogReg':26s} {acc:>9.3f} {f1:>9.3f}")
    acc, f1 = frozen_transformer(train, test)
    print(f"{'DistilBERT (frozen) + LR':26s} {acc:>9.3f} {f1:>9.3f}")


if __name__ == "__main__":
    main()
