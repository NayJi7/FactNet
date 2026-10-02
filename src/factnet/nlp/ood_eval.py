"""LIAR content models on our Bluesky posts.

Not really the same task: LIAR labels the claim, here the label is the outlet's
rating. So we expect it to fail, the point is how. Steps:
  1. how many posts are even claims (check-worthiness)
  2. LIAR models as-is, do they all predict one class?
  3. train/test on bluesky, split by domain (by post = memorising outlets)
  4. same with links stripped, posts often contain the url ("rt.com"...)

    uv run python -u -m factnet.nlp.ood_eval
"""

from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

from factnet.nlp import checkworthy
from factnet.serve.pipeline import CHECKWORTHY_FLOOR

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "data" / "results"
COLLECTED = ROOT / "data" / "raw" / "bluesky" / "cascades-by-source.jsonl"
SEEDS = (0, 1, 2)

# strip urls/domains from the text, they leak the label
URLISH = re.compile(
    r"https?://\S+"
    r"|\b[\w-]+\.(?:com|org|net|co|uk|news|tv|info|us|ru)\b\S*"
    r"|\bwww\.\S+",
    re.IGNORECASE,
)


def load() -> list[dict]:
    with COLLECTED.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [r for r in rows
            if r.get("label") is not None and (r.get("text") or "").strip()]


def strip_links(text: str) -> str:
    return URLISH.sub(" ", text).strip()


# 1. claims?
def claim_rate(rows: list[dict]) -> dict:
    scores = [checkworthy.score(r["text"]) for r in rows]
    kept = [s for s in scores if s >= CHECKWORTHY_FLOOR]
    by_class = {}
    for label in (0, 1):
        side = [checkworthy.score(r["text"]) for r in rows if r["label"] == label]
        by_class[label] = sum(1 for s in side if s >= CHECKWORTHY_FLOOR) / max(1, len(side))
    return {"n": len(rows), "checkable": len(kept), "rate": len(kept) / max(1, len(rows)),
            "mean_score": st.mean(scores) if scores else 0.0,
            "rate_misleading": by_class[0], "rate_reliable": by_class[1]}


# 2. transfer
def transfer(rows: list[dict]) -> list[dict]:
    from factnet.serve.content import linear_score, transformer_score
    from factnet.serve.registry import by_key, catalogue

    texts = [r["text"] for r in rows]
    truth = [r["label"] for r in rows]
    out = []
    for entry in catalogue("content"):
        if not entry["available"]:
            continue
        card = by_key(entry["key"])
        probabilities = []
        for text in texts:
            score = (linear_score if card.path.endswith(".joblib") else transformer_score)
            probabilities.append(score(text, card)[0])
        predicted = [1 if p >= 0.5 else 0 for p in probabilities]
        share_reliable = sum(predicted) / len(predicted)
        low, high = _bootstrap(truth, predicted)
        out.append({
            "model": card.name,
            "liar_macro_f1": card.macro_f1,
            "macro_f1": f1_score(truth, predicted, average="macro"),
            "ci95": [round(low, 4), round(high, 4)],
            "above_chance": bool(low > 0.5),
            "predicted_reliable": share_reliable,
            "mean_p": st.mean(probabilities),
            "spread": max(probabilities) - min(probabilities),
        })
    return out


def _bootstrap(truth, predicted, rounds: int = 5000, seed: int = 0):
    """bootstrap CI, only 398 posts so 0.59 vs 0.50 isn't obvious"""
    import numpy as np

    rng = np.random.default_rng(seed)
    y, p = np.array(truth), np.array(predicted)
    n = len(y)
    scores = np.empty(rounds)
    for i in range(rounds):
        idx = rng.integers(0, n, n)
        scores[i] = (f1_score(y[idx], p[idx], average="macro")
                     if len(np.unique(y[idx])) > 1 else np.nan)
    return tuple(np.nanpercentile(scores, [2.5, 97.5]))


# 3 + 4. in domain
def _fit_eval(train: list[dict], test: list[dict], field: str) -> float:
    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    clf.fit([r[field] for r in train], [r["label"] for r in train])
    predicted = clf.predict([r[field] for r in test])
    return f1_score([r["label"] for r in test], predicted, average="macro")


def _domain_split(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """split by domain"""
    import random

    domains = sorted({r.get("source_domain", "") for r in rows})
    rng = random.Random(seed)
    rng.shuffle(domains)
    held = set(domains[: max(1, len(domains) // 3)])
    return ([r for r in rows if r.get("source_domain") not in held],
            [r for r in rows if r.get("source_domain") in held])


def _post_split(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    import random

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * 0.7)
    return shuffled[:cut], shuffled[cut:]


def in_domain(rows: list[dict]) -> list[dict]:
    """{post split, domain split} x {with links, links stripped}"""
    for row in rows:
        row["raw"] = row["text"]
        row["stripped"] = strip_links(row["text"])

    conditions = []
    for split_name, split in (("by post", _post_split), ("by domain", _domain_split)):
        for field, field_name in (("raw", "text as posted"), ("stripped", "links removed")):
            scores = []
            for seed in SEEDS:
                train, test = split(rows, seed)
                if not train or not test:
                    continue
                scores.append(_fit_eval(train, test, field))
            conditions.append({"split": split_name, "text": field_name,
                               "macro_f1": st.mean(scores) if scores else float("nan"),
                               "sd": st.pstdev(scores) if len(scores) > 1 else 0.0})
    return conditions


def main() -> None:
    rows = load()
    print(f"Content models on collected Bluesky posts | {len(rows)} labelled posts, "
          f"{len({r.get('source_domain') for r in rows})} domains\n")

    print("1. Do these posts state checkable claims?")
    claims = claim_rate(rows)
    print(f"   above the {CHECKWORTHY_FLOOR} floor : {claims['checkable']}/{claims['n']}"
          f"  ({claims['rate']:.1%})")
    print(f"   mean check-worthiness   : {claims['mean_score']:.3f}")
    print(f"   by class                : misleading {claims['rate_misleading']:.1%}"
          f"  vs reliable {claims['rate_reliable']:.1%}\n")

    print("2. LIAR-trained models, applied unchanged")
    print(f"   {'model':<32}{'LIAR':>6}{'here':>7}{'95% interval':>18}"
          f"{'>chance':>9}{'said rel.':>11}")
    payload = transfer(rows)
    for row in payload:
        liar = f"{row['liar_macro_f1']:.3f}" if row["liar_macro_f1"] else "  -  "
        low, high = row["ci95"]
        span = f"[{low:.3f}, {high:.3f}]"
        print(f"   {row['model']:<32}{liar:>6}{row['macro_f1']:>7.3f}{span:>18}"
              f"{('yes' if row['above_chance'] else 'no'):>9}"
              f"{row['predicted_reliable']:>11.1%}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "nlp_transfer.json").write_text(json.dumps(payload, indent=2) + "\n")

    print("\n3 and 4. Trained here instead, and what the signal turns out to be")
    print(f"   {'split':<12}{'text':<18}{'macro-F1':>10}")
    for row in in_domain(rows):
        print(f"   {row['split']:<12}{row['text']:<18}{row['macro_f1']:>10.3f}"
              f"  (sd {row['sd']:.3f})")


if __name__ == "__main__":
    main()
