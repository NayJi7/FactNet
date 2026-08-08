"""Score a piece of text, and show what the score rests on.

Two explanation views accompany every transformer verdict, and they are not
redundant. Attention shows where the model looked, which is a description of
the computation and not, on its own, evidence of what mattered. Leave-one-out
occlusion removes each token in turn and measures how far the probability
moves, which is a direct causal statement about this input at the cost of one
forward pass per token. Reporting only the first would flatter the model;
reporting only the second would lose the picture the article shows.

Occlusion is preferred here over the sampling-based attribution the article
also reports, because it is exact and fast enough to stay interactive: a
twenty-token claim costs twenty forward passes, where a sampled method costs
hundreds for an approximation.
"""

from __future__ import annotations

import torch

from factnet.serve.registry import ModelCard, by_key, catalogue, load_transformer
from factnet.serve.trace import Figure

MAX_LEN = 64
SPECIAL = {"<s>", "</s>", "[CLS]", "[SEP]", "<pad>", "[PAD]"}


def _clean(token: str) -> str:
    """Undo the sub-word markers so the interface shows readable text."""
    return token.replace("Ġ", " ").replace("##", "").replace("▁", " ")


@torch.no_grad()
def _forward(model, encoded, want_attention: bool = False):
    return model(**encoded, output_attentions=want_attention)


def transformer_score(text: str, card: ModelCard) -> tuple[float, dict]:
    """Probability the text is reliable, with attention over its tokens."""
    tokenizer, model = load_transformer(card.path)
    encoded = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    out = _forward(model, encoded, want_attention=True)
    probability = torch.softmax(out.logits, dim=1)[0, 1].item()

    # last layer, averaged over heads, the row the classifier reads
    attention = out.attentions[-1][0].mean(dim=0)[0]
    ids = encoded["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    pairs = [(_clean(t), float(a)) for t, a in zip(tokens, attention, strict=False)
             if t not in SPECIAL]
    peak = max((a for _, a in pairs), default=1.0) or 1.0
    return probability, {"attention": [[t, round(a / peak, 4)] for t, a in pairs],
                         "n_tokens": len(pairs)}


@torch.no_grad()
def occlusion(text: str, card: ModelCard, baseline: float) -> list[list]:
    """Per-token effect on the verdict, measured by removing the token.

    A positive value means the token pushed the model towards *reliable*: the
    probability fell when it was taken away.
    """
    tokenizer, model = load_transformer(card.path)
    encoded = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    ids = encoded["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    keep = [i for i, t in enumerate(tokens) if t not in SPECIAL]
    if not keep:
        return []

    variants = []
    for index in keep:
        kept = [i for i in range(len(ids)) if i != index]
        variants.append(ids[kept])
    batch = torch.nn.utils.rnn.pad_sequence(
        variants, batch_first=True, padding_value=tokenizer.pad_token_id)
    mask = (batch != tokenizer.pad_token_id).long()
    probabilities = torch.softmax(model(input_ids=batch, attention_mask=mask).logits,
                                 dim=1)[:, 1]
    effects = [(_clean(tokens[i]), baseline - float(p))
               for i, p in zip(keep, probabilities, strict=False)]
    peak = max((abs(v) for _, v in effects), default=1.0) or 1.0
    return [[t, round(v / peak, 4)] for t, v in effects]


def linear_score(text: str, card: ModelCard) -> tuple[float, dict]:
    """The bag-of-words model, whose contributions need no approximation."""
    import joblib

    from factnet.serve.registry import MODELS
    pipeline = joblib.load(MODELS / card.path)
    probability = float(pipeline.predict_proba([text])[0, 1])

    vectoriser = pipeline.named_steps["tfidf"]
    weights = pipeline.named_steps["clf"].coef_[0]
    row = vectoriser.transform([text])
    names = vectoriser.get_feature_names_out()
    contributions = [(names[j], float(row[0, j] * weights[j])) for j in row.nonzero()[1]]
    contributions.sort(key=lambda kv: abs(kv[1]), reverse=True)
    top = contributions[:12]
    peak = max((abs(v) for _, v in top), default=1.0) or 1.0
    return probability, {"contributions": [[t, round(v / peak, 4)] for t, v in top]}


def score(text: str, model_key: str) -> tuple[float, dict]:
    card = by_key(model_key)
    if card.kind != "content":
        raise ValueError(f"{model_key} is not a content model")
    if card.path and card.path.endswith(".joblib"):
        return linear_score(text, card)
    return transformer_score(text, card)


def compare(text: str) -> tuple[Figure, list[str]]:
    """Every available content model on the same input.

    The disagreement is the point. Five models that a benchmark separates by
    two macro-F1 points will not agree on an arbitrary sentence, and seeing
    that is a more honest account of the content ceiling than any single
    number.
    """
    rows, missing = [], []
    for entry in catalogue("content"):
        if not entry["available"]:
            missing.append(entry["name"])
            continue
        try:
            probability, _ = score(text, entry["key"])
        except Exception:
            missing.append(entry["name"])
            continue
        rows.append({"model": entry["name"], "key": entry["key"],
                     "p_reliable": round(probability, 4),
                     "macro_f1": entry["macro_f1"], "primary": entry["primary"]})
    spread = (max(r["p_reliable"] for r in rows) - min(r["p_reliable"] for r in rows)
              if len(rows) > 1 else 0.0)
    figure = Figure(
        kind="compare", title="Every content model on this text",
        data={"rows": rows, "spread": round(spread, 4)},
        caption="Each verdict is shown beside the macro-F1 the model earned on the "
                "held-out benchmark, so a confident probability from a weak model "
                "cannot pass for a good one.")
    return figure, missing
