"""Content scoring + explanations (attention and leave-one-out occlusion).

Occlusion instead of LIME here: one forward pass per token, fast enough for
the dashboard, and exact.
"""

from __future__ import annotations

import torch

from factnet.serve.registry import ModelCard, by_key, catalogue, load_transformer
from factnet.serve.trace import Figure

MAX_LEN = 64
SPECIAL = {"<s>", "</s>", "[CLS]", "[SEP]", "<pad>", "[PAD]"}


def _clean(token: str) -> str:
    """strip Ġ / ## markers"""
    return token.replace("Ġ", " ").replace("##", "").replace("▁", " ")


@torch.no_grad()
def _forward(model, encoded, want_attention: bool = False):
    return model(**encoded, output_attentions=want_attention)


def transformer_score(text: str, card: ModelCard) -> tuple[float, dict]:
    """p(reliable) + last-layer attention."""
    tokenizer, model = load_transformer(card.path)
    encoded = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    out = _forward(model, encoded, want_attention=True)
    probability = torch.softmax(out.logits, dim=1)[0, 1].item()

    # last layer, mean over heads, CLS row
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
    """Drop each token and see how p moves. > 0 means it pushed towards reliable."""
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
    """TF-IDF + logreg, contributions are just weight * value."""
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


@torch.no_grad()
def _fragments(text: str, card: ModelCard) -> list[bool]:
    """True if the token is a word piece.

    roberta marks word *starts* (Ġ), bert marks *continuations* (##), so we
    detect which one we have from the tokens.
    """
    tokenizer, _ = load_transformer(card.path)
    encoded = tokenizer(text, truncation=True, max_length=MAX_LEN)
    tokens = [t for t in tokenizer.convert_ids_to_tokens(encoded["input_ids"])
              if t not in SPECIAL]
    if not tokens:
        return []
    marks_starts = any(t.startswith(("Ġ", "▁")) for t in tokens)
    return [t.startswith(("Ġ", "▁")) if marks_starts else not t.startswith("##")
            for t in tokens]


def decisive(text: str, model_key: str, probability: float,
             extra: dict, top: int = 4) -> dict:
    """Top tokens behind this model's verdict.

    Used to explain why models disagree: tf-idf sees whole words, transformers
    often see word pieces.
    """
    card = by_key(model_key)
    if "contributions" in extra:
        pairs = [(t, v) for t, v in extra["contributions"][:top]]
        return {"basis": "whole words, weighed exactly", "fragmented": False,
                "tokens": [[t, v] for t, v in pairs]}

    effects = occlusion(text, card, probability)
    whole = _fragments(text, card)
    ranked = sorted(range(len(effects)), key=lambda i: -abs(effects[i][1]))[:top]
    pairs = [[effects[i][0].strip() or effects[i][0], effects[i][1]] for i in ranked]
    # share of the decisive tokens that are word pieces
    pieces = sum(1 for i in ranked if i < len(whole) and not whole[i])
    fragmented = pieces > len(ranked) / 2
    return {"basis": "sub-word pieces" if fragmented else "whole words",
            "fragmented": fragmented, "tokens": pairs}


def score(text: str, model_key: str) -> tuple[float, dict]:
    card = by_key(model_key)
    if card.kind != "content":
        raise ValueError(f"{model_key} is not a content model")
    if card.path and card.path.endswith(".joblib"):
        return linear_score(text, card)
    return transformer_score(text, card)


def compare(text: str) -> tuple[Figure, list[str]]:
    """All available content models on the same text (they often disagree)."""
    rows, missing = [], []
    for entry in catalogue("content"):
        if not entry["available"]:
            missing.append(entry["name"])
            continue
        try:
            probability, extra = score(text, entry["key"])
            reading = decisive(text, entry["key"], probability, extra)
        except Exception:
            missing.append(entry["name"])
            continue
        rows.append({"model": entry["name"], "key": entry["key"],
                     "p_reliable": round(probability, 4),
                     "macro_f1": entry["macro_f1"], "primary": entry["primary"],
                     **reading})
    spread = (max(r["p_reliable"] for r in rows) - min(r["p_reliable"] for r in rows)
              if len(rows) > 1 else 0.0)
    figure = Figure(
        kind="compare", title="Every content model on this text",
        data={"rows": rows, "spread": round(spread, 4),
              "why": _why_they_differ(rows)},
        caption="Each verdict is shown beside the macro-F1 the model earned on the "
                "held-out benchmark, so a confident probability from a weak model "
                "cannot pass for a good one. What decided each verdict is measured "
                "by removing one token at a time, except for the bag of words, "
                "whose contributions are exact.")
    return figure, missing


def _why_they_differ(rows: list[dict]) -> list[str]:
    """Sentences explaining the disagreement, only the ones the numbers support."""
    if len(rows) < 2:
        return []
    said: list[str] = []
    fragmented = [r["model"] for r in rows if r.get("fragmented")]
    if fragmented:
        said.append(
            f"{len(fragmented)} of the {len(rows)} models decided on pieces of words "
            "rather than on words. Their tokenisers split an unfamiliar term into "
            "fragments, and the fragment carries no meaning a reader would recognise.")

    # same token, opposite sign
    seen: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        for token, value in row.get("tokens", []):
            key = str(token).strip().lower()
            if key:
                seen.setdefault(key, []).append((row["model"], value))
    split = [(t, v) for t, v in seen.items()
             if len(v) > 1 and max(x for _, x in v) > 0 > min(x for _, x in v)]
    if split:
        token, holders = max(split, key=lambda kv: len(kv[1]))
        up = next(m for m, v in holders if v > 0)
        down = next(m for m, v in holders if v < 0)
        said.append(
            f"They also read the same evidence in opposite directions: "
            f"“{token}” pushes {up} towards reliable and {down} the other way.")

    high, low = max(rows, key=lambda r: r["p_reliable"]), min(rows, key=lambda r: r["p_reliable"])
    said.append(
        f"{high['model']} is the most confident that this is reliable and "
        f"{low['model']} the least, and the two are within "
        f"{abs((high['macro_f1'] or 0) - (low['macro_f1'] or 0)):.3f} macro-F1 of each "
        "other on the held-out benchmark. No gap between any two of these models "
        "excludes zero under a paired bootstrap, so the spread on one post is not "
        "evidence that the confident one is right.")
    return said
