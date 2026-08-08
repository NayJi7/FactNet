"""What models exist, what they are worth, and how to load them.

The dashboard lets a viewer switch between every model the article reports, and
that only means something if each verdict is shown next to the score the model
actually earned on a held-out benchmark. A confident-looking probability from
the weakest model is otherwise indistinguishable from a good one.

Loading is lazy and cached: a transformer costs half a gigabyte, and a
demonstration should not pay for the four that were not asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

MODELS = Path(__file__).resolve().parents[3] / "models"


@dataclass(frozen=True)
class ModelCard:
    """A model and the claim that can honestly be made about it."""

    key: str
    name: str
    kind: Literal["content", "graph"]
    macro_f1: float | None            # on its own benchmark, from the article
    trained_on: str                   # the surface it actually saw
    note: str = ""
    path: str | None = None           # relative to models/, when persisted
    primary: bool = False             # the verdict of record
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        if self.path is None:
            return True               # trained on the fly, in seconds
        target = MODELS / self.path
        return target.exists() and any(target.iterdir()) if target.is_dir() else target.exists()


CONTENT_MODELS = (
    ModelCard(
        key="roberta", name="RoBERTa (fine-tuned)", kind="content", macro_f1=0.632,
        trained_on="LIAR, short political claims", path="roberta-liar", primary=True,
        note="The content model of record in the article: the best of the three "
             "transformer runs, and the one the reported figures describe.",
    ),
    ModelCard(
        key="roberta-opt", name="RoBERTa (optimised)", kind="content", macro_f1=0.624,
        trained_on="LIAR, short political claims", path="roberta-opt-liar",
        note="Longer inputs, warmup and weight decay. It wins on validation and "
             "loses on test, which is the generalisation lesson of the project.",
    ),
    ModelCard(
        key="bert", name="BERT (fine-tuned)", kind="content", macro_f1=0.616,
        trained_on="LIAR, short political claims", path="bert-liar",
        note="The transformer baseline, kept to show the architecture gain is small.",
    ),
    ModelCard(
        key="tfidf", name="TF-IDF + logistic regression", kind="content", macro_f1=0.633,
        trained_on="LIAR, short political claims", path="tfidf.joblib",
        note="Trains in seconds and edges out every transformer here. The clearest "
             "evidence that the ceiling belongs to the task, not to model capacity.",
    ),
    ModelCard(
        key="roberta-wd", name="RoBERTa (weight decay)", kind="content", macro_f1=0.647,
        trained_on="LIAR, short political claims", path="roberta-liar-wd",
        note="Not a configuration the article reports. One seed above the others, "
             "an interval that run-to-run variance alone can produce, so it is "
             "offered for comparison and no conclusion is drawn from it.",
    ),
)

GRAPH_MODELS = (
    ModelCard(
        key="bigcn-upfd-profile", name="Bi-GCN, benchmark (profile)", kind="graph",
        macro_f1=0.787, trained_on="UPFD PolitiFact, account features",
        path="graph/bigcn-upfd-profile.pt", primary=True,
        note="The propagation model of record for cascades that carry account features.",
    ),
    ModelCard(
        key="bigcn-upfd-profile-score", name="Bi-GCN + content score", kind="graph",
        macro_f1=0.843, trained_on="UPFD PolitiFact, account features plus a score column",
        path="graph/bigcn-upfd-profile-score.pt",
        note="The integrated model: the ablation that answers RQ4, reproduced live.",
    ),
    ModelCard(
        key="bigcn-collected", name="Bi-GCN, trained on Bluesky", kind="graph",
        macro_f1=0.787, trained_on="the 400 collected cascades",
        path="graph/bigcn-collected.pt",
        note="The model to apply to a real Bluesky cascade. Trained on the target "
             "platform because the benchmark model does not survive the move.",
    ),
    ModelCard(
        key="bigcn-structure", name="Bi-GCN, structure only", kind="graph",
        macro_f1=None, trained_on="UPFD PolitiFact, features replaced by a constant",
        path="graph/bigcn-structure.pt",
        note="Reads nothing but the shape of the cascade. The honest test of the "
             "structural claim, and the fallback when account features are missing.",
    ),
    ModelCard(
        key="gcn-upfd-profile", name="GCN, benchmark", kind="graph", macro_f1=0.765,
        trained_on="UPFD PolitiFact, account features", path="graph/gcn-upfd-profile.pt",
        note="Plain graph convolution, kept to show what the bidirectional design buys.",
    ),
    ModelCard(
        key="gat-upfd-profile", name="GAT, benchmark", kind="graph", macro_f1=0.765,
        trained_on="UPFD PolitiFact, account features", path="graph/gat-upfd-profile.pt",
        note="Attention over neighbours, same input as the GCN.",
    ),
)

ALL = CONTENT_MODELS + GRAPH_MODELS


def by_key(key: str) -> ModelCard:
    for card in ALL:
        if card.key == key:
            return card
    raise KeyError(f"unknown model: {key}")


def primary(kind: str) -> ModelCard:
    for card in ALL:
        if card.kind == kind and card.primary:
            return card
    raise KeyError(f"no primary model for {kind}")


def catalogue(kind: str | None = None, only_available: bool = False) -> list[dict[str, Any]]:
    """The model list the interface renders in its selector."""
    cards = [c for c in ALL if kind is None or c.kind == kind]
    if only_available:
        cards = [c for c in cards if c.available]
    return [{"key": c.key, "name": c.name, "kind": c.kind, "macro_f1": c.macro_f1,
             "trained_on": c.trained_on, "note": c.note, "primary": c.primary,
             "available": c.available} for c in cards]


@lru_cache(maxsize=8)
def load_transformer(path: str):
    """A fine-tuned sequence classifier, with attention left reachable.

    The default attention path returns ``None`` for the weights, which would
    silently empty the explanation, so the eager implementation is requested.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    full = str(MODELS / path)
    tokenizer = AutoTokenizer.from_pretrained(full)
    model = AutoModelForSequenceClassification.from_pretrained(
        full, attn_implementation="eager")
    model.eval()
    return tokenizer, model
