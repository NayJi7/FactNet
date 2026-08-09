"""The object the engine returns and the interface draws.

Everything the dashboard shows about one prediction travels in a single
structure, so that the reasoning lives in Python where it can be tested and the
interface only renders. Adding a stage to the explanation is a new ``Step`` in
a list, not a change to the front end.

The whole structure is JSON-serialisable by construction: the API hands it over
untouched, and every figure carries its data rather than a rendered image, so
the browser can draw it interactively.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# What the front end knows how to draw. Keeping the set small and explicit is
# what stops the payload from turning into an untyped bag.
FIGURE_KINDS = (
    "tokens",     # a sequence of (token, weight): attention or attribution
    "bars",       # labelled magnitudes, signed
    "line",       # one or more series over an x axis
    "graph",      # nodes and links, drawn as a force layout
    "compare",    # several models scoring the same input
    "table",      # rows of plain values
)


@dataclass
class Figure:
    kind: str
    title: str
    data: dict[str, Any]
    caption: str = ""

    def __post_init__(self) -> None:
        if self.kind not in FIGURE_KINDS:
            raise ValueError(f"unknown figure kind: {self.kind}")


@dataclass
class Step:
    """One stage of the reasoning, with what it concluded and why."""

    key: str
    title: str
    summary: str                                   # one sentence, always present
    detail: dict[str, Any] = field(default_factory=dict)
    figures: list[Figure] = field(default_factory=list)
    status: str = "ok"                             # ok | skipped | warning
    note: str = ""                                 # a caveat worth reading


@dataclass
class Trace:
    """A prediction and the whole path that produced it."""

    input_kind: str                                # text | url | cascade | benchmark
    steps: list[Step] = field(default_factory=list)
    verdict: float | None = None                   # probability the item is reliable
    label: str = "unknown"                         # reliable | misleading | not a claim
    confidence: str = "low"                        # low | moderate | high
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def listen(self, callback) -> None:
        """Be told about each stage as it lands, for streaming to a browser.

        The listener is a plain attribute rather than a field so that it never
        reaches ``to_dict``: a callable has no place in a JSON payload.
        """
        object.__setattr__(self, "_listener", callback)

    def add(self, step: Step) -> Step:
        self.steps.append(step)
        listener = getattr(self, "_listener", None)
        if listener is not None:
            listener(step)
        return step

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> dict[str, Any]:
        """Everything but the stages, for the end of a stream that already sent them."""
        payload = asdict(self)
        payload.pop("steps")
        return payload


def confidence_from(probability: float, accounts: int = 0, has_features: bool = True) -> str:
    """How much weight the interface should invite the viewer to put on a verdict.

    Distance from the decision boundary is the main term, but it is not the only
    one: a verdict on a cascade of four accounts, or on one whose account
    features had to be imputed, deserves to be reported as weak however sharp
    the probability looks. Overstating certainty is the failure mode that costs
    most in a demonstration.
    """
    margin = abs(probability - 0.5)
    if margin < 0.08 or not has_features or (accounts and accounts < 10):
        return "low"
    if margin < 0.2:
        return "moderate"
    return "high"
