"""Trace = what the API returns: a list of Steps + the final verdict. JSON-safe."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# figure kinds the front knows how to draw
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


# which module a step comes from (one paper each)
MODULES = {
    "content": "Content",                          # the NLP module, on the text
    "propagation": "Propagation",                  # the graph module, on the cascade
    "both": "Both",                                # where the two meet
}


@dataclass
class Step:

    key: str
    title: str
    summary: str                                   # one line
    detail: dict[str, Any] = field(default_factory=dict)
    figures: list[Figure] = field(default_factory=list)
    status: str = "ok"                             # ok | skipped | warning
    note: str = ""                                 # caveat
    module: str = ""                               # content | propagation | both

    def __post_init__(self) -> None:
        if self.module and self.module not in MODULES:
            raise ValueError(f"unknown module: {self.module}")


@dataclass
class Trace:

    input_kind: str                                # text | url | cascade | benchmark
    steps: list[Step] = field(default_factory=list)
    verdict: float | None = None                   # p(reliable)
    label: str = "unknown"                         # reliable | misleading | not a claim
    confidence: str = "low"                        # low | moderate | high
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def listen(self, callback) -> None:
        """Callback on each new step (streaming). Not a field so to_dict skips it."""
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
        """to_dict() without the steps (they were already streamed)."""
        payload = asdict(self)
        payload.pop("steps")
        return payload


def confidence_from(probability: float, accounts: int = 0, has_features: bool = True) -> str:
    """low / moderate / high. Mostly |p - 0.5|, but tiny cascades or missing
    features cap it at low whatever p is."""
    margin = abs(probability - 0.5)
    if margin < 0.08 or not has_features or (accounts and accounts < 10):
        return "low"
    if margin < 0.2:
        return "moderate"
    return "high"
