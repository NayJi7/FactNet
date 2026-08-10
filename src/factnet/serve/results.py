"""The article's measurements, served so the interface can show them beside a live run.

These are transcribed from the report rather than recomputed on request: they
are the numbers the supervisors read, and a dashboard that quietly displayed
slightly different ones would be worse than useless. Each table carries the
sentence that says what it means, because a table without its reading is an
invitation to read it wrong.

Anything measured live by the engine is computed by the engine; nothing here is
a prediction.
"""

from __future__ import annotations

from typing import Any

TABLES: list[dict[str, Any]] = [
    {
        "key": "graph",
        "title": "Propagation detection",
        "unit": "macro-F1",
        "columns": ["Dataset", "Features", "GCN", "GAT", "Bi-GCN"],
        "rows": [
            ["PolitiFact", "profile", 0.765, 0.764, 0.774],
            ["PolitiFact", "bert", 0.832, 0.832, 0.837],
            ["GossipCop", "profile", 0.884, 0.884, 0.873],
            ["GossipCop", "bert", 0.806, 0.900, 0.917],
        ],
        "best": "Bi-GCN",
        "reading": "The bidirectional design wins wherever the node features carry "
                   "text, and the margin over a plain graph convolution is small: "
                   "most of the signal is in the cascade, not in the architecture.",
    },
    {
        "key": "nlp",
        "title": "Content detection on LIAR",
        "unit": "macro-F1",
        "columns": ["Model", "Accuracy", "Macro-F1"],
        "rows": [
            ["TF-IDF + logistic regression", 0.637, 0.633],
            ["DistilBERT (frozen) + LR", 0.631, 0.626],
            ["BERT (fine-tuned)", 0.627, 0.616],
            ["RoBERTa (vanilla fine-tune)", 0.646, 0.632],
            ["RoBERTa (optimised fine-tune)", 0.636, 0.624],
        ],
        "best": None,
        "reading": "Five models within two points of each other, and a bag of words "
                   "at the top. The ceiling belongs to the task, not to model "
                   "capacity: on full article text the same approach reaches 0.822.",
    },
    {
        "key": "early",
        "title": "Early detection on truncated cascades",
        "unit": "macro-F1",
        "columns": ["Setting", "20%", "40%", "60%", "80%", "100%"],
        "rows": [
            ["PolitiFact, profile", 0.790, 0.803, 0.799, 0.796, 0.774],
            ["PolitiFact, bert", 0.837, 0.837, 0.837, 0.837, 0.837],
            ["GossipCop, bert", 0.882, 0.908, 0.916, 0.913, 0.911],
        ],
        "best": None,
        "reading": "The verdict is already there when a fifth of the diffusion has "
                   "happened, which is what makes acting on it conceivable. Seeing "
                   "the whole cascade adds nothing.",
    },
    {
        "key": "integration",
        "title": "Integration ablation",
        "unit": "macro-F1",
        "columns": ["Dataset", "Features", "Graph only", "+ random score",
                    "+ content score"],
        "rows": [
            ["PolitiFact", "profile", 0.787, 0.783, 0.843],
            ["PolitiFact", "bert", 0.828, 0.828, 0.831],
            ["GossipCop", "profile", 0.909, 0.909, 0.905],
            ["GossipCop", "bert", 0.914, 0.935, 0.929],
        ],
        "best": "+ content score",
        "reading": "The content score helps exactly where the cascade carries no "
                   "text of its own, and nothing where it already does. The random "
                   "column is the control: on GossipCop with bert features it moves "
                   "as much as the real score, so that block supports no conclusion.",
    },
    {
        "key": "ood",
        "title": "Cross-platform transfer to Bluesky",
        "unit": "macro-F1",
        "columns": ["Variant", "In-domain", "Bluesky", "Strict subset"],
        "rows": [
            ["With account features", 0.906, 0.380, 0.456],
            ["Structure only", 0.747, 0.306, 0.331],
        ],
        "best": None,
        "reading": "Not a degradation, a collapse: the predictions pile onto one "
                   "class and the score sits barely above what a constant answer "
                   "earns. It happens with or without account features, which is "
                   "what makes the failure structural rather than a scaling artefact.",
    },
    {
        "key": "adaptation",
        "title": "How much target data recovery needs",
        "unit": "macro-F1",
        "columns": ["Collected cascades used", "Fine-tuned from UPFD",
                    "Trained from scratch"],
        "rows": [
            [0, 0.419, None],
            [40, 0.424, 0.718],
            [100, 0.434, 0.760],
            [200, 0.455, 0.751],
        ],
        "best": "Trained from scratch",
        "reading": "Fine-tuning does not merely lag, it barely moves: four points "
                   "gained between no target data and two hundred cascades, while "
                   "training from scratch climbs past 0.75. The gap holds near 0.30 "
                   "at every budget instead of closing.",
    },
    {
        "key": "confound",
        "title": "What the collected labels are predictable from",
        "unit": "macro-F1",
        "columns": ["Condition", "One size threshold", "Bi-GCN"],
        "rows": [
            ["Split by cascade, all sizes", 0.902, 0.767],
            ["Split by domain, all sizes", 0.912, 0.720],
            ["Split by cascade, size matched", 0.504, 0.640],
            ["Split by domain, size matched", 0.309, 0.517],
        ],
        "best": None,
        "reading": "A rule with one parameter beats the detector: the credible "
                   "outlets we picked simply have larger audiences. Matching the "
                   "sizes away drops the rule to chance and leaves the detector "
                   "at 0.640, so a real but much smaller structural signal "
                   "survives. The last row rests on thirty to fifty test "
                   "cascades and is shown for completeness.",
    },
]

HEADLINES = [
    {"value": "0.917", "label": "propagation detection, GossipCop",
     "note": "Bi-GCN with text-derived node features"},
    {"value": "0.63", "label": "content ceiling on short claims",
     "note": "five models within two points, a bag of words at the top"},
    {"value": "+0.056", "label": "gain from reading both signals",
     "note": "where the cascade carries no text of its own"},
    {"value": "0.380", "label": "the same detector, one platform across",
     "note": "barely above a constant answer, not a degradation"},
]


def payload() -> dict[str, Any]:
    return {"headlines": HEADLINES, "tables": TABLES}
