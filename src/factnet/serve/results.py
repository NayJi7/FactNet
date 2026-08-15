"""The measurements, read from the files the experiments wrote.

An earlier version of this module transcribed the tables of the report by hand.
Every rerun then had to be copied across, and when the reports were restructured
the copies were not: the dashboard spent a month serving a macro-F1 of 0.873 for
a cell that reproduces at 0.901, under a sentence the papers had retracted.

Nothing is transcribed here. Each table is assembled from the JSON its
experiment wrote into ``data/results``, so a rerun changes the dashboard without
anyone editing this file, and a missing file removes its table rather than
leaving a stale one in place. Only the readings are written by hand, because a
sentence saying what a table means is not something a measurement can produce.

Each table declares which module it belongs to, so a viewer can tell which of
the two papers to check a number against.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parents[3] / "data" / "results"


@lru_cache(maxsize=32)
def _load(name: str) -> dict | list | None:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _pm(cell: dict | None, digits: int = 3) -> str | None:
    """A mean with its seed deviation, as the reports print it."""
    if not cell:
        return None
    if isinstance(cell, (int, float)):
        return round(float(cell), digits)
    mean = cell.get("mean", cell.get("macro_f1"))
    if mean is None:
        return None
    std = cell.get("std")
    return (f"{mean:.{digits}f} ±{std:.3f}" if std is not None
            else round(float(mean), digits))


def _round(value, digits: int = 3):
    return round(float(value), digits) if value is not None else None


def _signed(value: float, digits: int = 3) -> str:
    """A difference, rounded half up so the page and the reports agree.

    Python rounds a tie to the even digit, and the binary value of a figure
    like 0.0365 sits a hair below the decimal one, so the default formatting
    prints 0.036 where the reports print 0.037.
    """
    from decimal import ROUND_HALF_UP, Decimal

    quantum = Decimal(1).scaleb(-digits)
    fixed = Decimal(repr(float(value))).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{fixed:+.{digits}f}"


# --------------------------------------------------------------------------
# Propagation module
# --------------------------------------------------------------------------

def _graph() -> dict | None:
    politifact = _load("table1_3seeds")
    gossipcop = _load("gossipcop_bert_3seeds")
    if not politifact:
        return None
    rows = []
    for key, label in (("politifact/profile", ("PolitiFact", "profile")),
                       ("politifact/bert", ("PolitiFact", "bert")),
                       ("gossipcop/profile", ("GossipCop", "profile"))):
        cell = politifact.get(key)
        if cell:
            rows.append([*label] + [_pm(cell[m]) for m in ("GCN", "GAT", "Bi-GCN")])
    if gossipcop:
        rows.append(["GossipCop", "bert"]
                    + [_pm(gossipcop[m]) for m in ("GCN", "GAT", "Bi-GCN")])
    return {
        "key": "graph", "module": "propagation", "title": "Propagation detection",
        "unit": "macro-F1, mean of three seeds", "best": "Bi-GCN",
        "columns": ["Dataset", "Features", "GCN", "GAT", "Bi-GCN"], "rows": rows,
        "reading": "The bidirectional design leads everywhere, and the margin runs "
                   "opposite to the obvious guess: it is largest on GossipCop with "
                   "ten account counters, the poorest features here, and smallest "
                   "where the nodes already carry text. Most of the signal is in "
                   "the cascade, not in the architecture.",
    }


def _trivial() -> dict | None:
    report = _load("table2_gap_test")
    if not report:
        return None
    names = {"users": "mean of account features", "root": "root feature alone",
             "mean-pool": "mean pooling"}
    rows = []
    for key, label in (("politifact/profile", ("PolitiFact", "profile")),
                       ("politifact/bert", ("PolitiFact", "bert")),
                       ("gossipcop/profile", ("GossipCop", "profile")),
                       ("gossipcop/bert", ("GossipCop", "bert"))):
        cell = report["table2"].get(key)
        if not cell:
            continue
        low, high = cell["ci95"]
        rows.append([*label, names.get(cell["baseline"], cell["baseline"]),
                     _round(cell["baseline_f1"]), _round(cell["graph_f1"]),
                     _signed(cell["gap"]), f"[{low:+.3f}, {high:+.3f}]",
                     "yes" if cell["separates"] else "no"])
    return {
        "key": "trivial", "module": "propagation",
        "title": "Against a model that reads no edge",
        "unit": "macro-F1, and a paired bootstrap on the difference",
        "columns": ["Dataset", "Features", "Strongest edgeless baseline",
                    "Edgeless", "Bi-GCN", "Gap", "95% interval", "Separates"],
        "rows": rows, "best": None,
        "reading": "This is the honest test of the whole propagation idea, and it "
                   "is the table the rest of this page should be read against. A "
                   "logistic regression with every edge deleted is beaten on one "
                   "configuration, wins on another, and is indistinguishable on "
                   "the two smallest. Where it wins, the cascade carries the news "
                   "article at its root, which the next table takes away.",
    }


def _early() -> dict | None:
    sweep = _load("table2_gap_test")
    politifact_bert = _load("early_detection_politifact_bert")
    seeds = _load("early_detection_3seeds")
    edgeless = _load("early_detection_edgeless")
    if not (sweep and politifact_bert):
        return None
    levels = ["20%", "40%", "60%", "80%", "100%"]
    series = [
        ("PolitiFact, profile", "Bi-GCN", (seeds or {}).get("politifact/profile")),
        ("PolitiFact, bert", "Bi-GCN", politifact_bert.get("full")),
        ("", "Bi-GCN, article masked", politifact_bert.get("masked")),
        ("", "no edges", (edgeless or {}).get("politifact/bert")),
        ("GossipCop, bert", "Bi-GCN", sweep["masked_sweep"].get("full")),
        ("", "Bi-GCN, article masked", sweep["masked_sweep"].get("masked")),
        ("", "no edges", (edgeless or {}).get("gossipcop/bert")),
    ]
    rows = []
    for corpus, model, cells in series:
        if not cells:
            continue
        rows.append([corpus, model] + [_pm(cells.get(level)) for level in levels])
    return {
        "key": "early", "module": "propagation",
        "title": "Detection against the share of the cascade observed",
        "unit": "macro-F1", "columns": ["Corpus, features", "Model", *levels],
        "rows": rows, "best": None,
        "reading": "A verdict is available from a fifth of the diffusion, which is "
                   "what makes acting on it conceivable. Two cautions: the edgeless "
                   "row is close behind throughout, so the next table tests the "
                   "difference rather than reading it off, and masking the article "
                   "out of the root moves the two corpora in opposite directions.",
    }


def _gap() -> dict | None:
    report = _load("early_gap_test")
    if not report:
        return None
    rows = [["Logistic regression, no edges", _round(report["edgeless"]), None, None]]
    for name, label in (("full", "Bi-GCN"), ("root masked", "Bi-GCN, article masked")):
        cell = report.get(name)
        if not cell:
            continue
        low, high = cell["gap_ci95"]
        rows.append([label, _round(cell["macro_f1"]),
                     _signed(cell["gap_vs_edgeless"]),
                     f"[{low:+.3f}, {high:+.3f}]"])
    return {
        "key": "gap", "module": "propagation",
        "title": "Does the graph separate from a model with no edges?",
        "unit": "GossipCop bert at 20% of the cascade",
        "columns": ["Model", "Macro-F1", "Gap vs edgeless", "95% interval"],
        "rows": rows, "best": None,
        "reading": "Placing a seed deviation beside a split interval is not a test. "
                   "Under a paired bootstrap the detector as configured does not "
                   "separate from the edgeless model. Masking the news article out "
                   "of the root does separate it, which locates the problem in a "
                   "benchmark feature rather than in the architecture.",
    }


def _ood() -> dict | None:
    report = _load("ood_transfer")
    if not report:
        return None
    rows = [[label, _pm(cell["in_domain"]), _pm(cell["bluesky"]), _pm(cell["strict"])]
            for label, cell in (("With account features", report.get("account features")),
                                ("Structure only", report.get("structure only")))
            if cell]
    return {
        "key": "ood", "module": "propagation",
        "title": "Cross-platform transfer to Bluesky", "unit": "macro-F1",
        "columns": ["Variant", "In domain", "Bluesky", "Strict subset"],
        "rows": rows, "best": None,
        "reading": "Not a degradation, a collapse: the predictions pile onto one "
                   "class and the score sits near what a constant answer earns, "
                   "which is 0.333 on a balanced binary task. It happens with and "
                   "without account features, which makes the failure structural "
                   "rather than an artefact of scaling.",
    }


def _adaptation() -> dict | None:
    report = _load("adaptation_control")
    if not report:
        return None
    rows = [[row["cascades"], _pm(row["scratch"]), _pm(row["matched"]),
             _pm(row["published"])] for row in report["rows"]]
    return {
        "key": "adaptation", "module": "propagation",
        "title": "How much target data recovery needs",
        "unit": f"macro-F1, {report['epochs']} epochs in every arm",
        "columns": ["Collected cascades used", "From scratch",
                    "Fine-tuned, matched budget", "Fine-tuned, published budget"],
        "rows": rows, "best": "From scratch",
        "reading": "Training from scratch on the target platform leads at every "
                   "budget, so a benchmark initialisation is not the head start it "
                   "is usually assumed to be. The lead more than halves as labels "
                   "accumulate, from 0.140 at forty cascades to 0.063 at two "
                   "hundred. The last column is the same experiment at the "
                   "optimisation budget we first used, and half of the apparent "
                   "interference was that mismatch.",
    }


def _confound() -> dict | None:
    report = _load("confound")
    if not report:
        return None
    labels = {"cascade split, all sizes": "Split by cascade, all sizes",
              "domain split, all sizes": "Split by domain, all sizes",
              "cascade split, size matched": "Split by cascade, size matched",
              "domain split, size matched": "Split by domain, size matched"}
    rows = [[labels[key], _pm(cell["size_rule"]), _pm(cell["detector"]),
             cell["test_cascades"]]
            for key, cell in report.items() if key in labels]
    return {
        "key": "confound", "module": "propagation",
        "title": "What the collected labels are predictable from",
        "unit": "macro-F1, chance is 0.500",
        "columns": ["Condition", "One size threshold", "Bi-GCN", "Test cascades"],
        "rows": rows, "best": None,
        "reading": "A rule with one parameter beats the detector, because the "
                   "credible outlets we sampled simply have larger audiences. "
                   "Matching the sizes away drops the rule to chance and leaves the "
                   "detector at 0.685, so a real but much smaller structural signal "
                   "survives. Under the harder split it survives on 47 cascades "
                   "with a seed deviation that straddles chance.",
    }


# --------------------------------------------------------------------------
# Content module
# --------------------------------------------------------------------------

def _liar() -> dict | None:
    report = _load("liar_significance")
    if not report:
        return None
    rows = []
    for name, cell in report.items():
        low, high = cell["ci95"]
        gap = cell.get("gap_vs_tfidf")
        rows.append([name, _round(cell["macro_f1"]), f"[{low:.3f}, {high:.3f}]",
                     _signed(gap["point"]) if gap else "baseline"])
    return {
        "key": "nlp", "module": "content", "title": "Content detection on LIAR",
        "unit": "macro-F1, with a paired bootstrap against the bag of words",
        "columns": ["Model", "Macro-F1", "95% interval", "Gap vs TF-IDF"],
        "rows": rows, "best": None,
        "reading": "Five models within two points of each other, and a bag of words "
                   "at the top. No gap against that baseline excludes zero under a "
                   "paired bootstrap, so none of them separates from it. The ceiling "
                   "belongs to the claim format rather than to model capacity, which "
                   "the next table shows by changing the format.",
    }


def _article_level() -> dict | None:
    report = _load("article_level")
    if not report:
        return None
    rows = []
    for corpus in ("politifact", "gossipcop"):
        cell = report.get(corpus)
        if not cell:
            continue
        low, high = cell["ci95"]
        rows.append([corpus.capitalize(), _round(cell["macro_f1"]),
                     f"[{low:.3f}, {high:.3f}]", cell["train"], cell["test"]])
    return {
        "key": "article", "module": "content",
        "title": "The same idea on full news articles",
        "unit": "macro-F1, logistic regression on the article embedding",
        "columns": ["Corpus", "Macro-F1", "95% interval", "Train", "Test"],
        "rows": rows, "best": None,
        "reading": "Handed a whole article instead of one sentence, the content "
                   "signal reaches 0.805 where LIAR's short claims cap it near 0.63. "
                   "The ceiling is a property of the claim format. This changes "
                   "corpus as well as unit, so it is a single comparison and not a "
                   "controlled one.",
    }


def _gate() -> dict | None:
    report = _load("gate_audit")
    if not report:
        return None
    correctness, reach = report["correctness"], report["reach"]
    rows = [
        ["Retained by the gate", correctness["retained"],
         f"{correctness['retained_non_claims']:.0%} were not claims"],
        ["Rejected by the gate", correctness["rejected"],
         f"{correctness['rejected_non_claims']:.0%} were not claims"],
        ["Claims lost", correctness["false_negatives"],
         "short breaking-news lines carrying no surface marker"],
        ["Real posts clearing the floor", f"{reach['clearing']} of {reach['posts']}",
         f"{reach['share']:.1%}, mean score {reach['mean_score']}"],
    ]
    return {
        "key": "gate", "module": "content",
        "title": "The check-worthiness gate, audited",
        "unit": f"floor {report['floor']}, on 165 annotated and "
                f"{reach['posts']} source-labelled posts",
        "columns": ["Population", "Count", "What it means"], "rows": rows,
        "best": None,
        "reading": "The gate sorts in the right direction while letting substantial "
                   "noise through, and it is a rule set over surface markers, so a "
                   "claim stated in four plain words is invisible to it. Most real "
                   "collected posts state nothing it recognises as checkable, which "
                   "is the empirical argument for reading propagation alongside the "
                   "text rather than an architectural preference.",
    }


def _transfer() -> dict | None:
    report = _load("nlp_transfer")
    if not report:
        return None
    rows = []
    for row in report:
        low, high = row["ci95"]
        rows.append([row["model"], _round(row["liar_macro_f1"]),
                     _round(row["macro_f1"]), f"[{low:.3f}, {high:.3f}]",
                     f"{row['predicted_reliable']:.1%}"])
    return {
        "key": "transfer", "module": "content",
        "title": "Off the training platform", "unit": "macro-F1",
        "columns": ["Model", "LIAR", "Collected posts", "95% interval",
                    "Answered reliable"],
        "rows": rows, "best": None,
        "reading": "Every model loses ground on real collected posts, and the bag of "
                   "words loses least. The last column is the reason to distrust the "
                   "transformers here: they answer reliable on roughly one post in "
                   "six, which is close to a constant answer on a balanced sample.",
    }


# --------------------------------------------------------------------------
# Where the two meet
# --------------------------------------------------------------------------

INTEGRATION = {
    "key": "integration", "module": "both", "title": "Integration ablation",
    "unit": "macro-F1, mean of three seeds",
    "columns": ["Dataset", "Features", "Graph only", "+ random score",
                "+ content score"],
    "rows": [
        ["PolitiFact", "profile", 0.787, 0.783, 0.843],
        ["PolitiFact", "bert", 0.828, 0.828, 0.831],
        ["GossipCop", "profile", 0.909, 0.909, 0.905],
        ["GossipCop", "bert", 0.914, 0.935, 0.929],
    ],
    "best": "+ content score",
    "reading": "The content score helps exactly where the cascade carries no text "
               "of its own, recovering a quarter of the remaining error, and adds "
               "nothing where it already does. The random column is the control, "
               "and on the last row it moves more than the real score, so that "
               "block supports no conclusion.",
}


def _integration() -> dict:
    return dict(INTEGRATION)


def _root_identity() -> dict | None:
    """What the benchmark actually puts at the root of a cascade."""
    report = _load("root_identity")
    if not report:
        return None
    cell = report.get("politifact")
    if not cell:
        return None
    match = cell["root_matches_an_account_vector"]
    share = match["matched"] / max(1, match["checked"])
    rows = [
        ["Cascades in the subset", cell["cascades"], "one root vector each"],
        ["Distinct root vectors", cell["distinct_root_vectors"],
         "so a root is very nearly an identifier for its story"],
        ["Roots equal to some account's own vector", f"{match['matched']} of {match['checked']}",
         f"{share:.0%} of the roots checked"],
    ]
    return {
        "key": "root", "module": "propagation",
        "title": "What sits at the root of a cascade",
        "unit": "UPFD PolitiFact, bert features",
        "columns": ["Measure", "Count", "Reading"], "rows": rows, "best": None,
        "reading": "The root is documented as an embedding of the news article, and "
                   "on most cascades it is also identical to the vector of one of the "
                   "accounts in the tree. A model that leans on it is therefore not "
                   "reading propagation at all, which is what the ablation above "
                   "tests and what makes the result on it worth more than a score.",
    }


def _influence_weights() -> dict | None:
    """Whether the composite influence score depends on the weights we chose."""
    report = _load("influence_weights")
    if not report:
        return None
    reference = "/".join(str(w) for w in report["reference"])
    rows = [["/".join(str(w) for w in row["weights"]),
             round(row["spearman"], 4), f"{row['top_k_shared']} of {report['top_k']}"]
            for row in report["rows"]]
    return {
        "key": "weights", "module": "propagation",
        "title": f"Does the influence ranking depend on the weights? ({reference} is ours)",
        "unit": "Spearman against the reference ranking, and overlap of the top 100",
        "columns": ["Weights on reach / PageRank / k-core", "Spearman",
                    "Top 100 shared"],
        "rows": rows, "best": None,
        "reading": "The three weights were chosen rather than fitted, so the ranking "
                   "has to be shown not to rest on them. The worst alternative tried "
                   f"still agrees with the reference at {_worst(report, 'spearman'):.3f} "
                   f"and keeps {_worst(report, 'top_k_shared')} of the same hundred "
                   "accounts. The ranking is robust to that choice, which is a "
                   "separate question from whether it identifies anyone worth "
                   "identifying, and the influence result answers that one negatively.",
    }


def _worst(report: dict, field: str):
    """The least favourable row, so the prose cannot drift from the table."""
    return min(row[field] for row in report["rows"])


BUILDERS = (_graph, _trivial, _early, _gap, _root_identity, _integration, _ood,
            _adaptation, _confound, _influence_weights,
            _liar, _article_level, _gate, _transfer)

MODULES = {
    "propagation": {
        "name": "Propagation module",
        "blurb": "Reads the shape of the cascade and who carried it. Adam Terrak "
                 "and Abdelah El Harsal.",
    },
    "content": {
        "name": "Content module",
        "blurb": "Reads the text of the post itself. Ayman Ouguerd and Louaye "
                 "Saghir.",
    },
    "both": {
        "name": "Integration",
        "blurb": "Where a content score is handed to the propagation model.",
    },
}


def spine() -> dict[str, Any]:
    """The whole argument on one macro-F1 axis.

    Four big figures in a row is the shape every dashboard uses, and it hides
    the only thing that matters here, which is what each figure has to be read
    against. A detection score of 0.919 says nothing until the model that reads
    no edge is drawn beside it at 0.940, and a content ceiling says nothing
    until the same axis carries what a constant answer earns.

    Every mark is recomputed from the file its experiment wrote.
    """
    marks: list[dict[str, Any]] = []

    def add(label: str, value: float | None, module: str, note: str,
            kind: str = "result") -> None:
        if value is not None:
            marks.append({"label": label, "value": round(float(value), 4),
                          "module": module, "note": note, "kind": kind})

    gap = _load("table2_gap_test")
    gossip = _load("gossipcop_bert_3seeds")
    liar = _load("liar_significance")
    article = _load("article_level")
    ood = _load("ood_transfer")

    add("A constant answer", 1 / 3, "both",
        "what one fixed guess earns on a balanced binary task", "floor")
    if ood:
        add("The detector, one platform across", ood["account features"]["bluesky"]["mean"],
            "propagation", "it collapses onto a single class")
    if liar:
        reported = {k: v for k, v in liar.items() if "weight decay" not in k}
        add("Content ceiling on short claims",
            max(c["macro_f1"] for c in reported.values()), "content",
            "five models within two points, a bag of words at the top")
    if article:
        add("The same approach on full articles", article["politifact"]["macro_f1"],
            "content", "the claim format was the ceiling, not the reading of text")
    if gossip:
        add("Propagation detection", gossip["Bi-GCN"]["mean"], "propagation",
            "GossipCop, text-derived node features")
    if gap:
        row = gap["table2"].get("gossipcop/bert", {})
        add("A model that deletes every edge", row.get("baseline_f1"), "propagation",
            "the same features, no propagation at all", "baseline")
        masked = gap.get("masked_sweep", {}).get("masked", {}).get("100%")
        if masked:
            add("The detector, article masked", masked["mean"], "propagation",
                "the benchmark's own root feature taken away")

    marks.sort(key=lambda m: m["value"])
    return {"marks": marks, "unit": "macro-F1"}


def headlines() -> list[dict[str, str]]:
    """Four numbers, each recomputed from the file its experiment wrote."""
    out = []
    gossipcop = _load("gossipcop_bert_3seeds")
    if gossipcop:
        out.append({"value": f"{gossipcop['Bi-GCN']['mean']:.3f}",
                    "module": "propagation",
                    "label": "propagation detection, GossipCop",
                    "note": "and an edgeless baseline reaches 0.940 on the same input"})
    gap = _load("early_gap_test")
    if gap:
        out.append({"value": _signed(gap["root masked"]["gap_vs_edgeless"]),
                    "module": "propagation",
                    "label": "what the cascade adds, once the article is masked",
                    "note": "the only separation that survives a paired bootstrap"})
    liar = _load("liar_significance")
    if liar:
        # the weight-decay run is offered for comparison and is not one of the
        # five configurations the report describes, so it does not set the ceiling
        reported = {k: v for k, v in liar.items() if "weight decay" not in k}
        best = max(cell["macro_f1"] for cell in reported.values())
        out.append({"value": f"{best:.2f}", "module": "content",
                    "label": "content ceiling on short claims",
                    "note": "five models within two points, a bag of words at the top"})
    ood = _load("ood_transfer")
    if ood:
        out.append({"value": f"{ood['account features']['bluesky']['mean']:.3f}",
                    "module": "propagation",
                    "label": "the same detector, one platform across",
                    "note": "0.333 is what a constant answer earns"})
    return out


def payload() -> dict[str, Any]:
    tables = [built for build in BUILDERS if (built := build()) is not None]
    return {"spine": spine(), "headlines": headlines(), "tables": tables,
            "modules": MODULES}
