"""Audit of the check-worthiness gate.

correctness(): vs the annotator's skips (annotated set).
reach(): share of the source-labelled stream that passes.
Was done by hand before, moved here after the "confirmed?" regex bug.

    uv run python -m factnet.nlp.gate_audit
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from factnet.nlp.checkworthy import score

ROOT = Path(__file__).resolve().parents[3]
ANNOTATED = ROOT / "data" / "raw" / "bluesky" / "cascades.jsonl"
BY_SOURCE = ROOT / "data" / "raw" / "bluesky" / "cascades-by-source.jsonl"
RESULTS = ROOT / "data" / "results"
FLOOR = 0.25


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def correctness(floor: float = FLOOR) -> dict:
    """skip = annotator said "not a claim". kept skip = noise, rejected non-skip =
    lost claim (the worse one)."""
    rows = [r for r in _read(ANNOTATED) if r.get("annotation_meta")]
    kept = [r for r in rows if score(r.get("text", "")) >= floor]
    rejected = [r for r in rows if score(r.get("text", "")) < floor]
    skipped = lambda rs: sum(1 for r in rs if r["annotation_meta"]["verdict"] == "s")  # noqa: E731
    return {
        "annotated": len(rows),
        "retained": len(kept),
        "retained_non_claims": round(skipped(kept) / max(1, len(kept)), 3),
        "rejected": len(rejected),
        "rejected_non_claims": round(skipped(rejected) / max(1, len(rejected)), 3),
        "false_negatives": len(rejected) - skipped(rejected),
    }


def reach(floor: float = FLOOR) -> dict:
    rows = [r for r in _read(BY_SOURCE)
            if r.get("label") is not None and (r.get("text") or "").strip()]
    scores = [score(r["text"]) for r in rows]
    kept = [r for r, s in zip(rows, scores, strict=True) if s >= floor]
    by_class = {}
    for label, name in ((0, "misleading"), (1, "reliable")):
        total = sum(1 for r in rows if r["label"] == label)
        passing = sum(1 for r in kept if r["label"] == label)
        by_class[name] = {"posts": total, "clearing": passing,
                          "share": round(passing / max(1, total), 3)}
    return {"posts": len(rows), "clearing": len(kept),
            "share": round(len(kept) / len(rows), 3),
            "mean_score": round(st.fmean(scores), 3), "by_class": by_class}


def main() -> None:
    report = {"floor": FLOOR, "correctness": correctness(), "reach": reach()}
    c, r = report["correctness"], report["reach"]
    print(f"Check-worthiness gate, floor {FLOOR}\n")
    print(f"Correctness, {c['annotated']} annotated records")
    print(f"  retained {c['retained']}, of which {c['retained_non_claims']:.0%} non-claims")
    print(f"  rejected {c['rejected']}, of which {c['rejected_non_claims']:.0%} non-claims")
    print(f"  claims lost: {c['false_negatives']}")
    print(f"\nReach, {r['posts']} source-labelled posts")
    print(f"  {r['clearing']} clear the floor ({r['share']:.1%}), mean score {r['mean_score']}")
    for name, cell in r["by_class"].items():
        print(f"  {name:11s} {cell['clearing']:>3}/{cell['posts']} = {cell['share']:.1%}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "gate_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
