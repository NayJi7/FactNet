"""Annotate collected cascades, with inter-annotator agreement.

The literature review commits this project to a written protocol and more than
one annotator per item, so that agreement can be measured rather than assumed:
single-annotator labels conflate genuine signal with individual judgement.

Each annotator works on the same file and stores decisions under their own name,
so the same record can carry several independent labels. Cohen's kappa is then
computed over the items both annotators judged, and the consensus label (used
for evaluation) is written only where they agree.

    uv run python -m factnet.ingestion.annotate --annotator adam
    uv run python -m factnet.ingestion.annotate --agreement
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = {"r": 1, "m": 0}          # reliable / misleading
SKIP, QUIT = "s", "q"
DEFAULT = Path(__file__).resolve().parents[3] / "data" / "raw" / "bluesky" / "cascades.jsonl"

PROTOCOL = """Annotation protocol
  reliable (r)    the claim is verifiable and consistent with reporting from
                  established sources, or is a plain factual statement
  misleading (m)  the claim is false, unsupported, or true but framed so as to
                  mislead (missing context, misattributed, exaggerated)
  skip (s)        not a factual claim (opinion, joke, conversation), or not
                  decidable without expertise the annotator does not have
Judge the claim, not the account, and not whether you agree with it."""


def load(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def cohen_kappa(pairs: list[tuple[int, int]]) -> float:
    """Agreement beyond chance over jointly annotated items."""
    if not pairs:
        return float("nan")
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    expected = 0.0
    for label in (0, 1):
        expected += (sum(1 for a, _ in pairs if a == label) / n) * \
                    (sum(1 for _, b in pairs if b == label) / n)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def consensus(record: dict) -> int | None:
    """Label kept for evaluation: only where every annotator agrees."""
    votes = {v for v in (record.get("annotations") or {}).values() if v is not None}
    return votes.pop() if len(votes) == 1 else None


def annotate(path: Path, annotator: str, limit: int | None = None,
             biggest_first: bool = True, min_checkworthy: float = 0.3) -> None:
    from factnet.nlp.checkworthy import score as checkworthy

    records = load(path)
    print(PROTOCOL)

    # an annotation session is short, so spend it where it buys the most: posts
    # that actually state something checkable, largest cascade first among them
    for record in records:
        record.setdefault("checkworthy", round(checkworthy(record.get("text", "")), 3))
    queue = list(enumerate(records))
    if biggest_first:
        queue = [(i, r) for i, r in queue if r["checkworthy"] >= min_checkworthy]
        queue.sort(key=lambda p: (-p[1]["checkworthy"], -len(p[1]["nodes"])))
        print(f"\n{len(queue)} of {len(records)} cascades state a checkable claim "
              f"(check-worthiness >= {min_checkworthy}); the rest are skipped.")
    print("enter r / m / s, or q to stop.\n")

    done = 0
    for i, record in queue:
        record.setdefault("annotations", {})
        if annotator in record["annotations"] or (limit is not None and done >= limit):
            continue
        engagement = (f"{len(record['nodes'])} accounts, {record.get('repost_count', 0)} reposts,"
                      f" check-worthiness {record.get('checkworthy', 0):.2f}")
        print(f"[{i + 1}/{len(records)}] @{record.get('source_handle', '?')}  ({engagement})")
        print(f"  {record.get('text', '').strip()[:500]}")
        choice = input("  reliable/misleading/skip [r/m/s/q] > ").strip().lower()
        if choice == QUIT:
            break
        if choice == SKIP:
            record["annotations"][annotator] = None
        elif choice in LABELS:
            record["annotations"][annotator] = LABELS[choice]
        else:
            print("  unrecognised, skipping this record\n")
            continue
        record["label"] = consensus(record)
        done += 1
        print()

    dump(records, path)
    labelled = sum(1 for r in records if r.get("label") is not None)
    print(f"saved {path}  |  {done} annotated this session, {labelled} with a consensus label")


def report_agreement(path: Path) -> None:
    records = load(path)
    annotators = sorted({name for r in records for name in (r.get("annotations") or {})})
    print(f"annotators: {', '.join(annotators) if annotators else '(none yet)'}")
    for i, first in enumerate(annotators):
        for second in annotators[i + 1:]:
            pairs = [(r["annotations"][first], r["annotations"][second]) for r in records
                     if r.get("annotations", {}).get(first) is not None
                     and r.get("annotations", {}).get(second) is not None]
            if not pairs:
                continue
            agree = sum(1 for a, b in pairs if a == b)
            print(f"  {first} vs {second}: {len(pairs)} common items, "
                  f"raw agreement {agree / len(pairs):.2f}, "
                  f"Cohen's kappa {cohen_kappa(pairs):.2f}")
    labelled = [r for r in records if r.get("label") is not None]
    if labelled:
        reliable = sum(1 for r in labelled if r["label"] == 1)
        print(f"consensus labels: {len(labelled)} "
              f"({reliable} reliable, {len(labelled) - reliable} misleading)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=str(DEFAULT))
    parser.add_argument("--annotator", help="your name, stored with each decision")
    parser.add_argument("--limit", type=int, default=None, help="stop after N items")
    parser.add_argument("--file-order", action="store_true",
                        help="annotate every record in file order, unfiltered")
    parser.add_argument("--min-checkworthy", type=float, default=0.3,
                        help="skip posts below this check-worthiness score")
    parser.add_argument("--agreement", action="store_true",
                        help="report inter-annotator agreement and exit")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"no collected file at {path}")
    if args.agreement:
        report_agreement(path)
    elif args.annotator:
        annotate(path, args.annotator, args.limit,
                 biggest_first=not args.file_order,
                 min_checkworthy=args.min_checkworthy)
    else:
        parser.error("give --annotator NAME, or --agreement")


if __name__ == "__main__":
    main()
