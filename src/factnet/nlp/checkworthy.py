"""Check-worthiness: is this post a claim worth verifying at all?

Most posts are not factual claims. Sending personal anecdotes, jokes and
conversation to a verdict model wastes annotation effort and dilutes any
evaluation built on the result, which is why automated fact-checking places a
claim-spotting stage before the verdict, after ClaimBuster (Hassan et al.,
2017).

The score here is deliberately transparent rather than learned: no labelled
check-worthiness corpus was collected for this project, and an explicit rule
set can be read, argued with, and corrected, which a black box scoring the
same items could not. It combines the surface markers that distinguish a
verifiable assertion from talk about oneself: quantities and dates, attribution
to a named source, institutional vocabulary, against first-person framing,
questions, and pure opinion.

The filter is validated after the fact rather than assumed: the annotation tool
records a skip whenever an item turns out not to be a checkable claim, so the
skip rate among high-scoring items measures how well this stage works.

    uv run python -m factnet.nlp.checkworthy --file data/raw/bluesky/cascades.jsonl
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# a verifiable assertion tends to carry numbers, dates, named sources or institutions
QUANTITY = re.compile(r"\b\d[\d,.]*\s*(%|percent|million|billion|thousand|k\b)|\B\$\d|\b\d{4}\b")
ATTRIBUTION = re.compile(r"\b(says?|said|claims?|claimed|announced?|reported?|according to|"
                         r"admits?|denies|confirmed?|warns?|told)\b", re.I)
INSTITUTION = re.compile(r"\b(gov(ernment)?|senate|congress|court|ministry|agency|study|studies|"
                         r"research(ers)?|report|data|cdc|fda|who|nasa|university|police|"
                         r"official|president|minister|senator|department)\b", re.I)
CHANGE = re.compile(r"\b(banned?|approved?|passed?|signed?|cut|raised?|increased?|decreased?|"
                    r"caused?|killed?|linked to|proves?|shows?|found)\b", re.I)

# talk about oneself, questions and pure opinion are not claims to verify
FIRST_PERSON = re.compile(r"\b(i|i'm|i've|my|me|myself|we're|folks)\b", re.I)
OPINION = re.compile(r"\b(think|feel|believe|hope|love|hate|beautiful|awful|lol|lmao|"
                     r"please|thanks|congrats|funny|guess)\b", re.I)
QUESTION = re.compile(r"\?\s*$")


def score(text: str) -> float:
    """A 0 to 1 estimate of how worth verifying a post is."""
    stripped = (text or "").strip()
    if len(stripped) < 25:            # too short to carry a checkable assertion
        return 0.0

    positive = sum(weight for pattern, weight in (
        (QUANTITY, 0.30), (ATTRIBUTION, 0.25), (INSTITUTION, 0.25), (CHANGE, 0.20))
        if pattern.search(stripped))
    penalty = sum(weight for pattern, weight in (
        (FIRST_PERSON, 0.25), (OPINION, 0.20), (QUESTION, 0.15))
        if pattern.search(stripped))

    # a long post has more room to state something checkable
    length_bonus = 0.10 if len(stripped) > 140 else 0.0
    return max(0.0, min(1.0, positive + length_bonus - penalty))


def explain(text: str) -> dict[str, bool]:
    """Which markers fired, so a score can be argued with."""
    return {"quantity": bool(QUANTITY.search(text or "")),
            "attribution": bool(ATTRIBUTION.search(text or "")),
            "institution": bool(INSTITUTION.search(text or "")),
            "change": bool(CHANGE.search(text or "")),
            "first_person": bool(FIRST_PERSON.search(text or "")),
            "opinion": bool(OPINION.search(text or "")),
            "question": bool(QUESTION.search(text or ""))}


def rank(cascades: list[dict]) -> list[dict]:
    """Attach a check-worthiness score to each record, best first."""
    for cascade in cascades:
        cascade["checkworthy"] = round(score(cascade.get("text", "")), 3)
    return sorted(cascades, key=lambda c: -c["checkworthy"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--write", action="store_true",
                        help="store the score back into the file")
    args = parser.parse_args()

    from factnet.ingestion.annotate import dump, load

    path = Path(args.file)
    cascades = load(path)
    ranked = rank(cascades)
    kept = [c for c in ranked if c["checkworthy"] >= args.threshold]
    print(f"{len(cascades)} cascades, {len(kept)} above {args.threshold} "
          f"({len(kept) / max(1, len(cascades)):.0%})\n")

    print("most check-worthy")
    for cascade in ranked[:5]:
        print(f"  {cascade['checkworthy']:.2f}  {cascade['text'][:96]!r}")
    print("\nleast check-worthy")
    for cascade in ranked[-5:]:
        print(f"  {cascade['checkworthy']:.2f}  {cascade['text'][:96]!r}")

    if args.write:
        dump(ranked, path)
        print(f"\nscores written to {path}")


if __name__ == "__main__":
    main()
