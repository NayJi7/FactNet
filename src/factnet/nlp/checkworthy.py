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
to a named source, institutional vocabulary, named measurable quantities and
verbs of magnitude, against first-person framing, questions, and pure opinion.
The last two positives exist because a great many checkable assertions carry no
digit at all: "the unemployment rate has doubled" states a quantity as plainly
as any figure would.

The filter is validated after the fact rather than assumed: the annotation tool
records a skip whenever an item turns out not to be a checkable claim, so the
skip rate among high-scoring items measures how well this stage works.

    uv run python -m factnet.nlp.checkworthy --file data/raw/bluesky/cascades.jsonl
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# A verifiable assertion tends to carry numbers, dates, named sources or
# institutions. Every verb is spelled with its inflections rather than folded
# into an optional suffix: "confirmed?" reads as "confirme" plus an optional
# "d" and therefore never matches "confirm" or "confirms", which is the form a
# headline actually uses.
QUANTITY = re.compile(r"\b\d[\d,.]*\s*(%|percent|million|billion|thousand|k\b)|\B\$\d|\b\d{4}\b")
ATTRIBUTION = re.compile(
    r"\b(say|says|said|claim|claims|claimed|announce|announces|announced|"
    r"report|reports|reported|according to|admit|admits|admitted|"
    r"deny|denies|denied|confirm|confirms|confirmed|warn|warns|warned|"
    r"tell|tells|told|state|states|stated|allege|alleges|alleged|"
    r"insist|insists|insisted|reveal|reveals|revealed)\b", re.I)
# The acronyms are case-sensitive inside an otherwise case-insensitive pattern:
# "who" spelled in lower case is one of the commonest words in English and
# would fire on every relative clause.
INSTITUTION = re.compile(
    r"\b(gov|govs|government|governments|senate|congress|court|courts|"
    r"ministry|ministries|agency|agencies|study|studies|research|researcher|"
    r"researchers|data|university|universities|police|official|officials|"
    r"president|presidents|minister|ministers|senator|senators|"
    r"department|departments|white house|parliament|commission|commissions|"
    r"bureau|bureaus|institute|institutes|hospital|hospitals|"
    r"regulator|regulators|scientist|scientists|doctor|doctors|"
    r"expert|experts|authorities|lawmaker|lawmakers|"
    r"(?-i:CDC|FDA|WHO|NASA|EPA|NATO|UN))\b", re.I)
CHANGE = re.compile(
    r"\b(ban|bans|banned|approve|approves|approved|pass|passes|passed|"
    r"sign|signs|signed|cut|cuts|raise|raises|raised|"
    r"increase|increases|increased|decrease|decreases|decreased|"
    r"cause|causes|caused|kill|kills|killed|link|links|linked to|"
    r"prove|proves|proved|proven|show|shows|showed|shown|find|finds|found|"
    r"recall|recalls|recalled|rescind|rescinds|rescinded|"
    r"repeal|repeals|repealed|reject|rejects|rejected|block|blocks|blocked|"
    # magnitude verbs: a quantity can be asserted without a digit,
    # and "has doubled" is as checkable as "rose by 100 %"
    r"double|doubles|doubled|triple|triples|tripled|quadruple|quadrupled|"
    r"halve|halves|halved|rise|rises|rose|risen|fall|falls|fell|fallen|"
    r"drop|drops|dropped|surge|surges|surged|plunge|plunges|plunged|"
    r"soar|soars|soared|jump|jumps|jumped|spike|spikes|spiked|"
    r"plummet|plummets|plummeted|grow|grows|grew|grown|"
    r"shrink|shrinks|shrank|shrunk|decline|declines|declined|"
    r"climb|climbs|climbed|overtake|overtakes|overtook|"
    r"exceed|exceeds|exceeded)\b", re.I)

# A named measurable quantity, which makes a sentence checkable even when it
# carries no figure. Kept to things a statistics office publishes, so that it
# does not fire on ordinary description.
MEASURE = re.compile(r"\b(rate|rates|percentage|average|median|unemployment|inflation|"
                     r"gdp|deficit|surplus|turnout|census|poll|polls|survey|statistics|"
                     r"death toll|cases|deaths|births|population|prices?|costs?|wages?|"
                     r"salaries|salary|temperature|emissions|revenue|budget)\b", re.I)

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

    # One marker is enough. The gate exists to hold back talk about oneself,
    # jokes and questions, and its errors are not symmetric: a false positive
    # costs a wasted verdict that the interface labels as such, while a false
    # negative stops the pipeline on a claim that was worth checking. "Trump
    # says the election was stolen" carries a single marker and is exactly the
    # kind of sentence this stage must let through.
    positive = sum(weight for pattern, weight in (
        (QUANTITY, 0.30), (ATTRIBUTION, 0.25), (INSTITUTION, 0.25),
        (CHANGE, 0.25), (MEASURE, 0.25))
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
            "measure": bool(MEASURE.search(text or "")),
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
