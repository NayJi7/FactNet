"""Check-worthiness: is the post even a claim? (claim spotting, like ClaimBuster)

Rule-based, we had no labelled data to train one. Positive markers: numbers,
dates, named sources, institutions, measurable things ("unemployment rate"),
magnitude verbs ("has doubled"). Negative: first person, questions, opinion.
See gate_audit.py for how well it does.

    uv run python -m factnet.nlp.checkworthy --file data/raw/bluesky/cascades.jsonl
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# write every verb form out. had "confirmed?" before, which is confirme + d?
# and never matched "confirm"/"confirms"
QUANTITY = re.compile(r"\b\d[\d,.]*\s*(%|percent|million|billion|thousand|k\b)|\B\$\d|\b\d{4}\b")
ATTRIBUTION = re.compile(
    r"\b(say|says|said|claim|claims|claimed|announce|announces|announced|"
    r"report|reports|reported|according to|admit|admits|admitted|"
    r"deny|denies|denied|confirm|confirms|confirmed|warn|warns|warned|"
    r"tell|tells|told|state|states|stated|allege|alleges|alleged|"
    r"insist|insists|insisted|reveal|reveals|revealed)\b", re.I)
# acronyms are case sensitive (otherwise "who" matches WHO everywhere)
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
    # "has doubled" etc, a quantity without a number
    r"double|doubles|doubled|triple|triples|tripled|quadruple|quadrupled|"
    r"halve|halves|halved|rise|rises|rose|risen|fall|falls|fell|fallen|"
    r"drop|drops|dropped|surge|surges|surged|plunge|plunges|plunged|"
    r"soar|soars|soared|jump|jumps|jumped|spike|spikes|spiked|"
    r"plummet|plummets|plummeted|grow|grows|grew|grown|"
    r"shrink|shrinks|shrank|shrunk|decline|declines|declined|"
    r"climb|climbs|climbed|overtake|overtakes|overtook|"
    r"exceed|exceeds|exceeded)\b", re.I)

# stuff a statistics office would publish
MEASURE = re.compile(r"\b(rate|rates|percentage|average|median|unemployment|inflation|"
                     r"gdp|deficit|surplus|turnout|census|poll|polls|survey|statistics|"
                     r"death toll|cases|deaths|births|population|prices?|costs?|wages?|"
                     r"salaries|salary|temperature|emissions|revenue|budget)\b", re.I)

FIRST_PERSON = re.compile(r"\b(i|i'm|i've|my|me|myself|we're|folks)\b", re.I)
OPINION = re.compile(r"\b(think|feel|believe|hope|love|hate|beautiful|awful|lol|lmao|"
                     r"please|thanks|congrats|funny|guess)\b", re.I)
QUESTION = re.compile(r"\?\s*$")


def score(text: str) -> float:
    """0..1"""
    stripped = (text or "").strip()
    if len(stripped) < 25:
        return 0.0

    # one marker is enough. letting a non-claim through is cheap, blocking a real
    # one isn't ("Trump says the election was stolen" only has one marker)
    positive = sum(weight for pattern, weight in (
        (QUANTITY, 0.30), (ATTRIBUTION, 0.25), (INSTITUTION, 0.25),
        (CHANGE, 0.25), (MEASURE, 0.25))
        if pattern.search(stripped))
    penalty = sum(weight for pattern, weight in (
        (FIRST_PERSON, 0.25), (OPINION, 0.20), (QUESTION, 0.15))
        if pattern.search(stripped))

    length_bonus = 0.10 if len(stripped) > 140 else 0.0
    return max(0.0, min(1.0, positive + length_bonus - penalty))


def explain(text: str) -> dict[str, bool]:
    """which rules fired"""
    return {"quantity": bool(QUANTITY.search(text or "")),
            "attribution": bool(ATTRIBUTION.search(text or "")),
            "institution": bool(INSTITUTION.search(text or "")),
            "change": bool(CHANGE.search(text or "")),
            "measure": bool(MEASURE.search(text or "")),
            "first_person": bool(FIRST_PERSON.search(text or "")),
            "opinion": bool(OPINION.search(text or "")),
            "question": bool(QUESTION.search(text or ""))}


def rank(cascades: list[dict]) -> list[dict]:
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
