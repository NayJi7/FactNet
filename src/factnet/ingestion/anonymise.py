"""Export a shareable, anonymised sample of the collected cascades.

The collected records carry account identities, which are public but which this
project has no reason to redistribute: the object of study is how claims
propagate, not who the participants are. This module produces the sample that
can be shared, keeping the propagation structure and the account features that
the models actually read, and replacing every identity by a stable pseudonym.

The claim text is kept, since it is the object being classified and a sample
without it could not be inspected or reused. That is a deliberate limit rather
than an oversight: a public post can in principle be traced back from its own
wording, so the export removes identities without pretending to make the
records untraceable, and it is restricted to check-worthy claims so that what
is shared is material with research value.

    uv run python -m factnet.ingestion.anonymise --min-checkworthy 0.3
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from factnet.ingestion.annotate import load
from factnet.nlp.checkworthy import score

DEFAULT_IN = Path(__file__).resolve().parents[3] / "data" / "raw" / "bluesky" / "cascades.jsonl"
DEFAULT_OUT = DEFAULT_IN.with_name("cascades-anonymised.jsonl")

# a post can name other accounts inside its own text, which the node pseudonyms
# would otherwise leave untouched
MENTION = re.compile(r"@?\b[\w.-]+\.(bsky\.social|social|com|org|net)\b|did:plc:[a-z0-9]+")

# Bluesky renders the link inside the post, and on this sample the label was
# read from precisely that link, so leaving it in publishes the answer beside
# the question. The mention pattern above happened to remove most of them, but
# only because it lists three of the endings a link can have; this is explicit
# so that the next collection does not republish the leak on a suffix nobody
# thought of.
LINK = re.compile(
    r"https?://\S+"
    r"|\bwww\.\S+"
    r"|\b[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}/\S*",   # a bare domain followed by a path
    re.IGNORECASE,
)


def scrub_mentions(text: str) -> str:
    """Replace handles and outlet links cited inside a post by neutral markers."""
    return MENTION.sub("@account", LINK.sub("@link", text or ""))


def anonymise(cascades: list[dict]) -> list[dict]:
    """Replace every account identity by a stable pseudonym across the sample."""
    aliases: dict[str, str] = {}

    def alias(did: str) -> str:
        if did not in aliases:
            aliases[did] = f"A{len(aliases) + 1:05d}"
        return aliases[did]

    out = []
    for index, cascade in enumerate(cascades, 1):
        nodes = [{"account": alias(n["did"]), "kind": n.get("kind", ""),
                  "profile": n["profile"]} for n in cascade["nodes"]]
        edges = [{"source": alias(e["source"]), "target": alias(e["target"]),
                  "kind": e.get("kind", "")} for e in cascade["edges"]]
        record = {
            "cascade_id": f"C{index:04d}",
            "query": cascade.get("query", ""),
            "text": scrub_mentions(cascade.get("text", "")),
            "created_at": cascade.get("created_at", ""),
            "repost_count": cascade.get("repost_count", 0),
            "reply_count": cascade.get("reply_count", 0),
            "like_count": cascade.get("like_count", 0),
            "checkworthy": cascade.get("checkworthy", round(score(cascade.get("text", "")), 3)),
            "label": cascade.get("label"),
            "nodes": nodes,
            "edges": edges,
        }
        # where the label was read from the credibility of the linked outlet, the
        # outlet and its rating travel with the record: they are the provenance of
        # the label, and naming a publisher discloses nothing about a person
        if cascade.get("source_domain"):
            record["source_domain"] = cascade["source_domain"]
            record["source_label"] = cascade.get("source_label", "")
        out.append(record)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=str(DEFAULT_IN))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--min-checkworthy", type=float, default=0.3,
                        help="only export posts that state a checkable claim")
    args = parser.parse_args()

    cascades = load(args.file)
    kept = [c for c in cascades if score(c.get("text", "")) >= args.min_checkworthy]
    exported = anonymise(kept)

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as handle:
        for record in exported:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    accounts = {n["account"] for c in exported for n in c["nodes"]}
    print(f"{len(cascades)} collected, {len(exported)} exported "
          f"(check-worthiness >= {args.min_checkworthy})")
    print(f"  {len(accounts)} distinct accounts, all pseudonymised")
    print(f"  {sum(len(c['edges']) for c in exported)} shares")
    print(f"  written to {out}")


if __name__ == "__main__":
    main()
