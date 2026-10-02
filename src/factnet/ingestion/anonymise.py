"""Export an anonymised sample of the collected cascades (for HF / sharing).

Accounts -> stable pseudonyms, mentions and links in the text are masked.
Structure and features are kept. The post text stays (it's what we classify),
so a post could still be found by searching its wording, we don't claim more.

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

# @mentions inside the text
MENTION = re.compile(r"@?\b[\w.-]+\.(bsky\.social|social|com|org|net)\b|did:plc:[a-z0-9]+")

# links too: the label comes from the linked outlet, so leaving the link in
# basically gives the answer away
LINK = re.compile(
    r"https?://\S+"
    r"|\bwww\.\S+"
    r"|\b[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}/\S*",   # a bare domain followed by a path
    re.IGNORECASE,
)


def scrub_mentions(text: str) -> str:
    return MENTION.sub("@account", LINK.sub("@link", text or ""))


def anonymise(cascades: list[dict]) -> list[dict]:
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
        # keep the outlet + rating, that's where the label comes from (not personal data)
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
