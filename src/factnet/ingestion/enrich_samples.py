"""Fill in the display names the collector did not keep.

The collector records the handle, which is an identifier, and not the name an
account shows, which is what a reader recognises. The interface prints the post
that started each cascade, and "mcuban.bsky.social" beside a headline reads as
a database key rather than as a person.

Nothing about the graph changes: no node is added or removed, no edge is
touched, and the profile vectors the models read are left exactly as they were,
so every published figure still stands. Verified by comparing the two files
after the first run.

    uv run python -m factnet.ingestion.enrich_samples
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from factnet.ingestion.bluesky import BlueskyClient

SAMPLES = Path(__file__).resolve().parents[3] / "data" / "samples.jsonl"


def _names(node: dict, into: dict[str, str]) -> None:
    """Every display name in a thread, keyed by the account it belongs to."""
    for reply in node.get("replies") or []:
        author = (reply.get("post") or {}).get("author") or {}
        did = author.get("did")
        if did and did not in into:
            into[did] = author.get("displayName", "")
        _names(reply, into)


def enrich(cascade: dict[str, Any], client: BlueskyClient) -> tuple[int, int]:
    """Fill in what each account calls itself. Returns accounts named and total."""
    thread = client.thread(cascade["uri"], depth=10)
    found: dict[str, str] = {}
    _names(thread or {}, found)
    root_author = ((thread or {}).get("post") or {}).get("author") or {}
    if root_author.get("did"):
        found[root_author["did"]] = root_author.get("displayName", "")

    named = 0
    for node in cascade["nodes"]:
        name = found.get(node.get("did", ""))
        if name:
            node["display_name"] = name
            named += 1
    return named, len(cascade["nodes"])


def main() -> None:
    client = BlueskyClient()
    cascades = [json.loads(line) for line in SAMPLES.open(encoding="utf-8") if line.strip()]
    for cascade in cascades:
        try:
            named, total = enrich(cascade, client)
            print(f"  {cascade.get('source_domain', '?'):22s} {named:>4}/{total} named")
        except Exception as error:                       # reported, never silent
            print(f"  {cascade.get('source_domain', '?'):22s} "
                  f"failed: {type(error).__name__}: {error}")

    with SAMPLES.open("w", encoding="utf-8") as handle:
        for cascade in cascades:
            handle.write(json.dumps(cascade, ensure_ascii=False) + "\n")
    print(f"\nwritten to {SAMPLES}")


if __name__ == "__main__":
    main()
