"""Collect cascades whose label comes from the credibility of the linked source.

Keyword search returned an unusable class balance, so this pass searches by
domain instead: for each rated outlet, the posts that carry a link to it are
retrieved and their cascades rebuilt, and the label is read from the rating of
that outlet rather than from a judgement about the post.

    uv run python -m factnet.ingestion.collect_by_source --target 200
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from factnet.ingestion.bluesky import BlueskyClient, collect_cascade, save
from factnet.ingestion.domains import (
    CACHE,
    HIGH_CREDIBILITY,
    classify,
    fetch_iffy,
    is_article,
    load_low_credibility,
    post_links,
    registrable,
)

OUT = Path(__file__).resolve().parents[3] / "data" / "raw" / "bluesky" / "cascades-by-source.jsonl"


def ranked_low_domains(limit: int, path: Path = CACHE) -> list[str]:
    """Listed unreliable domains, most trafficked first: those with a real audience."""
    fetch_iffy(path)
    keep = set(load_low_credibility(path))
    with path.open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle)
                if (r.get("Domain") or "").strip().lower() in keep]

    def rank(row: dict) -> int:
        try:
            return int(row.get("Site Rank") or 10**9)
        except ValueError:
            return 10**9

    rows.sort(key=rank)
    return [r["Domain"].strip().lower() for r in rows if r.get("Domain")][:limit]


def collect(domains: list[str], label: str, client: BlueskyClient, low: dict[str, str],
            target: int, per_domain: int, min_engagement: int, seen: set[str]) -> list[dict]:
    cascades = []
    for domain in domains:
        if len(cascades) >= target:
            break
        try:
            posts = client.search_posts(f"domain:{domain}", limit=per_domain, sort="top")
        except Exception as err:
            print(f"  {domain}: search failed ({type(err).__name__})")
            continue

        kept = 0
        for post in posts:
            uri = post.get("uri")
            if not uri or uri in seen:
                continue
            # the link must be attached to the post, not merely named in its text,
            # and it must resolve to a rated source
            links = [url for url in post_links(post)
                     if is_article(url) and classify(url, low) == label]
            if not links:
                continue
            if int(post.get("repostCount", 0)) + int(post.get("replyCount", 0)) < min_engagement:
                continue
            seen.add(uri)
            try:
                cascade = collect_cascade(client, post)
            except Exception:
                continue
            cascade.update(query=f"domain:{domain}", source_domain=registrable(links[0]),
                           source_label=label, label=1 if label == "reliable" else 0,
                           annotations={"source": 1 if label == "reliable" else 0})
            cascades.append(cascade)
            kept += 1
            if len(cascades) >= target:
                break
        if kept:
            print(f"  {domain}: {kept} cascades ({len(cascades)}/{target})")
    return cascades


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=200, help="cascades per class")
    parser.add_argument("--domains", type=int, default=250, help="unreliable domains to try")
    parser.add_argument("--per-domain", type=int, default=25)
    parser.add_argument("--min-engagement", type=int, default=2)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    client = BlueskyClient()
    low = load_low_credibility()
    seen: set[str] = set()
    out = Path(args.out)

    print(f"low-credibility domains: {len(low)} listed, trying the {args.domains} most trafficked")
    misleading = collect(ranked_low_domains(args.domains), "misleading", client, low,
                         args.target, args.per_domain, args.min_engagement, seen)
    save(misleading, out)

    print(f"\nhigh-credibility domains: {len(HIGH_CREDIBILITY)}")
    reliable = collect(list(HIGH_CREDIBILITY), "reliable", client, low,
                       args.target, args.per_domain, args.min_engagement, seen)

    everything = misleading + reliable
    save(everything, out)
    accounts = sum(len(c["nodes"]) for c in everything)
    print(f"\n{len(misleading)} misleading and {len(reliable)} reliable cascades, "
          f"{accounts} accounts -> {out}")


if __name__ == "__main__":
    main()
