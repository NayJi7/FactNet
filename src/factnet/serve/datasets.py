"""Data tab: the 3 corpora (UPFD, LIAR, our Bluesky sample) and how we labelled ours.

Bluesky numbers are computed from the file, not hardcoded.
"""

from __future__ import annotations

import json
import statistics as st
from functools import lru_cache
from pathlib import Path
from typing import Any

COLLECTED = (Path(__file__).resolve().parents[3] / "data" / "raw" / "bluesky"
             / "cascades-by-source.jsonl")

BENCHMARKS = [
    {
        "name": "UPFD PolitiFact",
        "role": "propagation, primary benchmark",
        "size": "314 cascades",
        "labels": "fact-checkers, through FakeNewsNet",
        "note": "Small, political, and the corpus on which the integration gain "
                "is measured.",
    },
    {
        "name": "UPFD GossipCop",
        "role": "propagation, second benchmark",
        "size": "5,464 cascades",
        "labels": "fact-checkers, through FakeNewsNet",
        "note": "Larger and easier. It is where the 0.920 headline comes from, "
                "and where cascade shapes run opposite to PolitiFact.",
    },
    {
        "name": "LIAR",
        "role": "content",
        "size": "10,240 train / 1,267 test statements",
        "labels": "PolitiFact rulings, collapsed to two classes",
        "note": "Short, context-free claims. Its difficulty is the reason the "
                "content ceiling sits near 0.63, and reading a full article "
                "instead lifts the same approach to 0.805.",
    },
]

PROTOCOL = [
    "Posts are found by searching for the domain of an outlet whose factual "
    "record is already rated, not by keyword: an earlier keyword pass returned "
    "87 reliable claims against 6 misleading ones, which is unusable.",
    "Low-credibility domains come from the Iffy Index, restricted to the two "
    "lowest factual tiers. The 1,288 domains it rates 'mixed' are excluded, "
    "since treating a link to those as misleading would be wrong article by "
    "article.",
    "Only a link carried as an attached preview counts. Bluesky turns a bare "
    "domain typed in the text into a link, which would let mere mentions pass.",
    "The label describes the outlet, not the post. An account debunking an "
    "unreliable article is scored like one relaying it, and that residual noise "
    "is a known property of source-level labelling.",
]


@lru_cache(maxsize=1)
def collected_summary() -> dict[str, Any]:
    if not COLLECTED.exists():
        return {"available": False}
    cascades = [json.loads(line) for line in COLLECTED.open(encoding="utf-8")
                if line.strip()]
    by_class: dict[str, list[int]] = {"misleading": [], "reliable": []}
    domains: dict[str, int] = {}
    slots = [0] * 10
    accounts = 0
    for cascade in cascades:
        label = "reliable" if cascade.get("label") == 1 else "misleading"
        by_class[label].append(len(cascade.get("nodes", [])))
        domain = cascade.get("source_domain", "")
        if domain:
            domains[domain] = domains.get(domain, 0) + 1
        for node in cascade.get("nodes", []):
            accounts += 1
            for index, value in enumerate(node.get("profile", [])):
                if value:
                    slots[index] += 1

    names = ("verified", "geo", "followers", "follows", "listed", "posts",
             "age", "handle length", "name length", "description length")
    return {
        "available": True,
        "cascades": len(cascades),
        "accounts": accounts,
        "domains": len(domains),
        "top_domains": sorted(domains.items(), key=lambda kv: -kv[1])[:8],
        "classes": [
            {"name": name, "cascades": len(sizes),
             "mean_accounts": round(st.fmean(sizes), 1) if sizes else 0,
             "median_accounts": round(st.median(sizes), 1) if sizes else 0}
            for name, sizes in by_class.items()
        ],
        "feature_coverage": [
            {"feature": name, "populated": count,
             "share": round(count / accounts, 3) if accounts else 0}
            for name, count in zip(names, slots, strict=True)
        ],
    }


def payload() -> dict[str, Any]:
    return {"benchmarks": BENCHMARKS, "protocol": PROTOCOL,
            "collected": collected_summary()}
