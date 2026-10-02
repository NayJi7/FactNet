"""Second half of the fix: rewrite node features from the profile cache.

Topology was fine, only the features were wrong. Prints per-slot coverage before
and after so we can see how much was zeros.

    uv run python -m factnet.ingestion.rebuild_features
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from factnet.ingestion.bluesky import profile_features

DATA = Path(__file__).resolve().parents[3] / "data" / "raw" / "bluesky"
CASCADES = DATA / "cascades-by-source.jsonl"
PROFILES = DATA / "profiles.json"

SLOT_NAMES = ("verified", "geo", "followers", "follows", "listed",
              "posts", "age_days", "handle_len", "name_len", "desc_len")


def rebuild(cascades: list[dict], profiles: dict[str, dict]) -> tuple[int, int]:
    filled = total = 0
    for cascade in cascades:
        for node in cascade.get("nodes", []):
            total += 1
            found = profiles.get(node.get("did"))
            if found:
                node["profile"] = profile_features(found)
                filled += 1
    return filled, total


def coverage(cascades: list[dict]) -> list[int]:
    """non-zero count per slot"""
    counts = [0] * 10
    for cascade in cascades:
        for node in cascade.get("nodes", []):
            for slot, value in enumerate(node.get("profile", [])):
                if value:
                    counts[slot] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=str(CASCADES))
    parser.add_argument("--profiles", default=str(PROFILES))
    parser.add_argument("--out", default=None, help="defaults to rewriting in place")
    args = parser.parse_args()

    path = Path(args.file)
    cascades = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    profiles = {k: v for k, v in
                json.loads(Path(args.profiles).read_text(encoding="utf-8")).items() if v}

    before = coverage(cascades)
    filled, total = rebuild(cascades, profiles)
    after = coverage(cascades)

    print(f"{len(cascades)} cascades, {total} nodes, {filled} rebuilt "
          f"({filled / total:.1%})\n")
    print(f"  {'slot':<12s} {'before':>9s} {'after':>9s}")
    for name, was, now in zip(SLOT_NAMES, before, after, strict=True):
        mark = "  <-" if now != was else ""
        print(f"  {name:<12s} {was:>9d} {now:>9d}{mark}")

    out = Path(args.out or args.file)
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for cascade in cascades:
            handle.write(json.dumps(cascade, ensure_ascii=False) + "\n")
    tmp.replace(out)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
