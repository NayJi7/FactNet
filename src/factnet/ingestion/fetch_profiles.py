"""Fix for the zero-features bug: fetch real profiles for every collected account.

The first collection built features from ProfileView, which has no counters, and
.get(..., 0) hid it, so 6 slots were 0 everywhere. getProfiles has them (25/call).
Results are cached by DID; dead accounts are cached as null so we don't retry them.
Resumable.

    uv run python -u -m factnet.ingestion.fetch_profiles
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from factnet.ingestion.bluesky import BlueskyClient

DATA = Path(__file__).resolve().parents[3] / "data" / "raw" / "bluesky"
DEFAULT_SOURCES = [DATA / "cascades-by-source.jsonl"]
CACHE = DATA / "profiles.json"
SAVE_EVERY = 20  # batches
DONE = "PROFILE-FETCH-COMPLETE"  # printed at the end, I grep for it


def unique_dids(paths: list[Path]) -> list[str]:
    seen: dict[str, None] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                for node in json.loads(line).get("nodes", []):
                    did = node.get("did")
                    if did:
                        seen.setdefault(did, None)
    return list(seen)


def load_cache(path: Path) -> dict[str, dict | None]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:      # killed mid-write
        return {}


def save_cache(cache: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    tmp.replace(path)


def keep(profile: dict) -> dict:
    return {k: profile.get(k) for k in
            ("did", "handle", "displayName", "description", "createdAt",
             "followersCount", "followsCount", "postsCount")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="*", default=[str(p) for p in DEFAULT_SOURCES])
    parser.add_argument("--cache", default=str(CACHE))
    args = parser.parse_args()

    cache_path = Path(args.cache)
    cache = load_cache(cache_path)
    dids = unique_dids([Path(s) for s in args.sources])
    todo = [d for d in dids if d not in cache]

    print(f"{len(dids)} distinct accounts, {len(cache)} already cached, "
          f"{len(todo)} to fetch", flush=True)
    if not todo:
        print(f"{DONE} {len(cache)} profiles in {cache_path}", flush=True)
        return

    client = BlueskyClient()
    batches = [todo[i:i + client.PROFILE_BATCH]
               for i in range(0, len(todo), client.PROFILE_BATCH)]
    started, failures = time.time(), 0

    for number, batch in enumerate(batches, 1):
        try:
            found = {p["did"]: keep(p) for p in client.profiles(batch) if p.get("did")}
        except Exception as err:
            failures += 1
            print(f"  batch {number}: failed ({type(err).__name__}), skipped", flush=True)
            found = {}
            if failures > 40:     # probably rate limited, stop
                print("too many consecutive failures, stopping early", flush=True)
                break
        else:
            failures = 0
        for did in batch:
            # missing = deleted/suspended -> null
            cache[did] = found.get(did)

        if number % SAVE_EVERY == 0 or number == len(batches):
            save_cache(cache, cache_path)
            done = number * client.PROFILE_BATCH
            rate = done / max(time.time() - started, 1e-6)
            left = (len(todo) - done) / max(rate, 1e-6)
            print(f"  {min(done, len(todo)):>6d}/{len(todo)} accounts "
                  f"({rate:.0f}/s, ~{left / 60:.1f} min left)", flush=True)

    save_cache(cache, cache_path)
    resolved = sum(1 for v in cache.values() if v)
    with_counts = sum(1 for v in cache.values() if v and v.get("followersCount") is not None)
    print(f"cached {len(cache)} accounts: {resolved} resolved, "
          f"{len(cache) - resolved} unavailable, {with_counts} carrying counters",
          flush=True)
    print(f"{DONE} {len(cache)} profiles in {cache_path}", flush=True)


if __name__ == "__main__":
    main()
