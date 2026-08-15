"""Collect real propagation cascades from Bluesky.

Bluesky exposes an open read API, which makes it the practical source of
freshly collected data for this project: a post, the accounts that reposted or
replied to it, and their public profiles are enough to rebuild a propagation
cascade in the same shape as the benchmark ones.

Only public endpoints and public content are read; nothing is written, and the
collected records keep the account handle (already public) with the profile
counters needed for the node features. Reads are authenticated with an
application password when one is supplied, which is required from hosted
runners.

    export BSKY_IDENTIFIER=handle.bsky.social BSKY_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
    uv run python -m factnet.ingestion.bluesky --query "vaccine" --posts 25
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BASE = "https://public.api.bsky.app/xrpc"
AUTH_BASE = "https://bsky.social/xrpc"
USER_AGENT = "UM-FactNet-research/0.1 (academic study of misinformation propagation)"
OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "bluesky"


class BlueskyClient:
    """Read-only client over the AppView endpoints.

    The unauthenticated endpoints answer only to residential addresses; from a
    hosted runner they return 403, so a session is opened with an application
    password (created in the Bluesky settings, revocable, never the account
    password) whenever credentials are supplied.
    """

    def __init__(self, base: str | None = None, pause: float = 0.4, retries: int = 3,
                 identifier: str | None = None, password: str | None = None):
        self.identifier = identifier or os.environ.get("BSKY_IDENTIFIER")
        self.password = password or os.environ.get("BSKY_APP_PASSWORD")
        self.token: str | None = None
        if self.identifier and self.password:
            self.base = base or AUTH_BASE
            self._login()
        else:
            self.base = base or BASE
        self.pause, self.retries = pause, retries

    def _login(self) -> None:
        payload = json.dumps({"identifier": self.identifier,
                              "password": self.password}).encode()
        request = urllib.request.Request(
            f"{AUTH_BASE}/com.atproto.server.createSession", data=payload,
            headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            self.token = json.loads(response.read())["accessJwt"]

    def get(self, endpoint: str, **params: Any) -> dict:
        # doseq repeats a parameter given as a list (actors=a&actors=b), which is
        # how the AT protocol takes plural arguments; without it a list would be
        # serialised as its Python repr and the endpoint answers 400
        url = f"{self.base}/{endpoint}?{urllib.parse.urlencode(params, doseq=True)}"
        headers = {"User-Agent": USER_AGENT}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, headers=headers)
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                    payload = json.loads(response.read())
                time.sleep(self.pause)  # stay well under the public rate limit
                return payload
            except urllib.error.HTTPError as err:
                # rate limiting and transient gateway failures are worth retrying
                if err.code in (429, 500, 502, 503, 504) and attempt < self.retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
            except urllib.error.URLError:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"unreachable: {endpoint}")

    def search_posts(self, query: str, limit: int = 25, sort: str = "top") -> list[dict]:
        """Search public posts; ``sort='top'`` favours posts that were engaged with."""
        return self.get("app.bsky.feed.searchPosts", q=query, limit=min(limit, 100),
                        sort=sort).get("posts", [])

    def reposted_by(self, uri: str, limit: int = 100) -> list[dict]:
        return self.get("app.bsky.feed.getRepostedBy", uri=uri, limit=limit).get("repostedBy", [])

    def thread(self, uri: str, depth: int = 6) -> dict:
        return self.get("app.bsky.feed.getPostThread", uri=uri, depth=depth).get("thread", {})

    # The views returned beside posts (ProfileView, ProfileViewBasic) carry no
    # follower, following or post counters; only the detailed view does, and it
    # is reachable solely through this endpoint, 25 accounts at a time.
    PROFILE_BATCH = 25

    def profiles(self, dids: list[str]) -> list[dict]:
        """Detailed profiles, the only view carrying the account counters."""
        return self.get("app.bsky.actor.getProfiles",
                        actors=dids[:self.PROFILE_BATCH]).get("profiles", [])


def _age_days(created_at: str | None) -> float:
    if not created_at:
        return 0.0
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return max(0.0, (datetime.now(UTC) - created).days)


def profile_features(actor: dict) -> list[float]:
    """Account features mirroring the UPFD ``profile`` set, as far as Bluesky exposes.

    Bluesky has no verification or geolocation flags and no favourites counter,
    so the vector keeps the six counters that do exist plus two text lengths;
    the two missing slots are held at zero to preserve the layout.
    """
    return [
        0.0,  # verified: not exposed by Bluesky
        0.0,  # geo enabled: not exposed by Bluesky
        float(actor.get("followersCount", 0)),
        float(actor.get("followsCount", 0)),
        0.0,  # listed count: no equivalent
        float(actor.get("postsCount", 0)),
        _age_days(actor.get("createdAt")),
        float(len(actor.get("handle", ""))),
        float(len(actor.get("displayName") or "")),
        float(len(actor.get("description") or "")),
    ]


def _walk_replies(node: dict, parent: str, nodes: dict, edges: list) -> None:
    for reply in node.get("replies") or []:
        post = reply.get("post") or {}
        author = post.get("author") or {}
        did = author.get("did")
        if not did:
            continue
        nodes.setdefault(did, {"did": did, "handle": author.get("handle", ""),
                               "kind": "reply", "profile": profile_features(author),
                               "display_name": author.get("displayName", "")})
        edges.append({"source": parent, "target": did, "kind": "reply"})
        _walk_replies(reply, did, nodes, edges)


def collect_cascade(client: BlueskyClient, post: dict, thread: dict | None = None) -> dict:
    """One post with the accounts that reposted or replied to it.

    A caller that has already read the thread passes it in: fetching it a second
    time costs several seconds and returns the same conversation.
    """
    record = post.get("record") or {}
    author = post.get("author") or {}
    uri = post["uri"]
    root = author.get("did", uri)

    nodes: dict[str, dict] = {
        root: {"did": root, "handle": author.get("handle", ""), "kind": "source",
               "profile": profile_features(author),
               "display_name": author.get("displayName", ""),
               "text": record.get("text", ""),
               "created_at": record.get("createdAt", ""),
               "likes": int(post.get("likeCount", 0)),
               "replies": int(post.get("replyCount", 0)),
               "reposts": int(post.get("repostCount", 0))}
    }
    edges: list[dict] = []

    for account in client.reposted_by(uri):
        did = account.get("did")
        if not did or did in nodes:
            continue
        nodes[did] = {"did": did, "handle": account.get("handle", ""), "kind": "repost",
                      "profile": profile_features(account),
                      "display_name": account.get("displayName", "")}
        edges.append({"source": root, "target": did, "kind": "repost"})

    thread = thread if thread is not None else client.thread(uri)
    if thread:
        _walk_replies(thread, root, nodes, edges)

    return {
        "uri": uri,
        "text": record.get("text", ""),
        "created_at": record.get("createdAt", ""),
        "source_handle": author.get("handle", ""),
        "repost_count": int(post.get("repostCount", 0)),
        "reply_count": int(post.get("replyCount", 0)),
        "like_count": int(post.get("likeCount", 0)),
        "nodes": list(nodes.values()),
        "edges": edges,
        "label": None,  # filled during annotation
    }


def collect(queries: list[str], posts_per_query: int = 25,
            client: BlueskyClient | None = None, min_engagement: int = 0,
            verbose: bool = False, on_query=None) -> list[dict]:
    """Search each query, then rebuild a cascade for the engaged posts.

    Most posts are never reshared, and a cascade of one node teaches nothing, so
    posts below ``min_engagement`` reposts plus replies are skipped before any
    further request is spent on them.
    """
    client = client or BlueskyClient()
    cascades, seen = [], set()
    for query in queries:
        found = client.search_posts(query, limit=posts_per_query)
        kept = [p for p in found
                if int(p.get("repostCount", 0)) + int(p.get("replyCount", 0)) >= min_engagement]
        if verbose:
            print(f"  {query!r}: {len(found)} posts, {len(kept)} above the engagement floor")
        for post in kept:
            uri = post.get("uri")
            if not uri or uri in seen:
                continue
            seen.add(uri)
            try:
                cascade = collect_cascade(client, post)
            except Exception as err:  # one unavailable post must not end the run
                if verbose:
                    print(f"    skipped {uri.rsplit('/', 1)[-1]}: {type(err).__name__}")
                continue
            cascade["query"] = query
            cascades.append(cascade)
        if on_query is not None:
            on_query(query, cascades)
    return cascades


def save(cascades: list[dict], path: Path) -> Path:
    """Write the file atomically, so a reader or a crash never sees it half written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for cascade in cascades:
            handle.write(json.dumps(cascade, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", required=True,
                        help="search term (repeatable)")
    parser.add_argument("--posts", type=int, default=25, help="posts per query")
    parser.add_argument("--min-engagement", type=int, default=3,
                        help="minimum reposts plus replies to build a cascade")
    parser.add_argument("--out", default=str(OUT_DIR / "cascades.jsonl"))
    parser.add_argument("--identifier", default=None,
                        help="Bluesky handle (or set BSKY_IDENTIFIER)")
    parser.add_argument("--password", default=None,
                        help="app password (or set BSKY_APP_PASSWORD)")
    args = parser.parse_args()

    client = BlueskyClient(identifier=args.identifier, password=args.password)
    out = Path(args.out)
    # write after every query, so an interrupted run keeps what it collected
    cascades = collect(args.query, args.posts, client=client,
                       min_engagement=args.min_engagement, verbose=True,
                       on_query=lambda _q, done: save(done, out))
    path = save(cascades, out)
    sizes = [len(c["nodes"]) for c in cascades] or [0]
    print(f"collected {len(cascades)} cascades -> {path}")
    print(f"  accounts per cascade: min {min(sizes)}  mean {sum(sizes)/len(sizes):.1f}"
          f"  max {max(sizes)}")


if __name__ == "__main__":
    main()
