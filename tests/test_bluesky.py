"""Cascade assembly from Bluesky payloads, checked without network access."""

import json

import pytest

from factnet.ingestion.bluesky import collect_cascade, profile_features

SOURCE = {"did": "did:src", "handle": "source.bsky.social", "followersCount": 1200,
          "followsCount": 300, "postsCount": 900, "createdAt": "2024-01-01T00:00:00Z",
          "displayName": "Source", "description": "news account"}
RESHARER = {"did": "did:re", "handle": "resharer.bsky.social", "followersCount": 10}
REPLIER = {"did": "did:rep", "handle": "replier.bsky.social", "followersCount": 5}
NESTED = {"did": "did:nest", "handle": "nested.bsky.social"}

POST = {"uri": "at://did:src/app.bsky.feed.post/1", "author": SOURCE,
        "record": {"text": "a claim", "createdAt": "2026-07-30T10:00:00Z"},
        "repostCount": 1, "replyCount": 2, "likeCount": 7}


class FakeClient:
    """Serves fixed payloads in place of the public AppView."""

    def reposted_by(self, uri, limit=100):
        return [RESHARER, SOURCE]  # the source itself must not be duplicated

    def thread(self, uri, depth=6):
        return {"post": POST, "replies": [
            {"post": {"author": REPLIER}, "replies": [{"post": {"author": NESTED}}]},
            {"post": {"author": {}}},  # malformed entries are skipped
        ]}


def test_cascade_nodes_and_edges():
    cascade = collect_cascade(FakeClient(), POST)
    dids = {n["did"] for n in cascade["nodes"]}
    assert dids == {"did:src", "did:re", "did:rep", "did:nest"}
    assert {n["kind"] for n in cascade["nodes"]} == {"source", "repost", "reply"}
    # the nested reply hangs from its own parent, not from the source
    assert {"source": "did:rep", "target": "did:nest", "kind": "reply"} in cascade["edges"]
    assert len(cascade["edges"]) == 3
    assert cascade["text"] == "a claim"
    assert cascade["label"] is None


def test_profile_features_layout():
    features = profile_features(SOURCE)
    assert len(features) == 10
    assert features[2] == 1200.0  # followers
    assert features[6] > 0.0      # account age in days
    assert features[7] == float(len("source.bsky.social"))


def test_profile_features_tolerate_missing_fields():
    assert profile_features({}) == [0.0] * 10
