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


def test_cascade_to_pyg_puts_source_first_and_keeps_edges():
    from factnet.ingestion.to_graph import cascade_to_pyg, normalise_profile

    cascade = collect_cascade(FakeClient(), POST)
    cascade["label"] = 1
    data = cascade_to_pyg(cascade)

    assert data.num_nodes == 4
    assert data.x.shape == (4, 10)
    assert data.edge_index.shape == (2, 3)
    assert int(data.y) == 1
    # the source account is node 0, as in the benchmark cascades
    expected = normalise_profile(profile_features(SOURCE))
    assert data.x[0].tolist() == pytest.approx(expected, rel=1e-5)


def test_normalise_profile_compresses_counters():
    from factnet.ingestion.to_graph import normalise_profile

    raw = profile_features({"followersCount": 1_000_000, "handle": "a" * 400})
    scaled = normalise_profile(raw)
    assert 0.0 < scaled[2] < 1.0   # follower count compressed
    assert scaled[7] == 1.0        # over-long text saturates instead of exploding


def test_checkworthiness_separates_claims_from_chatter():
    from factnet.nlp.checkworthy import explain, score

    claim = ("The Ohio Supreme Court has nullified the $650 million judgement "
             "that multiple counties won against pharmacy chains.")
    chatter = "lol I think the chemtrails will continue until wokeness improves"

    assert score(claim) > score(chatter)
    assert score(claim) >= 0.5
    assert score(chatter) == 0.0
    assert score("ok") == 0.0            # too short to assert anything
    assert explain(claim)["institution"] is True
    assert explain(chatter)["opinion"] is True


def test_checkworthiness_stays_in_range():
    from factnet.nlp.checkworthy import score

    loaded = ("According to a 2024 government study, the agency reported that 90 percent "
              "of cases increased after the ban was signed into law.")
    assert 0.0 <= score(loaded) <= 1.0
    assert 0.0 <= score("I feel like maybe?") <= 1.0


def test_cohen_kappa_and_consensus():
    from factnet.ingestion.annotate import cohen_kappa, consensus

    assert cohen_kappa([(1, 1), (0, 0), (1, 1), (0, 0)]) == pytest.approx(1.0)
    assert cohen_kappa([(1, 0), (0, 1), (1, 0), (0, 1)]) < 0.0   # worse than chance
    assert cohen_kappa([]) != cohen_kappa([])                    # nan for no overlap

    assert consensus({"annotations": {"a": 1, "b": 1}}) == 1
    assert consensus({"annotations": {"a": 1, "b": 0}}) is None  # disagreement -> unusable
    assert consensus({"annotations": {"a": 1, "b": None}}) == 1  # a skip does not veto
    assert consensus({}) is None


def test_anonymise_removes_identities_and_keeps_structure():
    from factnet.ingestion.anonymise import anonymise

    cascade = collect_cascade(FakeClient(), POST)
    cascade["query"] = "measles"
    exported = anonymise([cascade, cascade])   # same accounts in both records

    first, second = exported
    assert first["cascade_id"] == "C0001"
    dumped = json.dumps(exported)
    for identity in ("did:src", "did:re", "source.bsky.social", "resharer.bsky.social"):
        assert identity not in dumped          # no handle, no DID survives
    assert "uri" not in first                  # the post URI embeds the author DID
    assert first["text"] == "a claim"          # the claim itself is kept
    assert len(first["nodes"]) == 4 and len(first["edges"]) == 3
    # pseudonyms are stable across the whole sample, so a recurring account stays one account
    assert {n["account"] for n in first["nodes"]} == {n["account"] for n in second["nodes"]}


def test_anonymise_scrubs_handles_cited_inside_the_text():
    from factnet.ingestion.anonymise import scrub_mentions

    scrubbed = scrub_mentions("As @gavinnewsom.bsky.social said, see nytimes.com and did:plc:abc123")
    assert "gavinnewsom" not in scrubbed
    assert "did:plc:abc123" not in scrubbed
    assert scrubbed.startswith("As @account said")
    assert scrub_mentions("a plain claim about vaccines") == "a plain claim about vaccines"
