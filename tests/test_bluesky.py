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


def test_a_quantity_can_be_asserted_without_a_digit():
    """regression: "unemployment rate has doubled" used to score 0.25 (no digit)"""
    from factnet.nlp.checkworthy import explain, score

    claim = "The unemployment rate has doubled since last year, according to sources online."
    assert score(claim) >= 0.3
    fired = explain(claim)
    assert fired["change"] and fired["measure"] and fired["attribution"]
    assert not fired["quantity"]          # no number in it

    for wording in ("Inflation fell for the third month running, the ministry said.",
                    "Prices tripled after the agency approved the merger."):
        assert score(wording) >= 0.3, wording

    # negatives still out
    for chatter in ("I think this is really funny, guess I'll never understand people",
                    "please someone tell me why my code does not compile, thanks",
                    "Anyone else awake at this hour? I love this city, it is beautiful"):
        assert score(chatter) < 0.3, chatter


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
    # same account -> same pseudonym across records
    assert {n["account"] for n in first["nodes"]} == {n["account"] for n in second["nodes"]}


def test_anonymise_carries_label_provenance_without_exposing_it_to_the_scrub():
    from factnet.ingestion.anonymise import anonymise

    cascade = collect_cascade(FakeClient(), POST)
    keyword_labelled, = anonymise([cascade])
    assert "source_domain" not in keyword_labelled   # no outlet to attribute the label to

    cascade.update(source_domain="infowars.com", source_label="misleading", label=0)
    source_labelled, = anonymise([cascade])
    # the scrubber eats ".com" in text but source_domain must stay intact
    assert source_labelled["source_domain"] == "infowars.com"
    assert source_labelled["source_label"] == "misleading"
    assert source_labelled["label"] == 0


def test_anonymise_scrubs_handles_cited_inside_the_text():
    from factnet.ingestion.anonymise import scrub_mentions

    scrubbed = scrub_mentions(
        "As @gavinnewsom.bsky.social said, see nytimes.com and did:plc:abc123")
    assert "gavinnewsom" not in scrubbed
    assert "did:plc:abc123" not in scrubbed
    assert scrubbed.startswith("As @account said")
    assert scrub_mentions("a plain claim about vaccines") == "a plain claim about vaccines"


def test_slot_baseline_accounts_for_unequal_cascade_sizes(tmp_path):
    """The baseline is the share of account slots, not the share of cascades."""
    from factnet.graph.influence_veracity import account_labels, misleading_share

    # 1 small misleading + 1 big reliable cascade
    cascades = [
        {"label": 0, "nodes": [{"did": f"m{i}"} for i in range(2)] + [{"did": "both"}]},
        {"label": 1, "nodes": [{"did": f"r{i}"} for i in range(8)] + [{"did": "both"}]},
    ]
    path = tmp_path / "cascades.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in cascades), encoding="utf-8")

    per_account, misleading_slots, reliable_slots = account_labels(path)
    assert (misleading_slots, reliable_slots) == (3, 9)
    # a 50/50 split of cascades yields a 0.25 chance of sitting in the misleading one
    assert misleading_slots / (misleading_slots + reliable_slots) == 0.25

    assert misleading_share(per_account["both"]) == 0.5   # one of each
    assert misleading_share(per_account["m0"]) == 1.0
    assert misleading_share(per_account["r0"]) == 0.0


def test_stratified_null_holds_activity_fixed():
    """Shuffling inside equal-activity strata cannot invent an association."""
    from factnet.graph.influence_veracity import stratified_null

    # one account per stratum -> nothing to shuffle
    influences = [0.1, 0.2, 0.3, 0.4]
    shares = [0.0, 0.25, 0.5, 1.0]
    observed, p = stratified_null(influences, shares, strata=[1, 2, 3, 4], rounds=50)
    assert observed == pytest.approx(1.0)
    assert p == pytest.approx(1.0)   # every permutation reproduces the observation

    # one big stratum -> shuffle works
    _, p_free = stratified_null(influences, shares, strata=[1, 1, 1, 1], rounds=200)
    assert p_free > 0.05
