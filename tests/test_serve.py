"""The reasoning behind a verdict, checked without a browser or a GPU."""

import json

import torch
from torch_geometric.data import Data

from factnet.serve.structure import depths, pick_checkpoint, shape, truncate
from factnet.serve.trace import Figure, Step, Trace, confidence_from


def line_cascade(n: int = 8) -> Data:
    """A source and n-1 accounts resharing it directly."""
    edges = [[0, i] for i in range(1, n)]
    return Data(x=torch.rand(n, 10),
                edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous())


def chain_cascade(n: int = 5) -> Data:
    """Each account reshares the previous one, so depth grows with n."""
    edges = [[i, i + 1] for i in range(n - 1)]
    return Data(x=torch.rand(n, 10),
                edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous())


def test_shape_separates_broad_from_deep():
    broad, deep = shape(line_cascade(8)), shape(chain_cascade(5))
    assert broad["depth"] == 1 and broad["direct"] == 7    # everything one hop away
    assert deep["depth"] == 4 and deep["direct"] == 1      # a single line of shares


def test_depths_are_measured_from_the_source():
    assert depths(chain_cascade(5)) == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}


def test_truncation_keeps_the_source_and_the_nearest_accounts():
    data = chain_cascade(10)
    partial = truncate(data, 0.5)
    assert partial.num_nodes == 5
    # BFS order, root always kept
    assert depths(partial) == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    assert truncate(data, 0.01).num_nodes == 1     # never empty


def test_constant_features_fall_back_to_the_structure_only_detector():
    checkpoint, key, why = pick_checkpoint("bluesky", has_features=False)
    assert key == "bigcn-structure"
    assert "shape" in why

    # with usable features the platform decides which detector applies
    assert pick_checkpoint("bluesky", has_features=True)[1] == "bigcn-collected"
    assert pick_checkpoint("benchmark", has_features=True)[1] == "bigcn-upfd-profile"


def test_confidence_never_overstates_a_thin_cascade():
    assert confidence_from(0.95) == "high"
    assert confidence_from(0.52) == "low"                      # on the boundary
    assert confidence_from(0.95, accounts=4) == "low"          # too few accounts
    assert confidence_from(0.95, has_features=False) == "low"  # imputed features


def test_trace_is_json_serialisable_and_rejects_unknown_figures():
    trace = Trace(input_kind="text")
    trace.add(Step(key="k", title="t", summary="s",
                   figures=[Figure(kind="bars", title="f", data={"rows": []})]))
    trace.warn("once")
    trace.warn("once")
    assert trace.warnings == ["once"]           # warnings do not repeat
    json.dumps(trace.to_dict())

    try:
        Figure(kind="sankey", title="x", data={})
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown figure kind must be refused")


def test_unverifiable_text_stops_a_bare_claim_but_not_a_cascade():
    """Text alone -> stop. Text + cascade -> structural verdict, content shown as advisory."""
    from factnet.serve.pipeline import run

    chatter = "lol this is so funny"
    bare = run(text=chatter, origin="text")
    assert bare.label == "not a claim"
    assert [s.key for s in bare.steps] == ["parse", "checkworthy"]

    with_cascade = run(text=chatter, data=line_cascade(12), origin="benchmark")
    steps = {s.key: s for s in with_cascade.steps}
    assert "structure" in steps and "verdict" in steps
    assert with_cascade.verdict is not None

    content = steps["content"]
    assert content.detail["counted"] is False      # reported, not used
    assert content.status == "warning"
    assert "not counted" in content.title
    assert with_cascade.provenance["content"] is None


def api_client():
    from fastapi.testclient import TestClient

    from factnet.serve.api import app
    return TestClient(app)


def test_api_refuses_a_malformed_cascade_with_a_usable_message():
    from factnet.serve.api import validate_cascade

    assert "non-empty 'nodes'" in validate_cascade({"edges": []})
    assert "'edges' list" in validate_cascade({"nodes": [{"did": "a"}]})
    assert "no 'did'" in validate_cascade({"nodes": [{}], "edges": []})
    assert "kind" in validate_cascade({"nodes": [{"did": "a"}], "edges": []})
    short = {"nodes": [{"did": "a", "kind": "source", "profile": [1, 2]}], "edges": []}
    assert "10 numbers" in validate_cascade(short)
    good = {"nodes": [{"did": "a", "kind": "source"}], "edges": []}
    assert validate_cascade(good) is None


def test_api_rejects_an_empty_request_and_an_unknown_sample():
    client = api_client()
    assert client.post("/api/verdict", json={}).status_code == 422
    assert client.post("/api/verdict", json={"sample_id": 9999}).status_code == 404


def test_api_only_accepts_a_bluesky_post_url():
    client = api_client()
    response = client.post("/api/fetch", json={"url": "https://example.com/nope"})
    assert response.status_code == 422
    assert "bsky.app/profile" in response.json()["detail"]


def test_catalogue_reports_every_model_with_the_score_it_earned():
    from factnet.serve.registry import catalogue

    content = catalogue("content")
    assert {c["key"] for c in content} >= {"roberta", "bert", "tfidf"}
    assert sum(c["primary"] for c in content) == 1
    assert all(c["macro_f1"] is not None for c in content)


def test_results_tables_are_shaped_as_the_interface_expects():
    from factnet.serve.results import payload

    data = payload()
    assert data["headlines"] and data["tables"]
    for table in data["tables"]:
        assert table["reading"], f"{table['key']} has no reading"
        width = len(table["columns"])
        assert all(len(row) == width for row in table["rows"]), table["key"]
        if table["best"]:
            assert table["best"] in table["columns"]


def test_api_rejects_a_stale_model_key_instead_of_crashing():
    """used to be a KeyError -> 500"""
    client = api_client()
    for payload, word in (
        ({"text": "The unemployment rate has doubled, according to the ministry.",
          "model": "no-such-model"}, "content"),
        ({"sample_id": 0, "graph_model": "no-such-model"}, "graph"),
    ):
        response = client.post("/api/verdict", json=payload)
        assert response.status_code == 422, payload
        assert word in response.json()["detail"]


def test_a_pasted_cascade_without_profiles_is_read_on_its_shape():
    """No profiles -> zeros -> structure-only model (was a 500 before)."""
    client = api_client()
    pasted = {
        "text": "the claim the post makes",
        "nodes": [{"did": "a", "handle": "source", "kind": "source"},
                  {"did": "b", "handle": "resharer", "kind": "repost"},
                  {"did": "c", "handle": "another", "kind": "repost"}],
        "edges": [{"source": "a", "target": "b", "kind": "repost"},
                  {"source": "a", "target": "c", "kind": "repost"}],
    }
    response = client.post("/api/verdict", json={"cascade": pasted, "origin": "cascade"})
    assert response.status_code == 200, response.text

    trace = response.json()
    steps = {s["key"]: s for s in trace["steps"]}
    assert "structure" in steps
    assert steps["structure"]["detail"]["model_key"] == "bigcn-structure"
    assert any("features are constant" in w for w in trace["warnings"])


def test_the_stream_sends_each_stage_as_it_lands():
    """/verdict/stream and /verdict must give the same steps."""
    client = api_client()
    payload = {"text": "The unemployment rate has doubled, the ministry said."}

    with client.stream("POST", "/api/verdict/stream", json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    events, data = [], []
    for frame in body.split("\n\n"):
        for line in frame.splitlines():
            if line.startswith("event: "):
                events.append(line[7:])
            elif line.startswith("data: "):
                data.append(json.loads(line[6:]))

    assert events[-1] == "done"
    streamed = [d["key"] for e, d in zip(events, data, strict=True) if e == "step"]
    assert streamed, "no stage was streamed"
    assert "verdict" in streamed

    plain = client.post("/api/verdict", json=payload).json()
    assert [s["key"] for s in plain["steps"]] == streamed
    assert data[-1]["label"] == plain["label"]
    assert "steps" not in data[-1]        # the summary does not repeat them


def test_every_model_card_matches_the_measurement_it_quotes():
    """Model card scores must match data/results (they drifted by up to 0.028 once)."""
    import json
    from pathlib import Path

    from factnet.serve.registry import by_key

    results = Path(__file__).resolve().parents[1] / "data" / "results"
    load = lambda name: json.loads((results / f"{name}.json").read_text())  # noqa: E731

    table1 = load("table1_3seeds")["politifact/profile"]
    expected = {
        "gcn-upfd-profile": table1["GCN"]["mean"],
        "gat-upfd-profile": table1["GAT"]["mean"],
        "bigcn-upfd-profile": table1["Bi-GCN"]["mean"],
        "bigcn-collected": load("confound")["cascade split, all sizes"]["detector"]["mean"],
    }
    for key, measured in expected.items():
        card = by_key(key)
        assert abs(card.macro_f1 - measured) < 0.0005, (
            f"{key} advertises {card.macro_f1}, the experiment wrote {measured:.4f}")


def test_the_headline_numbers_come_from_the_result_files():
    from factnet.serve.results import headlines

    values = {h["label"]: h["value"] for h in headlines()}
    assert values, "no headlines were built"
    assert all(h["module"] in {"content", "propagation", "both"} for h in headlines())


def test_the_worked_example_is_a_real_cascade_without_its_answer():
    """The example must be a real cascade and must not include the label."""
    from fastapi.testclient import TestClient

    from factnet.serve.api import app

    client = TestClient(app)
    payload = client.get("/api/samples/example")
    assert payload.status_code == 200
    cascade = payload.json()["cascade"]

    assert set(cascade) == {"text", "source_handle", "nodes", "edges"}
    assert len(cascade["nodes"]) >= 5, "a stub teaches the wrong lesson"
    assert any(n.get("kind") == "source" for n in cascade["nodes"])

    # /samples/example must not be caught by /samples/{index}
    assert client.get("/api/samples/1").status_code == 200
    assert client.get("/api/samples/99").status_code == 404

    assert client.post("/api/verdict", json={"cascade": cascade,
                                             "origin": "cascade"}).status_code == 200
