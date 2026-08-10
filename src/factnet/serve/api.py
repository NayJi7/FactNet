"""HTTP surface over the engine.

Thin by design: every endpoint parses its input, calls the pipeline, and hands
back the trace untouched. No reasoning happens here, so that what the browser
shows and what ``pytest`` checks are the same computation.

    uv run uvicorn factnet.serve.api:app --reload --port 8000
"""

from __future__ import annotations

import json
import queue
import re
import threading
import urllib.error
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from factnet.serve import datasets as datasets_module
from factnet.serve import results as results_module
from factnet.serve import samples
from factnet.serve.pipeline import run
from factnet.serve.registry import catalogue

app = FastAPI(title="UM-FactNet", version="1.0")

# the front end is served by Vite in development, on another port
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

POST_URL = re.compile(r"bsky\.app/profile/([^/]+)/post/([A-Za-z0-9]+)")
MAX_LIVE_ACCOUNTS = 400


class VerdictRequest(BaseModel):
    text: str = ""
    cascade: dict[str, Any] | None = None
    sample_id: int | None = None
    model: str | None = None
    graph_model: str | None = None
    origin: str = "text"


class FetchRequest(BaseModel):
    url: str = Field(..., description="a bsky.app post URL")


@app.get("/api/models")
def models() -> dict[str, Any]:
    """The selector: every model, with the score it actually earned."""
    return {"content": catalogue("content"), "graph": catalogue("graph")}


@app.get("/api/samples")
def sample_list() -> dict[str, Any]:
    """Cascades kept beside the code, so a demonstration never needs the network."""
    return {"samples": samples.summaries()}


@app.get("/api/samples/{index}")
def sample_detail(index: int) -> dict[str, Any]:
    """One cascade in full, before any model touches it."""
    found = samples.detail(index)
    if found is None:
        raise HTTPException(404, "no such sample")
    return found


def _resolve(request: VerdictRequest) -> tuple[dict[str, Any] | None, str]:
    """Validate a request and settle what is being read, for either endpoint."""
    cascade = request.cascade
    origin = request.origin

    if request.sample_id is not None:
        loaded = samples.load()
        if not 0 <= request.sample_id < len(loaded):
            raise HTTPException(404, "no such sample")
        cascade = loaded[request.sample_id]
        origin = "bluesky"

    if cascade is not None:
        problem = validate_cascade(cascade)
        if problem:
            raise HTTPException(422, problem)

    if not request.text.strip() and cascade is None:
        raise HTTPException(422, "give a text, a cascade, or a sample id")

    # a stale model key must not reach the engine: it would surface as a 500
    # with a traceback in the middle of a demonstration
    for value, kind in ((request.model, "content"), (request.graph_model, "graph")):
        if value and value not in {c["key"] for c in catalogue(kind)}:
            raise HTTPException(
                422, f"unknown {kind} model {value!r}; ask /api/models for the list")

    return cascade, origin


@app.post("/api/verdict")
def verdict(request: VerdictRequest) -> dict[str, Any]:
    cascade, origin = _resolve(request)
    trace = run(text=request.text, cascade=cascade,
                content_model=request.model, graph_model=request.graph_model,
                origin=origin)
    return trace.to_dict()


@app.post("/api/verdict/stream")
def verdict_stream(request: VerdictRequest) -> StreamingResponse:
    """The same reading, sent stage by stage as each one lands.

    A run takes seconds and produces its stages in order, so there is no reason
    to withhold them until the last one finishes. The engine is unchanged: it
    reports each stage to a listener, and this endpoint forwards them.
    """
    cascade, origin = _resolve(request)

    events: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            trace = run(text=request.text, cascade=cascade,
                        content_model=request.model, graph_model=request.graph_model,
                        origin=origin, on_step=lambda s: events.put(("step", asdict(s))))
            events.put(("done", trace.summary()))
        except Exception as error:                       # reported, never swallowed
            events.put(("failed", {"detail": f"{type(error).__name__}: {error}"}))
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def frames() -> Iterator[str]:
        while True:
            item = events.get()
            if item is None:
                return
            name, payload = item
            yield f"event: {name}\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(frames(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def validate_cascade(cascade: dict) -> str | None:
    """Reject a malformed cascade with a message a person can act on."""
    if not isinstance(cascade, dict):
        return "the cascade must be a JSON object"
    nodes = cascade.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return "the cascade needs a non-empty 'nodes' list"
    if not isinstance(cascade.get("edges"), list):
        return "the cascade needs an 'edges' list, empty if there are no shares"
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or "did" not in node:
            return f"node {index} has no 'did'"
        profile = node.get("profile")
        if profile is not None and (not isinstance(profile, list) or len(profile) != 10):
            return (f"node {index}: 'profile' must be a list of 10 numbers, "
                    "or be left out entirely")
    if not any(n.get("kind") == "source" for n in nodes):
        return "one node must carry \"kind\": \"source\""
    return None


@app.post("/api/fetch")
def fetch(request: FetchRequest) -> dict[str, Any]:
    """Rebuild a live cascade from a Bluesky post URL.

    The account counters are fetched separately, because the views returned
    beside posts do not carry them: without this second call the features would
    silently be zeros, which is the bug the collected sample was built with.
    """
    match = POST_URL.search(request.url)
    if not match:
        raise HTTPException(422, "expected a URL like "
                                 "https://bsky.app/profile/<handle>/post/<id>")
    handle, rkey = match.groups()

    try:
        from factnet.ingestion.bluesky import BlueskyClient, collect_cascade
        client = BlueskyClient()
        did = (handle if handle.startswith("did:") else
               client.get("com.atproto.identity.resolveHandle", handle=handle)["did"])
        thread = client.thread(f"at://{did}/app.bsky.feed.post/{rkey}")
        post = thread.get("post") if isinstance(thread, dict) else None
        if not post:
            raise HTTPException(404, "the post could not be read")
        cascade = collect_cascade(client, post, thread)
        # trim before enriching: the counters are fetched twenty-five accounts
        # at a time, and there is no point paying for accounts about to be cut
        trim(cascade)
        enrich_profiles(client, cascade)
    except HTTPException:
        raise
    except urllib.error.HTTPError as error:
        # a refusal from Bluesky is not a network failure, and saying so would
        # send someone with a mistyped link off debugging their connection
        if error.code in (400, 404):
            raise HTTPException(
                404, "Bluesky has no such post. Check the link, or use one of "
                     "the collected cascades.") from error
        if error.code == 429:
            raise HTTPException(
                429, "Bluesky is rate limiting this account. The collected "
                     "cascades work without it.") from error
        raise HTTPException(
            502, f"Bluesky answered {error.code}. The collected cascades work "
                 "offline.") from error
    except Exception as error:                       # network, auth, timeout
        raise HTTPException(
            502, f"Bluesky could not be reached ({type(error).__name__}). "
                 "The collected cascades work offline.") from error

    return {"cascade": cascade}


def trim(cascade: dict) -> None:
    """Cut a live cascade to a size a demonstration can wait for."""
    if len(cascade["nodes"]) <= MAX_LIVE_ACCOUNTS:
        return
    cascade["nodes"] = cascade["nodes"][:MAX_LIVE_ACCOUNTS]
    keep = {n["did"] for n in cascade["nodes"]}
    cascade["edges"] = [e for e in cascade["edges"]
                        if e["source"] in keep and e["target"] in keep]
    cascade["_truncated"] = MAX_LIVE_ACCOUNTS


def enrich_profiles(client, cascade: dict, workers: int = 6) -> int:
    """Fill the account counters, twenty-five accounts per call.

    The calls are independent and each costs well over a second, so running
    them one after another dominated the wait: sixteen batches took nearly forty
    seconds while the person who pasted the link watched nothing happen. They
    are issued concurrently instead, which is a handful of requests against a
    limit measured in thousands per five minutes.

    A batch that fails leaves its accounts as they were rather than aborting the
    rest: a cascade with some counters is more useful than none.
    """
    from concurrent.futures import ThreadPoolExecutor

    from factnet.ingestion.bluesky import profile_features

    dids = [n["did"] for n in cascade["nodes"] if n.get("did")]
    batches = [dids[i:i + client.PROFILE_BATCH]
               for i in range(0, len(dids), client.PROFILE_BATCH)]

    def fetch(batch: list[str]) -> list[dict]:
        try:
            return client.profiles(batch)
        except Exception:
            return []

    detailed: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for found in pool.map(fetch, batches):
            for profile in found:
                if profile.get("did"):
                    detailed[profile["did"]] = profile

    for node in cascade["nodes"]:
        found = detailed.get(node.get("did"))
        if found:
            node["profile"] = profile_features(found)
    return len(detailed)


@app.get("/api/results")
def results() -> dict[str, Any]:
    """What the article measured, so a live run can be read beside it."""
    return results_module.payload()


@app.get("/api/data")
def data() -> dict[str, Any]:
    """The corpora in play, and how the collected labels were obtained."""
    return datasets_module.payload()


@app.get("/api/health")
def health() -> dict[str, Any]:
    content = catalogue("content", only_available=True)
    graph = catalogue("graph", only_available=True)
    return {"ok": bool(content and graph),
            "content_models": len(content), "graph_models": len(graph),
            "samples": len(samples.load())}
