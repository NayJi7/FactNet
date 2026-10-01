"""HTTP surface over the engine.

Thin by design: every endpoint parses its input, calls the pipeline, and hands
back the trace untouched. No reasoning happens here, so that what the browser
shows and what ``pytest`` checks are the same computation.

    uv run uvicorn factnet.serve.api:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from factnet.serve import datasets as datasets_module
from factnet.serve import jobs, samples
from factnet.serve import results as results_module
from factnet.serve.pipeline import run
from factnet.serve.registry import catalogue

app = FastAPI(title="FactNet", version="1.0")

# In development Vite serves the front end from another port, so the browser
# makes cross-origin calls and anything is allowed. A deployment serves both
# from one origin and needs none of that, so the wildcard is narrowed to the
# local dev servers as soon as a built front end is present: a deployment that
# forgets to set FACTNET_CORS_ORIGINS should not end up with the permissive
# default, since the correct answer there is to allow nothing.
_DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
_origins = os.environ.get("FACTNET_CORS_ORIGINS", "")
app.add_middleware(
    CORSMiddleware,
    allow_origins=(["*"] if _origins.strip() == "*" else
                   [o.strip() for o in _origins.split(",") if o.strip()] or _DEV_ORIGINS),
    allow_methods=["GET", "POST"], allow_headers=["*"])

POST_URL = re.compile(r"bsky\.app/profile/([^/]+)/post/([A-Za-z0-9]+)")
MAX_LIVE_ACCOUNTS = 400


class VerdictRequest(BaseModel):
    text: str = ""
    cascade: dict[str, Any] | None = None
    sample_id: int | None = None
    model: str | None = None
    graph_model: str | None = None
    origin: str = "text"
    # the share of the cascade the system may see, for the early-detection view
    observed: int = 100


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


@app.get("/api/samples/example")
def sample_record() -> dict[str, Any]:
    """The shortest collected cascade, stripped to the fields a reader supplies.

    Offered as the worked example behind the paste box. A real record rather
    than a placeholder, because a two-node stub produces a confident number on
    an object with no shape, which teaches the wrong thing about the system.
    """
    found = samples.record(samples.smallest())
    if found is None:
        raise HTTPException(404, "no cascade is available to show")
    return {"cascade": found}


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

    if request.observed not in (20, 40, 60, 80, 100):
        raise HTTPException(422, "observed must be one of 20, 40, 60, 80, 100")

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
                origin=origin, observed=request.observed)
    return trace.to_dict()


def _label(request: VerdictRequest, cascade: dict[str, Any] | None) -> tuple[str, str]:
    """Which view the reading belongs to, and a name for it.

    The interface needs both to put someone back where they were after a
    reload, so they are settled once, here, rather than guessed by the browser.
    """
    if cascade is not None:
        domain = cascade.get("source_domain") or cascade.get("source_handle") or ""
        accounts = len(cascade.get("nodes", []))
        tab = "cascades" if request.sample_id is not None else "verdict"
        return tab, f"{domain or 'a cascade'}, {accounts} accounts"
    words = request.text.split()
    short = " ".join(words[:9]) + ("..." if len(words) > 9 else "")
    return "verdict", short or "a post"


def _stream(job: jobs.Job, start: int = 0) -> StreamingResponse:
    """Follow a reading from a given stage, as server-sent events."""
    def frames() -> Iterator[str]:
        yield f"event: job\ndata: {json.dumps(job.describe())}\n\n"
        for name, payload in job.follow(start):
            yield f"event: {name}\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(frames(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/verdict/stream")
def verdict_stream(request: VerdictRequest) -> StreamingResponse:
    """Start a reading and follow it, stage by stage as each one lands.

    The reading is registered as a job before anything is computed, so that a
    browser which goes away mid-run can find it again instead of losing it.
    Asking while a reading is already under way follows that one rather than
    starting a second: the work is heavy, and nothing here needs two at once.
    """
    cascade, origin = _resolve(request)
    tab, label = _label(request, cascade)
    job, is_new = jobs.REGISTRY.start(tab, label)
    if not is_new:
        return _stream(job)

    def work() -> None:
        try:
            trace = run(text=request.text, cascade=cascade,
                        content_model=request.model, graph_model=request.graph_model,
                        origin=origin, observed=request.observed,
                        on_step=lambda s: job.append(asdict(s)))
            job.finish(summary=trace.summary())
        except Exception as error:                       # reported, never swallowed
            job.finish(error=f"{type(error).__name__}: {error}")

    threading.Thread(target=work, daemon=True).start()
    return _stream(job)


@app.get("/api/jobs/current")
def current_job() -> dict[str, Any]:
    """What is being read right now, if anything.

    A page that has just loaded asks this before showing an idle screen: if a
    reading it started earlier is still going, it rejoins that instead.
    """
    job = jobs.REGISTRY.running()
    return {"job": job.describe() if job else None}


@app.get("/api/jobs/{job_id}/stream")
def job_stream(job_id: str, since: int = 0) -> StreamingResponse:
    """Rejoin a reading, replaying the stages already produced."""
    job = jobs.REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(404, "no such reading, it may have finished long ago")
    return _stream(job, start=max(0, since))


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


# ---------------------------------------------------------------------------
# The built front end, when there is one.
#
# In development Vite serves it and this does nothing. A deployment builds it
# into the image and mounts it here, so the container answers both the API and
# the page from one origin and the reverse proxy in front has nothing to route.
# It is mounted last: every /api route is already registered, so the catch-all
# below can never shadow one.
_dist = Path(os.environ.get(
    "FACTNET_WEB_DIST",
    Path(__file__).resolve().parents[3] / "web" / "dist"))

if (_dist / "index.html").is_file():
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        """Serve a real file when one exists, and the page otherwise."""
        candidate = (_dist / path).resolve()
        # resolve() then compare, so ../ in a request cannot escape the tree
        if path and candidate.is_file() and candidate.is_relative_to(_dist.resolve()):
            return FileResponse(candidate)
        return FileResponse(_dist / "index.html")
