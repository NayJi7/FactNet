"""FastAPI app for the dashboard. Endpoints just validate and call the pipeline.

    uv run uvicorn factnet.serve.api:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import urllib.error
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from factnet.serve import datasets as datasets_module
from factnet.serve import jobs, quota, samples
from factnet.serve import results as results_module
from factnet.serve.pipeline import run
from factnet.serve.registry import catalogue

app = FastAPI(title="FactNet", version="1.0")

# CORS is only needed in dev (vite on :5173). In prod everything is same-origin,
# so default to the dev servers instead of "*" if FACTNET_CORS_ORIGINS isn't set
_DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
_origins = os.environ.get("FACTNET_CORS_ORIGINS", "")
app.add_middleware(
    CORSMiddleware,
    allow_origins=(["*"] if _origins.strip() == "*" else
                   [o.strip() for o in _origins.split(",") if o.strip()] or _DEV_ORIGINS),
    allow_methods=["GET", "POST"], allow_headers=["*"])

POST_URL = re.compile(r"bsky\.app/profile/([^/]+)/post/([A-Za-z0-9]+)")
MAX_LIVE_ACCOUNTS = 400
FETCH_TTL = 3600                        # 1h
REPLAY_SECONDS = (5.5, 9.0)             # cached results are replayed slowly, see work()


def _visitor(http: Request) -> str:
    return quota.visitor_of(http.client.host if http.client else None,
                            dict(http.headers))


def _site_data(request: VerdictRequest, cascade: dict[str, Any] | None) -> bool:
    """True for the samples / example posts we ship (no quota for those)."""
    if request.sample_id is not None:
        return True
    if cascade is None:
        return request.text.strip() in samples.EXAMPLE_POSTS
    example = samples.record(samples.smallest())
    return example is not None and quota.Cache.key(cascade) == quota.Cache.key(example)


def _take(http: Request, kind: str) -> None:
    try:
        quota.QUOTA.take(_visitor(http), kind)
    except quota.LimitReached as reached:
        raise HTTPException(429, reached.message) from reached


class VerdictRequest(BaseModel):
    text: str = ""
    cascade: dict[str, Any] | None = None
    sample_id: int | None = None
    model: str | None = None
    graph_model: str | None = None
    origin: str = "text"
    observed: int = 100     # % of the cascade visible (early detection)


class FetchRequest(BaseModel):
    url: str = Field(..., description="a bsky.app post URL")


@app.get("/api/models")
def models() -> dict[str, Any]:
    return {"content": catalogue("content"), "graph": catalogue("graph")}


@app.get("/api/samples")
def sample_list() -> dict[str, Any]:
    return {"samples": samples.summaries()}


@app.get("/api/samples/example")
def sample_record() -> dict[str, Any]:
    """Smallest collected cascade, used as the example in the paste box."""
    found = samples.record(samples.smallest())
    if found is None:
        raise HTTPException(404, "no cascade is available to show")
    return {"cascade": found}


@app.get("/api/samples/{index}")
def sample_detail(index: int) -> dict[str, Any]:
    found = samples.detail(index)
    if found is None:
        raise HTTPException(404, "no such sample")
    return found


def _resolve(request: VerdictRequest) -> tuple[dict[str, Any] | None, str]:
    """Shared validation for /verdict and /verdict/stream."""
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

    # otherwise an old model key ends up as a 500
    for value, kind in ((request.model, "content"), (request.graph_model, "graph")):
        if value and value not in {c["key"] for c in catalogue(kind)}:
            raise HTTPException(
                422, f"unknown {kind} model {value!r}; ask /api/models for the list")

    return cascade, origin


@app.post("/api/verdict")
def verdict(request: VerdictRequest, http: Request) -> dict[str, Any]:
    cascade, origin = _resolve(request)
    free = _site_data(request, cascade)
    cache = quota.SITE_DATA if free else quota.READINGS
    key = "plain:" + quota.Cache.key(request.model_dump())
    cached = cache.get(key)
    if cached is not None:
        return cached
    if not free:
        _take(http, "reading")
    trace = run(text=request.text, cascade=cascade,
                content_model=request.model, graph_model=request.graph_model,
                origin=origin, observed=request.observed).to_dict()
    cache.put(key, trace)
    return trace


def _label(request: VerdictRequest, cascade: dict[str, Any] | None) -> tuple[str, str]:
    """(tab, label) for the job, so the front can restore the right view on reload."""
    if cascade is not None:
        domain = cascade.get("source_domain") or cascade.get("source_handle") or ""
        accounts = len(cascade.get("nodes", []))
        tab = "cascades" if request.sample_id is not None else "verdict"
        return tab, f"{domain or 'a cascade'}, {accounts} accounts"
    words = request.text.split()
    short = " ".join(words[:9]) + ("..." if len(words) > 9 else "")
    return "verdict", short or "a post"


def _stream(job: jobs.Job, start: int = 0) -> StreamingResponse:
    def frames() -> Iterator[str]:
        yield f"event: job\ndata: {json.dumps(job.describe())}\n\n"
        for name, payload in job.follow(start):
            yield f"event: {name}\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(frames(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/verdict/stream")
def verdict_stream(request: VerdictRequest, http: Request) -> StreamingResponse:
    """Run a reading as a job and stream its steps (SSE).

    Only one job runs at a time; if one is already running we just follow it.
    """
    cascade, origin = _resolve(request)
    tab, label = _label(request, cascade)
    free = _site_data(request, cascade)
    cache = quota.SITE_DATA if free else quota.READINGS
    key = "stream:" + quota.Cache.key(request.model_dump())
    cached = cache.get(key)
    # joining a running job is free
    if cached is None and not free and jobs.REGISTRY.running() is None:
        _take(http, "reading")
    job, is_new = jobs.REGISTRY.start(tab, label)
    if not is_new:
        return _stream(job)

    def work() -> None:
        if cached is not None:
            # fake the wait so cached results don't pop instantly
            steps = cached["steps"]
            weights = [random.uniform(0.6, 1.6) for _ in range(len(steps) + 1)]
            total = random.uniform(*REPLAY_SECONDS)
            for step, w in zip(steps, weights[:-1], strict=True):
                time.sleep(total * w / sum(weights))
                job.append(step)
            time.sleep(total * weights[-1] / sum(weights))
            job.finish(summary=cached["summary"])
            return
        steps: list[dict[str, Any]] = []

        def on_step(s) -> None:
            steps.append(asdict(s))
            job.append(steps[-1])

        try:
            trace = run(text=request.text, cascade=cascade,
                        content_model=request.model, graph_model=request.graph_model,
                        origin=origin, observed=request.observed, on_step=on_step)
            summary = trace.summary()
            cache.put(key, {"steps": steps, "summary": summary})
            job.finish(summary=summary)
        except Exception as error:
            job.finish(error=f"{type(error).__name__}: {error}")

    threading.Thread(target=work, daemon=True).start()
    return _stream(job)


@app.get("/api/jobs/current")
def current_job() -> dict[str, Any]:
    """Lets a freshly loaded page rejoin a job that's still running."""
    job = jobs.REGISTRY.running()
    return {"job": job.describe() if job else None}


@app.get("/api/jobs/{job_id}/stream")
def job_stream(job_id: str, since: int = 0) -> StreamingResponse:
    job = jobs.REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(404, "no such reading, it may have finished long ago")
    return _stream(job, start=max(0, since))


def validate_cascade(cascade: dict) -> str | None:
    """Returns an error message, or None if the cascade looks ok."""
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
def fetch(request: FetchRequest, http: Request) -> dict[str, Any]:
    """Build a cascade from a bsky.app post URL.

    Profiles have to be fetched separately: the post views don't include the
    follower/post counts, and without them all features are 0 (we had that bug
    in the first collection).
    """
    match = POST_URL.search(request.url)
    if not match:
        raise HTTPException(422, "expected a URL like "
                                 "https://bsky.app/profile/<handle>/post/<id>")
    handle, rkey = match.groups()

    key = f"{handle.lower()}/{rkey}"
    hit = quota.FETCHES.get(key)
    if hit is not None and time.time() - hit[0] < FETCH_TTL:
        return {"cascade": hit[1]}
    _take(http, "fetch")

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
        trim(cascade)   # before enrich, no point fetching profiles we drop
        enrich_profiles(client, cascade)
    except HTTPException:
        raise
    except urllib.error.HTTPError as error:
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
    except Exception as error:   # network / auth / timeout
        raise HTTPException(
            502, f"Bluesky could not be reached ({type(error).__name__}). "
                 "The collected cascades work offline.") from error

    quota.FETCHES.put(key, (time.time(), cascade))
    return {"cascade": cascade}


def trim(cascade: dict) -> None:
    """Keep the first MAX_LIVE_ACCOUNTS nodes so a live fetch stays fast."""
    if len(cascade["nodes"]) <= MAX_LIVE_ACCOUNTS:
        return
    cascade["nodes"] = cascade["nodes"][:MAX_LIVE_ACCOUNTS]
    keep = {n["did"] for n in cascade["nodes"]}
    cascade["edges"] = [e for e in cascade["edges"]
                        if e["source"] in keep and e["target"] in keep]
    cascade["_truncated"] = MAX_LIVE_ACCOUNTS


def enrich_profiles(client, cascade: dict, workers: int = 6) -> int:
    """Fetch profile counters in batches of 25, in parallel.

    Sequentially it took ~40s for a big cascade. A failed batch is skipped.
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
    return results_module.payload()


@app.get("/api/data")
def data() -> dict[str, Any]:
    return datasets_module.payload()


@app.get("/api/quota")
def usage(http: Request) -> dict[str, Any]:
    return quota.QUOTA.status(_visitor(http))


@app.get("/api/health")
def health() -> dict[str, Any]:
    content = catalogue("content", only_available=True)
    graph = catalogue("graph", only_available=True)
    return {"ok": bool(content and graph),
            "content_models": len(content), "graph_models": len(graph),
            "samples": len(samples.load())}


# serve the built front (web/dist) if it exists, i.e. in the docker image.
# must stay at the end so the catch-all doesn't shadow /api routes
_dist = Path(os.environ.get(
    "FACTNET_WEB_DIST",
    Path(__file__).resolve().parents[3] / "web" / "dist"))

if (_dist / "index.html").is_file():
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        candidate = (_dist / path).resolve()
        # no ../ escapes
        if path and candidate.is_file() and candidate.is_relative_to(_dist.resolve()):
            return FileResponse(candidate)
        return FileResponse(_dist / "index.html")
