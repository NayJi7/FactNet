"""A reading must outlive the connection that asked for it.

The defect these guard against: reloading the page during a run threw the work
away, and the interface came back with no idea anything had been running. The
person then clicked again and nothing happened, because the view that had to
re-fetch was keyed on a value that had not changed.
"""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from factnet.serve import jobs
from factnet.serve.api import app


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Each test gets its own registry, so ordering cannot matter."""
    previous = jobs.REGISTRY
    jobs.REGISTRY = jobs.Registry()
    yield
    jobs.REGISTRY = previous


def test_a_job_replays_the_stages_a_late_viewer_missed():
    job, is_new = jobs.REGISTRY.start("verdict", "a post")
    assert is_new
    job.append({"key": "parse"})
    job.append({"key": "checkworthy"})

    # a viewer that arrives now must see both, not wait for a third
    seen = []
    follower = job.follow()
    seen.append(next(follower))
    seen.append(next(follower))
    assert [name for name, _ in seen] == ["step", "step"]

    job.finish(summary={"label": "reliable"})
    assert next(follower) == ("done", {"label": "reliable"})


def test_two_viewers_follow_the_same_reading():
    job, _ = jobs.REGISTRY.start("verdict", "a post")
    job.append({"key": "parse"})
    job.finish(summary={"label": "misleading"})

    for _ in range(2):
        events = list(job.follow())
        assert [name for name, _ in events] == ["step", "done"]


def test_a_second_request_joins_the_reading_instead_of_starting_another():
    """Two concurrent runs would double the memory of a process holding 2 GB."""
    first, new_first = jobs.REGISTRY.start("verdict", "a post")
    second, new_second = jobs.REGISTRY.start("cascades", "a cascade")
    assert new_first and not new_second
    assert second is first

    first.finish(summary={})
    third, new_third = jobs.REGISTRY.start("verdict", "another post")
    assert new_third and third is not first


def test_the_registry_reports_what_is_running_and_forgets_it_when_done():
    assert jobs.REGISTRY.running() is None
    job, _ = jobs.REGISTRY.start("cascades", "apnews.com, 581 accounts")
    running = jobs.REGISTRY.running()
    assert running is not None
    assert running.describe()["tab"] == "cascades"
    assert running.describe()["label"] == "apnews.com, 581 accounts"

    job.finish(summary={})
    assert jobs.REGISTRY.running() is None
    # but it stays reachable, so a viewer who returns late still collects it
    assert jobs.REGISTRY.get(job.id) is not None


def test_a_follower_wakes_when_a_stage_lands():
    """A viewer waiting on a slow stage must not miss it."""
    job, _ = jobs.REGISTRY.start("verdict", "a post")
    collected: list = []

    def watch() -> None:
        for name, payload in job.follow():
            collected.append((name, payload))

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    job.append({"key": "parse"})
    job.finish(summary={"label": "reliable"})
    watcher.join(timeout=10)
    assert not watcher.is_alive(), "the follower never saw the reading end"
    assert [name for name, _ in collected] == ["step", "done"]


def test_the_api_reports_no_reading_when_none_is_running():
    with TestClient(app) as client:
        assert client.get("/api/jobs/current").json() == {"job": None}


def test_rejoining_a_reading_that_never_existed_says_so():
    with TestClient(app) as client:
        response = client.get("/api/jobs/nosuchthing/stream")
        assert response.status_code == 404
        assert "finished" in response.json()["detail"]
