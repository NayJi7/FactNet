"""A reading outlives the connection that asked for it.

The engine used to stream its stages straight down the request that started it,
which meant the reading belonged to a browser tab. Reloading the page halfway
through threw the work away: the thread finished, nobody was listening, and the
interface came back with no idea that anything had ever been running. A person
who reloaded during a cascade read then clicked again saw nothing happen.

So a reading is a job here, held by the server and identified by name. The
connection is a viewer of it rather than its owner, several viewers can follow
the same job, and a viewer that goes away changes nothing about whether the work
finishes. A page that opens asks what is running and rejoins it, replaying the
stages already produced before following the rest.

One reading runs at a time, deliberately. Two would double the memory of a
process that already holds two gigabytes of weights, and the interface offers no
reason to start a second: the point of a job surviving is that nobody needs to.

This registry lives in the process, so a restart forgets what was running. That
is the right trade for a single-container deployment, and the interface treats
a job it cannot find as one that is simply over.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

# how long a finished reading stays reachable, so that a viewer who returns
# late still collects the result instead of being told there is nothing
KEEP_FINISHED = 30 * 60
FOLLOW_TICK = 1.0                       # how often a waiting viewer re-checks


@dataclass
class Job:
    """One reading, its stages so far, and whatever it ended as."""

    id: str
    tab: str                            # the view it belongs to: verdict | cascades
    label: str                          # what it is reading, for the interface
    started: float
    state: str = "running"              # running | done | failed
    steps: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    error: str | None = None
    finished: float | None = None
    _cv: threading.Condition = field(default_factory=threading.Condition, repr=False)

    # -- written by the worker ------------------------------------------------

    def append(self, step: dict[str, Any]) -> None:
        with self._cv:
            self.steps.append(step)
            self._cv.notify_all()

    def finish(self, summary: dict[str, Any] | None = None,
               error: str | None = None) -> None:
        with self._cv:
            self.summary, self.error = summary, error
            self.state = "failed" if error else "done"
            self.finished = time.time()
            self._cv.notify_all()

    # -- read by any number of viewers ---------------------------------------

    def follow(self, start: int = 0) -> Iterator[tuple[str, Any]]:
        """Every stage from ``start`` on, then how the reading ended.

        Yields what is already there before waiting, so a viewer that arrives
        after a reload sees the stages it missed rather than an empty page.
        """
        index = start
        while True:
            with self._cv:
                while index >= len(self.steps) and self.state == "running":
                    self._cv.wait(FOLLOW_TICK)
                pending = self.steps[index:]
                index = len(self.steps)
                state, summary, error = self.state, self.summary, self.error
            for step in pending:
                yield "step", step
            if state != "running":
                # the worker may have appended a last stage between the two
                with self._cv:
                    trailing = self.steps[index:]
                    index = len(self.steps)
                for step in trailing:
                    yield "step", step
                yield ("done", summary) if state == "done" else \
                      ("failed", {"detail": error or "the reading failed"})
                return

    def describe(self) -> dict[str, Any]:
        """What the interface needs to decide whether to rejoin."""
        return {"id": self.id, "tab": self.tab, "label": self.label,
                "state": self.state, "stages": len(self.steps),
                "started": self.started,
                "elapsed": round((self.finished or time.time()) - self.started, 1)}


class Registry:
    """The jobs this process knows about, at most one of them running."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def _prune(self) -> None:
        cutoff = time.time() - KEEP_FINISHED
        for key in [k for k, j in self._jobs.items()
                    if j.finished is not None and j.finished < cutoff]:
            del self._jobs[key]

    def running(self) -> Job | None:
        with self._lock:
            self._prune()
            return next((j for j in self._jobs.values() if j.state == "running"), None)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, tab: str, label: str) -> tuple[Job, bool]:
        """Claim the slot. Returns the job, and whether it is a new one.

        A caller that finds a reading already under way is handed that reading
        rather than an error, because from the interface's point of view the
        answer to "read this" is the same either way: follow what is running.
        """
        with self._lock:
            self._prune()
            existing = next((j for j in self._jobs.values() if j.state == "running"),
                            None)
            if existing is not None:
                return existing, False
            job = Job(id=uuid.uuid4().hex[:12], tab=tab, label=label,
                      started=time.time())
            self._jobs[job.id] = job
            return job, True


REGISTRY = Registry()
