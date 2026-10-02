"""Readings run as server-side jobs so a page reload doesn't lose them.

Before this, reloading mid-reading killed it. Now the browser just follows a
job and can rejoin it. One job at a time (the weights already take ~2GB).
In-memory only, a restart forgets everything, fine for one container.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

# keep finished jobs around a bit so a late reload still gets the result
KEEP_FINISHED = 30 * 60
FOLLOW_TICK = 1.0                       # s


@dataclass
class Job:

    id: str
    tab: str                            # the view it belongs to: verdict | cascades
    label: str                          # shown in the UI
    started: float
    state: str = "running"              # running | done | failed
    steps: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    error: str | None = None
    finished: float | None = None
    _cv: threading.Condition = field(default_factory=threading.Condition, repr=False)

    # worker side

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

    # viewer side

    def follow(self, start: int = 0) -> Iterator[tuple[str, Any]]:
        """Yield steps from `start` (replays what's done first), then done/error."""
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
                # a step may have landed in between
                with self._cv:
                    trailing = self.steps[index:]
                    index = len(self.steps)
                for step in trailing:
                    yield "step", step
                yield ("done", summary) if state == "done" else \
                      ("failed", {"detail": error or "the reading failed"})
                return

    def describe(self) -> dict[str, Any]:
        return {"id": self.id, "tab": self.tab, "label": self.label,
                "state": self.state, "stages": len(self.steps),
                "started": self.started,
                "elapsed": round((self.finished or time.time()) - self.started, 1)}


class Registry:
    """In-memory jobs, max one running."""

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
        """-> (job, is_new). If a job is already running, returns that one."""
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
