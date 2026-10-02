"""Daily limits for the public server: per IP + a global cap, reset at midnight (Paris).

Only what the visitor brings is counted (own text / cascade / bluesky link).
Samples and example posts are free and cached forever. In memory, a restart
resets everything, good enough for a demo.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import threading
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Paris")


def _env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except ValueError:
        return default


LIMITS = {
    "reading": {"visitor": _env("FACTNET_LIMIT_READINGS", 30),
                "site": _env("FACTNET_LIMIT_READINGS_SITE", 300)},
    "fetch": {"visitor": _env("FACTNET_LIMIT_FETCHES", 5),
              "site": _env("FACTNET_LIMIT_FETCHES_SITE", 50)},
}


class LimitReached(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _today() -> str:
    return datetime.now(TZ).date().isoformat()


def resets_at() -> str:
    midnight = datetime.combine(datetime.now(TZ).date() + timedelta(days=1),
                                datetime.min.time(), TZ)
    return midnight.isoformat()


class Quota:
    def __init__(self, limits: dict[str, dict[str, int]] | None = None):
        self.limits = limits or LIMITS
        self._lock = threading.Lock()
        self._day = _today()
        self._visitors: dict[tuple[str, str], int] = {}
        self._site: dict[str, int] = {}

    def _roll(self) -> None:
        day = _today()
        if day != self._day:
            self._day, self._visitors, self._site = day, {}, {}

    def take(self, visitor: str, kind: str) -> None:
        limit = self.limits[kind]
        noun = "readings" if kind == "reading" else "Bluesky links"
        with self._lock:
            self._roll()
            if self._site.get(kind, 0) >= limit["site"]:
                raise LimitReached(
                    f"The site has reached its {noun} for today. It resets at "
                    "midnight, Paris time. The collected cascades still work.")
            if self._visitors.get((visitor, kind), 0) >= limit["visitor"]:
                raise LimitReached(
                    f"Daily limit reached: {limit['visitor']} {noun} per visitor. "
                    "It resets at midnight, Paris time. The collected cascades "
                    "still work.")
            self._visitors[(visitor, kind)] = self._visitors.get((visitor, kind), 0) + 1
            self._site[kind] = self._site.get(kind, 0) + 1

    def status(self, visitor: str) -> dict[str, Any]:
        with self._lock:
            self._roll()
            out: dict[str, Any] = {"resets_at": resets_at()}
            for kind, limit in self.limits.items():
                used = self._visitors.get((visitor, kind), 0)
                site_left = limit["site"] - self._site.get(kind, 0)
                out[kind] = {"limit": limit["visitor"],
                             "left": max(0, min(limit["visitor"] - used, site_left))}
            return out


def visitor_of(peer: str | None, headers: dict[str, str]) -> str:
    """Client IP. X-Real-IP / X-Forwarded-For are only trusted from nginx or the
    docker bridge (private/loopback peer), otherwise anyone could fake them."""
    peer = peer or "unknown"
    try:
        trusted = ipaddress.ip_address(peer).is_private or \
            ipaddress.ip_address(peer).is_loopback
    except ValueError:
        trusted = False
    if trusted:
        real = headers.get("x-real-ip") or \
            headers.get("x-forwarded-for", "").split(",")[0].strip()
        if real:
            return real
    return peer


class Cache:
    """Tiny LRU. size=None = never evict (for site data, bounded anyway)."""

    def __init__(self, size: int | None):
        self.size = size
        self._lock = threading.Lock()
        self._items: OrderedDict[str, Any] = OrderedDict()

    @staticmethod
    def key(payload: Any) -> str:
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()

    def get(self, key: str) -> Any:
        with self._lock:
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while self.size is not None and len(self._items) > self.size:
                self._items.popitem(last=False)


QUOTA = Quota()
READINGS = Cache(64)
SITE_DATA = Cache(None)
FETCHES = Cache(50)
