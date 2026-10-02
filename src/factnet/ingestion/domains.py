"""Source-level labels: a post is labelled by the outlet it links to.

Hand-labelling didn't work out, our first keyword sample was 87 reliable vs 6
misleading. Misleading = Iffy Index (MBFC low / very low factual). Reliable =
a short hand-picked list of high-factual outlets, no open list exists for that.

Known noise: someone debunking a fake site still links to it. We only keep
attached link previews, which removes plain mentions, but that's it.
"""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

IFFY_CSV = ("https://docs.google.com/spreadsheets/d/"
            "1ck1_FZC-97uDLIlvRJDTrGqBk0FuDe9yHkluROgpGS8/gviz/tq?tqx=out:csv&sheet=Iffy-news")
CACHE = Path(__file__).resolve().parents[3] / "data" / "raw" / "iffy-index.csv"

# MBFC high / very high factual. kept short on purpose
HIGH_CREDIBILITY = (
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org", "pbs.org",
    "nature.com", "science.org", "sciencenews.org", "scientificamerican.com",
    "theguardian.com", "nytimes.com", "washingtonpost.com", "wsj.com",
    "economist.com", "ft.com", "afp.com", "lemonde.fr", "statnews.com",
    "nejm.org", "thelancet.com", "jamanetwork.com", "bmj.com", "cdc.gov",
    "who.int", "nih.gov", "fda.gov", "nasa.gov", "noaa.gov",
)


def fetch_iffy(path: Path = CACHE, refresh: bool = False) -> Path:
    """Download the Iffy csv once (cached)."""
    if path.exists() and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(IFFY_CSV, timeout=60) as response:  # noqa: S310
        path.write_bytes(response.read())
    return path


# the csv also has ~1300 "mixed" sites (incl. mainstream partisan ones), we only
# keep low / very low like the index itself says
KEPT_RATINGS = ("L", "VL")


def load_low_credibility(path: Path = CACHE,
                         ratings: tuple[str, ...] = KEPT_RATINGS) -> dict[str, str]:
    """{domain: rating} for the low tiers."""
    fetch_iffy(path)
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {r["Domain"].strip().lower(): (r.get("MBFC Fact") or "").strip()
            for r in rows if r.get("Domain")
            and (r.get("MBFC Fact") or "").strip() in ratings}


def registrable(url: str) -> str:
    """host without www., or ''"""
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def classify(url: str, low: dict[str, str]) -> str | None:
    """'misleading' / 'reliable' / None"""
    host = registrable(url)
    if not host:
        return None
    parts = host.split(".")
    candidates = {host} | {".".join(parts[i:]) for i in range(len(parts) - 1)}
    if candidates & set(low):
        return "misleading"
    if candidates & set(HIGH_CREDIBILITY):
        return "reliable"
    return None


def is_article(url: str) -> bool:
    """Link to an actual article, not a homepage (homepage = usually talking about the site)."""
    path = urlparse(url).path.strip("/")
    return len(path) > 3


def post_links(post: dict) -> list[str]:
    """Attached link (embed) only."""
    record = post.get("record") or {}
    external = ((record.get("embed") or {}).get("external") or {}).get("uri")
    # facets would also catch domains just typed in the text
    return [external] if external else []
