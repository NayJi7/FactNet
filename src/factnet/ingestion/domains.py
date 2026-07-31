"""Source-level credibility labels for collected posts.

Judging every collected claim by hand does not scale, and on a platform whose
users mostly discuss misinformation rather than produce it, keyword search
returns almost no misleading content: our first sample held 87 reliable claims
against 6 misleading ones, which is unusable for evaluation.

The standard answer in this literature is to label at the level of the source
rather than the claim, collecting posts that link to outlets whose factual
record is already established, so that the label comes from a published rating
instead of the annotator's judgement. Low-credibility domains are read from the
Iffy Index, which lists sites rated low factual reporting by Media Bias/Fact
Check and repeatedly failing IFCN-verified fact checks, with political leaning
explicitly excluded as a criterion.

The reliable side has no equivalent open file, so it uses a short list of
outlets rated high or very high for factual reporting, kept deliberately small
and conservative.

A caveat travels with every label produced here: a post linking to an unreliable
outlet is not necessarily endorsing it, since debunkings link to what they
debunk. Only posts carrying the link as an attached preview are kept, which
removes bare mentions, and the residual noise is a known property of
source-level labelling rather than something this module can resolve.
"""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

IFFY_CSV = ("https://docs.google.com/spreadsheets/d/"
            "1ck1_FZC-97uDLIlvRJDTrGqBk0FuDe9yHkluROgpGS8/gviz/tq?tqx=out:csv&sheet=Iffy-news")
CACHE = Path(__file__).resolve().parents[3] / "data" / "raw" / "iffy-index.csv"

# Outlets rated high or very high for factual reporting; kept short on purpose,
# since every addition is a judgement this project would have to defend.
HIGH_CREDIBILITY = (
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org", "pbs.org",
    "nature.com", "science.org", "sciencenews.org", "scientificamerican.com",
    "theguardian.com", "nytimes.com", "washingtonpost.com", "wsj.com",
    "economist.com", "ft.com", "afp.com", "lemonde.fr", "statnews.com",
    "nejm.org", "thelancet.com", "jamanetwork.com", "bmj.com", "cdc.gov",
    "who.int", "nih.gov", "fda.gov", "nasa.gov", "noaa.gov",
)


def fetch_iffy(path: Path = CACHE, refresh: bool = False) -> Path:
    """Download the Iffy Index once and keep it beside the collected data."""
    if path.exists() and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(IFFY_CSV, timeout=60) as response:  # noqa: S310
        path.write_bytes(response.read())
    return path


# The published file is broader than the index it documents: alongside sites
# rated low and very low for factual reporting it carries some 1,300 rated
# mixed, which include mainstream partisan outlets. Treating a link to those as
# misleading would be wrong at the level of the individual article, so only the
# two lowest tiers are kept, which is the inclusion rule the index states.
KEPT_RATINGS = ("L", "VL")


def load_low_credibility(path: Path = CACHE,
                         ratings: tuple[str, ...] = KEPT_RATINGS) -> dict[str, str]:
    """Domain to MBFC factual rating, restricted to the lowest tiers."""
    fetch_iffy(path)
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {r["Domain"].strip().lower(): (r.get("MBFC Fact") or "").strip()
            for r in rows if r.get("Domain")
            and (r.get("MBFC Fact") or "").strip() in ratings}


def registrable(url: str) -> str:
    """Host of a URL without the leading www, empty when there is no host."""
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def classify(url: str, low: dict[str, str]) -> str | None:
    """``misleading`` for a listed unreliable source, ``reliable`` for a vetted one."""
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
    """True when the link points at a specific piece rather than at a site.

    Posts that merely name an outlet link to its home page, and they are usually
    commentary about that outlet rather than diffusion of its content.
    """
    path = urlparse(url).path.strip("/")
    return len(path) > 3


def post_links(post: dict) -> list[str]:
    """Links the post actually carries, ignoring domains merely named in the text."""
    record = post.get("record") or {}
    external = ((record.get("embed") or {}).get("external") or {}).get("uri")
    # only the attached preview counts: Bluesky turns a bare domain typed in the
    # text into a link of its own, which would let simple mentions through
    return [external] if external else []
