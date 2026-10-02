"""Daily limits on the public deployment."""

import pytest

from factnet.serve import quota


def small(reading=2, fetch=1, site_reading=100, site_fetch=100):
    return quota.Quota({"reading": {"visitor": reading, "site": site_reading},
                        "fetch": {"visitor": fetch, "site": site_fetch}})


def test_a_visitor_is_stopped_at_the_daily_limit():
    q = small(reading=2)
    q.take("1.2.3.4", "reading")
    q.take("1.2.3.4", "reading")
    with pytest.raises(quota.LimitReached):
        q.take("1.2.3.4", "reading")
    # someone else is not affected
    q.take("5.6.7.8", "reading")


def test_the_site_ceiling_holds_whatever_the_address():
    q = small(reading=10, site_reading=3)
    for ip in ("a", "b", "c"):
        q.take(ip, "reading")
    with pytest.raises(quota.LimitReached, match="site"):
        q.take("d", "reading")


def test_status_reports_what_is_left_and_never_goes_negative():
    q = small(reading=2, fetch=1, site_reading=1)
    q.take("x", "reading")
    status = q.status("x")
    assert status["reading"] == {"limit": 2, "left": 0}    # site ceiling reached
    assert status["fetch"] == {"limit": 1, "left": 1}


def test_counts_reset_on_a_new_day(monkeypatch):
    q = small(reading=1)
    q.take("x", "reading")
    monkeypatch.setattr(quota, "_today", lambda: "2999-01-01")
    q.take("x", "reading")


def test_forwarded_address_only_trusted_from_the_proxy():
    headers = {"x-real-ip": "203.0.113.7"}
    assert quota.visitor_of("127.0.0.1", headers) == "203.0.113.7"
    assert quota.visitor_of("172.18.0.1", headers) == "203.0.113.7"    # docker bridge
    assert quota.visitor_of("8.8.8.8", headers) == "8.8.8.8"  # forged from outside
    assert quota.visitor_of("127.0.0.1", {"x-forwarded-for": "203.0.113.9, 10.0.0.1"}) \
        == "203.0.113.9"


def test_cache_drops_the_oldest_entry():
    cache = quota.Cache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")              # a is now the most recent
    cache.put("c", 3)
    assert cache.get("b") is None and cache.get("a") == 1


def test_api_refuses_past_the_limit_and_does_not_charge_a_repeat(monkeypatch):
    from fastapi.testclient import TestClient

    from factnet.serve.api import app

    monkeypatch.setattr(quota, "QUOTA", small(reading=1))
    monkeypatch.setattr(quota, "READINGS", quota.Cache(8))
    client = TestClient(app)

    first = {"text": "The unemployment rate has doubled, the ministry said."}
    assert client.post("/api/verdict", json=first).status_code == 200
    # the same request again comes from the cache and costs nothing
    assert client.post("/api/verdict", json=first).status_code == 200

    other = {"text": "Inflation fell to two percent last month, the bank said."}
    refused = client.post("/api/verdict", json=other)
    assert refused.status_code == 429
    assert "Daily limit" in refused.json()["detail"]

    status = client.get("/api/quota").json()
    assert status["reading"] == {"limit": 1, "left": 0}
    assert status["fetch"]["left"] == 1


def test_data_the_site_holds_is_never_charged(monkeypatch):
    from fastapi.testclient import TestClient

    from factnet.serve import samples
    from factnet.serve.api import app

    monkeypatch.setattr(quota, "QUOTA", small(reading=0))
    monkeypatch.setattr(quota, "SITE_DATA", quota.Cache(None))
    client = TestClient(app)

    # an example post, a collected cascade and the worked example all go through
    assert client.post("/api/verdict", json={"text": samples.EXAMPLE_POSTS[0]}).status_code == 200
    if samples.load():
        assert client.post("/api/verdict", json={"sample_id": 0}).status_code == 200
        example = client.get("/api/samples/example").json()["cascade"]
        assert client.post("/api/verdict", json={"cascade": example,
                                                 "origin": "cascade"}).status_code == 200
    # anything a visitor brings is still charged
    own = client.post("/api/verdict", json={"text": "My cousin says the moon is hollow."})
    assert own.status_code == 429


def test_the_site_data_cache_never_evicts():
    cache = quota.Cache(None)
    for i in range(1000):
        cache.put(str(i), i)
    assert cache.get("0") == 0


def test_example_posts_match_the_buttons_on_the_page():
    """The server recognises an example post by its exact text, so the two lists must agree."""
    import re
    from pathlib import Path

    from factnet.serve import samples

    source = (Path(__file__).resolve().parents[1] / "web" / "src" / "components"
              / "Examples.tsx").read_text(encoding="utf-8")
    block = source.split("const EXAMPLES")[1].split("];")[0]
    page = []
    for entry in re.findall(r'\[\s*"[^"]*",\s*((?:"[^"]*"\s*\+?\s*)+)\]', block):
        page.append("".join(re.findall(r'"([^"]*)"', entry)))
    assert tuple(page) == samples.EXAMPLE_POSTS
