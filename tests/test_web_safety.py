import sys
from types import SimpleNamespace

try:
    import backoff  # noqa: F401
except ImportError:
    dummy_backoff = SimpleNamespace(
        expo=lambda *args, **kwargs: 1,
        on_exception=lambda *args, **kwargs: (lambda func: func),
    )
    sys.modules['backoff'] = dummy_backoff

import pytest

from tools.web import allowlist
from tools.web import robots_guard
from tools.web.http_client import CACHE, fetch_url


def teardown_module(module):
    # Reset caches between tests
    allowlist.load_allowed_hosts.cache_clear()
    robots_guard._CACHE.clear()
    CACHE.clear()


def test_allowlist_known_host():
    allowlist.load_allowed_hosts.cache_clear()
    hosts = allowlist.load_allowed_hosts()
    # Ensure fixture includes cdc.gov (present in default file)
    assert "cdc.gov" in hosts
    assert allowlist.is_host_allowed("cdc.gov")


def test_allowlist_unknown_host():
    allowlist.load_allowed_hosts.cache_clear()
    assert not allowlist.is_host_allowed("unknown-domain.example")


def test_robots_guard_disallow(monkeypatch):
    robots_guard._CACHE.clear()

    class FakeResponse:
        status_code = 200

        def __init__(self, text: str):
            self.text = text

    def fake_get(url, timeout=10.0):  # noqa: D401
        return FakeResponse("User-agent: *\nDisallow: /blocked")

    monkeypatch.setattr(robots_guard, "httpx", SimpleNamespace(get=fake_get))
    url = "https://example.com/blocked/page"
    assert not robots_guard.is_allowed(url)


def test_fetch_url_respects_allowlist(monkeypatch):
    CACHE.clear()

    monkeypatch.setattr("tools.web.http_client.is_host_allowed", lambda host: False)
    result = fetch_url("https://forbidden.example.com/path")
    assert result is None


def test_fetch_url_respects_robots(monkeypatch):
    CACHE.clear()

    def fake_allowed(url):
        return False

    monkeypatch.setattr("tools.web.http_client.robots_allowed", fake_allowed)
    result = fetch_url("https://www.cdc.gov/some/path")
    assert result is None
