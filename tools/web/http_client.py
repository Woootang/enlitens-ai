"""Shared HTTP client with caching and polite defaults."""

from __future__ import annotations

import logging
import time
from typing import Optional
from urllib.parse import urlparse

import backoff
import httpx
from diskcache import Cache

from .allowlist import is_host_allowed
from .robots_guard import is_allowed as robots_allowed

CACHE = Cache("./cache/http")
DEFAULT_HEADERS = {
    "User-Agent": "EnlitensWebTool/0.1 (+https://enlitens.org)",
    "Accept-Language": "en-US,en;q=0.9",
}

logger = logging.getLogger(__name__)


@backoff.on_exception(backoff.expo, httpx.RequestError, max_time=60)
def fetch_url(url: str, *, ttl: int = 60 * 60 * 24) -> Optional[str]:
    """Fetch a URL with caching, headers, and retry logic."""

    parsed = urlparse(url)
    host = parsed.netloc.lower() if parsed.netloc else ""
    if host and not is_host_allowed(host):
        logger.debug("Host %s not in allowlist", host)
        return None
    if not robots_allowed(url):
        return None

    cached = CACHE.get(url)
    if cached is not None:
        return cached

    with httpx.Client(
        headers=DEFAULT_HEADERS,
        timeout=httpx.Timeout(20.0),
        follow_redirects=True,
    ) as client:
        response = client.get(url)
        if response.status_code >= 400:
            return None
        text = response.text
        CACHE.set(url, text, expire=ttl)
        # small sleep to avoid aggressive hammering
        time.sleep(0.2)
        return text
