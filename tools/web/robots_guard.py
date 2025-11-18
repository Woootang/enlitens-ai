"""Robots.txt checker with lightweight caching."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "EnlitensWebTool/0.1 (+https://enlitens.org)"
_CACHE: Dict[str, tuple[RobotFileParser, datetime]] = {}
_CACHE_TTL = timedelta(hours=6)


def _fetch_robot_parser(base_url: str) -> RobotFileParser:
    now = datetime.utcnow()
    entry = _CACHE.get(base_url)
    if entry and now - entry[1] < _CACHE_TTL:
        return entry[0]

    parser = RobotFileParser()
    robots_url = f"{base_url}/robots.txt"
    try:
        response = httpx.get(robots_url, timeout=10.0)
        if response.status_code >= 400:
            parser.parse([])
        else:
            parser.parse(response.text.splitlines())
    except Exception as exc:  # pragma: no cover - network guard
        logger.debug("Robots fetch failed for %s: %s", robots_url, exc)
        parser.parse([])
    _CACHE[base_url] = (parser, now)
    return parser


def is_allowed(url: str) -> bool:
    parsed = urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc
    if not netloc:
        return True
    base = f"{scheme}://{netloc}"
    parser = _fetch_robot_parser(base)
    try:
        allowed = parser.can_fetch(USER_AGENT, url)
    except Exception:  # pragma: no cover - defensive
        allowed = True
    if not allowed:
        logger.debug("Robots disallowed %s", url)
    return allowed
