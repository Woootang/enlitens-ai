"""Wrapper for SearXNG meta-search."""

from __future__ import annotations

import os
from typing import List

import backoff
import httpx
from pydantic import BaseModel, Field

from .web_search_ddg import WebSearchResult

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080")


class SearxSearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    max_results: int = Field(default=7, ge=1, le=20)


@backoff.on_exception(backoff.expo, httpx.RequestError, max_time=60)
def searxng_search(req: SearxSearchRequest) -> List[WebSearchResult]:
    params = {
        "q": req.query,
        "format": "json",
        "language": "en-US",
        "safesearch": 1,
    }
    response = httpx.get(f"{SEARXNG_URL}/search", params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    items: List[WebSearchResult] = []
    for raw in payload.get("results", [])[: req.max_results]:
        items.append(
            WebSearchResult(
                title=raw.get("title", ""),
                url=raw.get("url", ""),
                snippet=raw.get("content"),
                source=f"searxng:{raw.get('engine', '')}",
            )
        )
    return items
