"""DuckDuckGo/DDGS search integration."""

from __future__ import annotations

from typing import List

import backoff
from pydantic import BaseModel, Field

try:  # Prefer the renamed package
    from ddgs import DDGS  # type: ignore
except ImportError:  # pragma: no cover
    from duckduckgo_search import DDGS  # type: ignore


class WebSearchResult(BaseModel):
    """Normalized search result item."""

    title: str
    url: str
    snippet: str | None = None
    source: str = "duckduckgo"


class WebSearchRequest(BaseModel):
    """Validated input for the DDG search tool."""

    query: str = Field(..., min_length=3, description="Search query string")
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return",
    )


@backoff.on_exception(backoff.expo, Exception, max_time=60)
def ddg_text_search(req: WebSearchRequest) -> List[WebSearchResult]:
    """Execute a DuckDuckGo/DDGS text search and return normalized results."""

    results: List[WebSearchResult] = []
    with DDGS() as ddgs:
        for raw in ddgs.text(
            req.query,
            safesearch="moderate",
            max_results=req.max_results,
        ):
            title = raw.get("title") or ""
            url = raw.get("href") or ""
            snippet = raw.get("body")
            if not url:
                continue
            results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                )
            )
    return results
