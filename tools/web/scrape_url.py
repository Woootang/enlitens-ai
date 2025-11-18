"""HTTP scrape tool backed by httpx + Trafilatura."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, HttpUrl

from .http_client import fetch_url
from .extractors import extract_main_text


class ScrapeUrlRequest(BaseModel):
    """Input schema for scrape_url_tool."""

    url: HttpUrl


class ScrapeUrlResult(BaseModel):
    """Output schema for scrape_url_tool."""

    url: HttpUrl
    title: str | None
    text: str


def scrape_url(req: ScrapeUrlRequest) -> Optional[ScrapeUrlResult]:
    """Fetch and extract the readable text for a URL."""

    html = fetch_url(str(req.url))
    if not html:
        return None
    extracted = extract_main_text(html, url=str(req.url))
    if not extracted or not extracted.get("text"):
        return None
    return ScrapeUrlResult(
        url=req.url,
        title=extracted.get("title"),
        text=extracted["text"],
    )
