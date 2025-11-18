"""RSS/Atom feed helpers."""

from __future__ import annotations

from typing import List, Optional

import feedparser
from pydantic import BaseModel


class FeedItem(BaseModel):
    title: str
    url: str
    summary: str | None = None
    published: str | None = None
    source_feed: str | None = None


def fetch_feed(url: str, *, max_items: int = 20) -> List[FeedItem]:
    """Parse an RSS/Atom feed into normalized entries."""

    parsed = feedparser.parse(url)
    items: List[FeedItem] = []
    for entry in parsed.entries[:max_items]:
        items.append(
            FeedItem(
                title=getattr(entry, "title", ""),
                url=getattr(entry, "link", ""),
                summary=getattr(entry, "summary", None),
                published=getattr(entry, "published", None),
                source_feed=url,
            )
        )
    return items


DEFAULT_FEEDS: dict[str, str] = {
    "stl_doh_news": "https://www.stlouis-mo.gov/government/departments/health/news/index.cfm?dept=10&action=rss",
    "stl_county_health": "https://stlouiscountymo.gov/st-louis-county-departments/public-health/newsroom/?rss=true",
    "missouri_dmh": "https://dmh.mo.gov/news.rss",
}
