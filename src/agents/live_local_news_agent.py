"""Agent that surfaces live local news and public-health updates."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from .base_agent import BaseAgent
from tools.web.feeds import DEFAULT_FEEDS, fetch_feed
from tools.web.web_search_ddg import WebSearchRequest, ddg_text_search
from src.utils.web_data_store import append_jsonl

logger = logging.getLogger(__name__)


class LiveLocalNewsAgent(BaseAgent):
    """Fetches local news stories and headlines relevant to Enlitens."""

    def __init__(self) -> None:
        super().__init__(
            name="LiveLocalNews",
            role="Local news and public-health signal aggregation",
            enable_cot=False,
        )

    async def initialize(self) -> bool:
        self.is_initialized = True
        logger.info("✅ LiveLocalNews agent initialized")
        return True

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        document_text = context.get("document_text", "")
        topic_hint = document_text[:140].strip().replace("\n", " ")
        queries = [
            "St. Louis mental health news",
            "St. Louis trauma support programs",
        ]
        if topic_hint:
            queries.append(f"St. Louis {topic_hint[:60]}")

        feed_items: List[Dict[str, Any]] = []
        for feed_name, feed_url in DEFAULT_FEEDS.items():
            try:
                entries = await asyncio.to_thread(fetch_feed, feed_url, max_items=5)
                for entry in entries:
                    data = entry.model_dump()
                    data["feed_name"] = feed_name
                    feed_items.append(data)
            except Exception as exc:
                logger.debug("Feed fetch failed for %s: %s", feed_url, exc)

        search_hits: List[Dict[str, Any]] = []
        for query in queries:
            try:
                request = WebSearchRequest(query=query, max_results=4)
                results = await asyncio.to_thread(ddg_text_search, request)
                for res in results:
                    item = res.model_dump()
                    item["query"] = query
                    search_hits.append(item)
            except Exception as exc:
                logger.debug("Search failed for %s: %s", query, exc)

        payload = {
            "live_local_news": {
                "feeds": feed_items,
                "search_hits": search_hits,
            }
        }
        append_jsonl("local_news.jsonl", [{"document_id": context.get("document_id"), **payload["live_local_news"]}])
        return payload

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        payload = output.get("live_local_news") or {}
        return bool(payload.get("feeds") or payload.get("search_hits"))
