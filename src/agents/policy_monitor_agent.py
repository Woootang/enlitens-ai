"""Agent that tracks policy updates and advisories."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from .base_agent import BaseAgent
from tools.web.feeds import fetch_feed
from tools.web.web_search_ddg import WebSearchRequest, ddg_text_search
from src.utils.web_data_store import append_jsonl

logger = logging.getLogger(__name__)

POLICY_FEEDS = {
    "cdc_press": "https://www.cdc.gov/media/rss/rss.xml",
    "missouri_health_alerts": "https://health.mo.gov/living/healthcondiseases/communicable/communicabledisease/news-rss.php",
}

POLICY_QUERIES = [
    "Missouri mental health policy 2025",
    "St. Louis trauma legislation",
    "Missouri medicaid behavioral health updates",
]


class PolicyMonitorAgent(BaseAgent):
    """Collects policy-related updates for downstream agents."""

    def __init__(self) -> None:
        super().__init__(
            name="PolicyMonitor",
            role="Policy and advisory monitoring",
            enable_cot=False,
        )

    async def initialize(self) -> bool:
        self.is_initialized = True
        logger.info("✅ PolicyMonitor agent initialized")
        return True

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        feed_updates: List[Dict[str, Any]] = []
        for feed_name, feed_url in POLICY_FEEDS.items():
            try:
                entries = await asyncio.to_thread(fetch_feed, feed_url, max_items=5)
                for entry in entries:
                    data = entry.model_dump()
                    data["feed_name"] = feed_name
                    feed_updates.append(data)
            except Exception as exc:
                logger.debug("Policy feed failed %s: %s", feed_url, exc)

        query_results: List[Dict[str, Any]] = []
        for query in POLICY_QUERIES:
            try:
                req = WebSearchRequest(query=query, max_results=3)
                results = await asyncio.to_thread(ddg_text_search, req)
                for res in results:
                    payload = res.model_dump()
                    payload["query"] = query
                    query_results.append(payload)
            except Exception as exc:
                logger.debug("Policy search failed %s: %s", query, exc)

        payload = {
            "policy_updates": {
                "feeds": feed_updates,
                "search_hits": query_results,
            }
        }
        append_jsonl("policy_updates.jsonl", [{"document_id": context.get("document_id"), **payload["policy_updates"]}])
        return payload

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        payload = output.get("policy_updates") or {}
        return bool(payload.get("feeds") or payload.get("search_hits"))
