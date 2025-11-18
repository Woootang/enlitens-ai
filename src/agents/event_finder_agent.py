"""Agent that finds local educational events and workshops."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from .base_agent import BaseAgent
from tools.web.web_search_ddg import WebSearchRequest, ddg_text_search
from src.utils.web_data_store import append_jsonl

logger = logging.getLogger(__name__)

EVENT_QUERIES = [
    "St. Louis mental health workshop",
    "St. Louis neurodiversity support group",
    "St. Louis trauma informed training",
    "Missouri autism parent event",
]


class EventFinderAgent(BaseAgent):
    """Discovers community events for educational outreach."""

    def __init__(self) -> None:
        super().__init__(
            name="EventFinder",
            role="Community event discovery",
            enable_cot=False,
        )

    async def initialize(self) -> bool:
        self.is_initialized = True
        logger.info("✅ EventFinder agent initialized")
        return True

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        collected: List[Dict[str, Any]] = []
        for query in EVENT_QUERIES:
            try:
                req = WebSearchRequest(query=query, max_results=4)
                results = await asyncio.to_thread(ddg_text_search, req)
                for res in results:
                    collected.append(
                        {
                            "query": query,
                            "title": res.title,
                            "url": res.url,
                            "summary": res.snippet,
                        }
                    )
            except Exception as exc:
                logger.debug("Event search failed for %s: %s", query, exc)
        payload = {"event_leads": collected}
        append_jsonl("events.jsonl", [{"document_id": context.get("document_id"), "events": collected}])
        return payload

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        return bool(output.get("event_leads"))
