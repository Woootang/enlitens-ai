"""Agent that surfaces myth narratives for adversarial reframing."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from .base_agent import BaseAgent
from tools.web.scrape_url import ScrapeUrlRequest, scrape_url
from tools.web.web_search_ddg import WebSearchRequest, ddg_text_search
from src.utils.web_data_store import append_jsonl

logger = logging.getLogger(__name__)

MYTH_QUERIES = [
    "autism cure myth",
    "trauma healing quick fix",
    "ADHD not real",
    "mental health miracle cure St. Louis",
]


class MythScraperAgent(BaseAgent):
    """Collects misinformation snippets for the Rebellion Framework agent."""

    def __init__(self) -> None:
        super().__init__(
            name="MythScraper",
            role="Myth and misinformation harvesting",
            enable_cot=False,
        )

    async def initialize(self) -> bool:
        self.is_initialized = True
        logger.info("✅ MythScraper agent initialized")
        return True

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        document_text = context.get("document_text", "")
        topic_hint = document_text[:100].lower()
        queries = list(MYTH_QUERIES)
        if "autism" in topic_hint:
            queries.append("autism recovery stories scam")
        if "trauma" in topic_hint:
            queries.append("trauma release detox myth")
        if "adhd" in topic_hint:
            queries.append("adhd fake disorder myth")

        harvested: List[Dict[str, Any]] = []
        for query in queries:
            try:
                search_req = WebSearchRequest(query=query, max_results=3)
                results = await asyncio.to_thread(ddg_text_search, search_req)
                for res in results:
                    snippet = res.snippet or ""
                    body_text = ""
                    scrape_req = ScrapeUrlRequest(url=res.url)
                    try:
                        scrape_result = await asyncio.to_thread(scrape_url, scrape_req)
                        if scrape_result:
                            body_text = " ".join(scrape_result.text.split()[:80])
                    except Exception as scrape_exc:
                        logger.debug("Scrape failed for %s: %s", res.url, scrape_exc)
                    harvested.append(
                        {
                            "query": query,
                            "title": res.title,
                            "url": res.url,
                            "snippet": snippet,
                            "excerpt": body_text,
                        }
                    )
            except Exception as exc:
                logger.debug("Myth search failed for %s: %s", query, exc)

        payload = {"myth_records": harvested}
        append_jsonl("myth_records.jsonl", [{"document_id": context.get("document_id"), "records": harvested}])
        return payload

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        return bool(output.get("myth_records"))
