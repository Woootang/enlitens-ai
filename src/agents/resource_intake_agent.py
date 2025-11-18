"""Agent that discovers local community resources."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from .base_agent import BaseAgent
from tools.web.web_search_ddg import WebSearchRequest, WebSearchResult, ddg_text_search
from src.utils.web_data_store import append_jsonl

logger = logging.getLogger(__name__)

RESOURCE_QUERIES = [
    "St. Louis sliding scale therapy",
    "St. Louis trauma counseling program",
    "St. Louis autism support services",
    "Missouri peer support mental health",
]


class ResourceIntakeAgent(BaseAgent):
    """Collects local resource leads for manual verification."""

    def __init__(self) -> None:
        super().__init__(
            name="ResourceIntake",
            role="Community resource discovery",
            enable_cot=False,
        )

    async def initialize(self) -> bool:
        self.is_initialized = True
        logger.info("✅ ResourceIntake agent initialized")
        return True

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        queries = list(RESOURCE_QUERIES)
        persona = (context.get("client_insights") or {}).get("persona_summary") or {}
        top_pain = persona.get("top_pain_points") or []
        for pain in top_pain[:2]:
            queries.append(f"St. Louis support {pain}")

        collected: List[Dict[str, Any]] = []
        for query in queries:
            try:
                req = WebSearchRequest(query=query, max_results=4)
                results = await asyncio.to_thread(ddg_text_search, req)
                collected.extend(self._summarize_results(query, results))
            except Exception as exc:
                logger.debug("Resource search failed for %s: %s", query, exc)

        payload = {"resource_leads": collected}
        append_jsonl("resource_leads.jsonl", [{"document_id": context.get("document_id"), "leads": collected}])
        return payload

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        return bool(output.get("resource_leads"))

    def _summarize_results(
        self, query: str, results: List[WebSearchResult]
    ) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for res in results:
            payload.append(
                {
                    "query": query,
                    "title": res.title,
                    "url": res.url,
                    "summary": res.snippet,
                }
            )
        return payload
