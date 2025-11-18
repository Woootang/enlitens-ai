"""Agent that monitors symptom and conversation trends."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from .base_agent import BaseAgent
from tools.web.web_search_ddg import WebSearchRequest, ddg_text_search
from src.utils.web_data_store import append_jsonl

logger = logging.getLogger(__name__)

SYMPTOM_QUERIES = [
    "autistic burnout signs 2025",
    "executive dysfunction fatigue",
    "polyvagal safety cues",
    "trauma nervous system flare",
]


class SymptomTrendTrackerAgent(BaseAgent):
    """Surfaces public symptom narratives for educational emphasis."""

    def __init__(self) -> None:
        super().__init__(
            name="SymptomTrendTracker",
            role="Symptom and conversation trend monitoring",
            enable_cot=False,
        )

    async def initialize(self) -> bool:
        self.is_initialized = True
        logger.info("✅ SymptomTrendTracker agent initialized")
        return True

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        document_text = context.get("document_text", "").lower()
        queries = list(SYMPTOM_QUERIES)
        if "sleep" in document_text:
            queries.append("sleep deprivation trauma nervous system")
        if "anxiety" in document_text:
            queries.append("anxiety nervous system dysregulation")

        trends: List[Dict[str, Any]] = []
        for query in queries[:6]:
            try:
                req = WebSearchRequest(query=query, max_results=3)
                results = await asyncio.to_thread(ddg_text_search, req)
                for res in results:
                    trends.append(
                        {
                            "query": query,
                            "title": res.title,
                            "url": res.url,
                            "snippet": res.snippet,
                        }
                    )
            except Exception as exc:
                logger.debug("Symptom trend search failed for %s: %s", query, exc)

        payload = {"symptom_trends": trends}
        append_jsonl("symptom_trends.jsonl", [{"document_id": context.get("document_id"), "trends": trends}])
        return payload

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        return bool(output.get("symptom_trends"))
