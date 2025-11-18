"""Agent that retrieves recent scholarly updates."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from .base_agent import BaseAgent
from tools.web.openalex_client import OpenAlexWork, search_openalex
from src.utils.web_data_store import append_jsonl

logger = logging.getLogger(__name__)

BASE_TOPICS = [
    "autism neurodiversity",
    "trauma memory reconsolidation",
    "executive function burnout",
    "polyvagal social engagement",
]


class ResearchUpdateAgent(BaseAgent):
    """Fetches recent papers relevant to Enlitens focus areas."""

    def __init__(self) -> None:
        super().__init__(
            name="ResearchUpdate",
            role="Scholarly update aggregation",
            enable_cot=False,
        )

    async def initialize(self) -> bool:
        self.is_initialized = True
        logger.info("✅ ResearchUpdate agent initialized")
        return True

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        document_text = context.get("document_text", "")
        derived_topics = self._derive_topics(document_text)
        topics = BASE_TOPICS + derived_topics

        updates: List[Dict[str, Any]] = []
        for topic in topics[:6]:
            try:
                works = await asyncio.to_thread(search_openalex, topic, per_page=3)
                updates.extend(self._serialize(topic, works))
            except Exception as exc:
                logger.debug("OpenAlex search failed for %s: %s", topic, exc)

        payload = {"research_updates": updates}
        append_jsonl("research_updates.jsonl", [{"document_id": context.get("document_id"), "updates": updates}])
        return payload

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        return bool(output.get("research_updates"))

    def _derive_topics(self, text: str) -> List[str]:
        lower = text.lower()
        derived: List[str] = []
        if "autism" in lower:
            derived.append("autism sensory regulation")
        if "adhd" in lower:
            derived.append("adhd dopamine motivation")
        if "trauma" in lower:
            derived.append("trauma stress neuroimmune")
        if "burnout" in lower:
            derived.append("burnout executive dysfunction neuroscience")
        return derived

    def _serialize(self, topic: str, works: List[OpenAlexWork]) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for work in works:
            payload.append(
                {
                    "topic": topic,
                    "title": work.title,
                    "year": work.publication_year,
                    "doi": work.doi,
                    "concepts": work.concepts,
                    "abstract": work.abstract,
                    "primary_location": work.primary_location,
                    "id": work.id,
                }
            )
        return payload
