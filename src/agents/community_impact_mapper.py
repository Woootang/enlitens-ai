"""Agent that maps research topics to local St. Louis impact data."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .base_agent import BaseAgent
from src.utils.web_data_store import append_jsonl

logger = logging.getLogger(__name__)


class CommunityImpactMapper(BaseAgent):
    """Summarises how topics connect to local public-health signals."""

    def __init__(self) -> None:
        super().__init__(
            name="CommunityImpactMapper",
            role="Local impact synthesis",
            enable_cot=False,
        )

    async def initialize(self) -> bool:
        self.is_initialized = True
        logger.info("✅ CommunityImpactMapper agent initialized")
        return True

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stl_context = context.get("st_louis_context") or {}
        client_insights = context.get("client_insights") or {}
        document_text = context.get("document_text", "")

        highlights: List[str] = []
        if stl_context.get("demographics"):
            highlights.append("St. Louis demographics data available for tailoring messaging.")
        if stl_context.get("mental_health_trends"):
            highlights.append("Local mental health trend metrics present in context payload.")
        if "trauma" in document_text.lower():
            highlights.append("Document references trauma; align with local trauma prevalence data.")
        if client_insights.get("pain_points"):
            highlights.append("Client pain points identified: " + ", ".join(client_insights["pain_points"][:3]))

        impact_payload = {
            "st_louis_context": stl_context,
            "client_insights": client_insights,
            "highlights": highlights,
        }

        append_jsonl("community_impact.jsonl", [{"document_id": context.get("document_id"), **impact_payload}])
        return {"community_impact": impact_payload}

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        return bool(output.get("community_impact"))
