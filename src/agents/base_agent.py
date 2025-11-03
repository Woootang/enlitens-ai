"""
Base Agent class for the Enlitens Multi-Agent System.

This provides the foundation for all specialized agents in the system.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime

from src.utils.settings import get_settings
from src.monitoring.error_telemetry import TelemetrySeverity, log_with_telemetry

logger = logging.getLogger(__name__)
TELEMETRY_AGENT = "base_agent"

class BaseAgent(ABC):
    """
    Base class for all agents in the Enlitens multi-agent system.
    """

    def __init__(self, name: str, role: str, model: Optional[str] = None):
        settings = get_settings()
        agent_key = self.__class__.__name__
        resolved_model = model or settings.model_for_agent(agent_key)
        if not resolved_model:
            raise ValueError(f"No model configured for agent '{agent_key}'")

        self.name = name
        self.role = role
        self.model = resolved_model
        self.created_at = datetime.now()
        self.is_initialized = False
        self.settings = settings
        self.llm_provider = settings.llm.provider
        self.connection_info = {
            "base_url": settings.llm.endpoint_for(agent_key),
            "provider": settings.llm.provider,
        }
        logger.info(f"Initializing agent: {name} ({role})")

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the agent with necessary resources."""
        pass

    @abstractmethod
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return results."""
        pass

    @abstractmethod
    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate the output quality."""
        pass

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main function with error handling.
        """
        try:
            # Work with a copy so downstream modifications don't leak across nodes
            context = dict(context)
            context.setdefault("cache_prefix", self.name)
            context.setdefault("cache_chunk_id", context.get("document_id", "global"))

            if not self.is_initialized:
                success = await self.initialize()
                if not success:
                    raise RuntimeError(f"Failed to initialize agent {self.name}")

            logger.info(f"Agent {self.name} starting processing")
            result = await self.process(context)

            if await self.validate_output(result):
                logger.info(f"Agent {self.name} completed successfully")
                return result
            else:
                log_with_telemetry(
                    logger.warning,
                    "Agent %s output validation failed",
                    self.name,
                    agent=TELEMETRY_AGENT,
                    severity=TelemetrySeverity.MAJOR,
                    impact="Agent output failed validation",
                    doc_id=context.get("document_id"),
                )
                return {}

        except Exception as e:
            log_with_telemetry(
                logger.error,
                "Agent %s execution failed: %s",
                self.name,
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Agent execution crashed",
                doc_id=context.get("document_id"),
                details={"error": str(e)},
            )
            return {}

    async def cleanup(self):
        """Clean up agent resources."""
        logger.info(f"Cleaning up agent: {self.name}")

    def get_status(self) -> Dict[str, Any]:
        """Get agent status information."""
        return {
            "name": self.name,
            "role": self.role,
            "model": self.model,
            "initialized": self.is_initialized,
            "created_at": self.created_at.isoformat()
        }

    def _cache_kwargs(self, context: Dict[str, Any], suffix: Optional[str] = None) -> Dict[str, str]:
        """Helper to build cache arguments for prompt-based agents."""
        prefix = context.get("cache_prefix", self.name)
        if suffix:
            prefix = f"{prefix}:{suffix}"
        chunk_id = context.get("cache_chunk_id") or context.get("document_id", "global")
        return {
            "cache_prefix": prefix,
            "cache_chunk_id": chunk_id,
        }

    @staticmethod
    def _summarize_text_block(text: Optional[str], max_chars: int = 400) -> str:
        if not text:
            return ""
        cleaned = " ".join(str(text).split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 1].rstrip() + "…"

    def _render_retrieved_passages_block(
        self,
        retrieved_passages: Optional[List[Dict[str, Any]]],
        *,
        raw_client_context: Optional[str] = None,
        raw_founder_context: Optional[str] = None,
        max_passages: int = 5,
    ) -> str:
        """Render retrieved passages or fall back to contextual summaries."""

        lines: List[str] = []
        passages = retrieved_passages or []

        if passages:
            lines.append("CITE USING THE BRACKETED TAGS BELOW (e.g., [Source 1]).")
            for idx, passage in enumerate(passages[:max_passages], start=1):
                text = self._summarize_text_block(passage.get("text"), max_chars=420)
                if not text:
                    continue
                doc_id = passage.get("document_id") or passage.get("metadata", {}).get("document_id")
                chunk_id = passage.get("chunk_id") or passage.get("metadata", {}).get("chunk_id")
                source_type = passage.get("source_type") or passage.get("metadata", {}).get("source_type")
                meta_parts = []
                if doc_id:
                    meta_parts.append(f"doc={doc_id}")
                if chunk_id:
                    meta_parts.append(f"chunk={chunk_id}")
                if source_type:
                    meta_parts.append(f"type={source_type}")
                if meta_parts:
                    lines.append(f"[Source {idx}] {text}\n    ({', '.join(meta_parts)})")
                else:
                    lines.append(f"[Source {idx}] {text}")

            return "\n".join(lines)

        fallback_sections: List[str] = []
        client_summary = self._summarize_text_block(raw_client_context, max_chars=420)
        founder_summary = self._summarize_text_block(raw_founder_context, max_chars=420)

        if client_summary:
            fallback_sections.append(f"[Source F1 – Client Context] {client_summary}")
        if founder_summary:
            index = len(fallback_sections) + 1
            fallback_sections.append(f"[Source F{index} – Founder Context] {founder_summary}")

        if fallback_sections:
            lines.append(
                "No retrieved passages available. Use these summaries and cite using the [Source F#] tags."
            )
            lines.extend(fallback_sections)
        else:
            lines.append(
                "No retrieved passages or raw context available. Note the absence but maintain factual grounding."
            )

        return "\n".join(lines)
