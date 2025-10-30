"""Supervisor agent orchestrating the LangGraph-based workflow."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable

from langgraph.graph import StateGraph, END

from .base_agent import BaseAgent
from .science_extraction_agent import ScienceExtractionAgent
from .clinical_synthesis_agent import ClinicalSynthesisAgent
from .founder_voice_agent import FounderVoiceAgent
from .context_rag_agent import ContextRAGAgent
from .marketing_seo_agent import MarketingSEOAgent
from .validation_agent import ValidationAgent
from .educational_content_agent import EducationalContentAgent
from .rebellion_framework_agent import RebellionFrameworkAgent
from .workflow_state import WorkflowState

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Supervisor agent that coordinates the multi-agent workflow with LangGraph."""

    def __init__(self):
        super().__init__(
            name="EnlitensSupervisor",
            role="Multi-Agent System Orchestrator",
            model="qwen3:32b",
        )
        self.agents: Dict[str, BaseAgent] = {}
        self.processing_history: List[Dict[str, Any]] = []
        self.workflow_graph = None

        self.quality_thresholds = {
            "minimum_quality": 0.6,
            "good_quality": 0.8,
            "excellent_quality": 0.9,
        }

        self.retry_policy = {
            "max_attempts": 3,
            "base_delay": 2.0,
            "backoff_factor": 1.8,
        }

        self.doc_type_shortcuts: Dict[str, Dict[str, Any]] = {
            "science_only": {
                "skip": {
                    "clinical_synthesis",
                    "educational_content",
                    "rebellion_framework",
                    "founder_voice",
                    "marketing_seo",
                    "validation",
                }
            },
            "marketing_refresh": {
                "skip": {"science_extraction", "clinical_synthesis"},
            },
            "validation_only": {
                "skip": {
                    "science_extraction",
                    "clinical_synthesis",
                    "educational_content",
                    "rebellion_framework",
                    "founder_voice",
                    "marketing_seo",
                }
            },
        }

    async def initialize(self) -> bool:
        """Initialize supervisor and all specialized agents."""
        try:
            logger.info("Initializing Supervisor Agent and all specialized agents...")
            self.agents = {
                "science_extraction": ScienceExtractionAgent(),
                "clinical_synthesis": ClinicalSynthesisAgent(),
                "educational_content": EducationalContentAgent(),
                "rebellion_framework": RebellionFrameworkAgent(),
                "founder_voice": FounderVoiceAgent(),
                "context_rag": ContextRAGAgent(),
                "marketing_seo": MarketingSEOAgent(),
                "validation": ValidationAgent(),
            }

            init_tasks = [
                self._initialize_agent_with_retry(agent, name)
                for name, agent in self.agents.items()
            ]
            results = await asyncio.gather(*init_tasks, return_exceptions=True)

            failed_agents = []
            for (name, _), result in zip(self.agents.items(), results):
                if isinstance(result, Exception) or result is False:
                    failed_agents.append(name)

            if failed_agents:
                logger.error("Failed to initialize agents: %s", failed_agents)
                return False

            self.workflow_graph = self._build_graph()
            self.is_initialized = True
            logger.info("✅ Supervisor and all specialized agents initialized successfully")
            return True
        except Exception as exc:
            logger.error("Failed to initialize supervisor: %s", exc)
            return False

    async def _initialize_agent_with_retry(
        self, agent: BaseAgent, agent_name: str, max_retries: int = 3
    ) -> bool:
        for attempt in range(max_retries):
            try:
                if await agent.initialize():
                    logger.info("✅ Agent %s initialized on attempt %d", agent_name, attempt + 1)
                    return True
                logger.warning(
                    "Agent %s initialization failed on attempt %d", agent_name, attempt + 1
                )
            except Exception as exc:
                logger.warning(
                    "Agent %s initialization error on attempt %d: %s",
                    agent_name,
                    attempt + 1,
                    exc,
                )

            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))

        return False

    def _build_graph(self):
        graph = StateGraph(WorkflowState)

        graph.add_node("entry", self._entry_node)
        graph.add_node("science_extraction", self._science_node)
        graph.add_node("context_rag", self._context_node)
        graph.add_node("clinical_synthesis", self._clinical_node)
        graph.add_node("educational_content", self._education_node)
        graph.add_node("rebellion_framework", self._rebellion_node)
        graph.add_node("founder_voice", self._founder_node)
        graph.add_node("marketing_seo", self._marketing_node)
        graph.add_node("validation", self._validation_node)

        graph.set_entry_point("entry")
        graph.add_edge("entry", "science_extraction")
        graph.add_edge("entry", "context_rag")
        graph.add_edge("science_extraction", "clinical_synthesis")
        graph.add_edge("context_rag", "clinical_synthesis")
        graph.add_edge("clinical_synthesis", "educational_content")
        graph.add_edge("clinical_synthesis", "rebellion_framework")
        graph.add_edge("clinical_synthesis", "founder_voice")
        graph.add_edge("educational_content", "marketing_seo")
        graph.add_edge("rebellion_framework", "marketing_seo")
        graph.add_edge("founder_voice", "marketing_seo")
        graph.add_edge("educational_content", "validation")
        graph.add_edge("rebellion_framework", "validation")
        graph.add_edge("founder_voice", "validation")
        graph.add_edge("marketing_seo", "validation")
        graph.add_edge("validation", END)

        return graph.compile()

    async def process_document(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("🎯 Supervisor starting document processing: %s", payload.get("document_id"))
        state = WorkflowState(
            document_id=payload["document_id"],
            document_text=payload["document_text"],
            doc_type=payload.get("doc_type"),
            client_insights=payload.get("client_insights"),
            founder_insights=payload.get("founder_insights"),
            st_louis_context=payload.get("st_louis_context"),
            cache_prefix=payload.get("cache_prefix", payload.get("document_id", "doc")),
            cache_chunk_id=payload.get("cache_chunk_id", f"{payload.get('document_id', 'doc')}:root"),
        )

        final_state = await self.workflow_graph.ainvoke(state)
        return self._finalize_output(final_state)

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return await self.process_document(context)

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        required_fields = ["document_id", "processing_timestamp", "supervisor_status"]
        return all(field in output for field in required_fields)

    async def get_system_status(self) -> Dict[str, Any]:
        return {
            "supervisor": self.get_status(),
            "agents": {name: agent.get_status() for name, agent in self.agents.items()},
            "processing_history": self.processing_history[-10:],
            "total_processed": len([h for h in self.processing_history if h.get("success", False)]),
            "total_failed": len([h for h in self.processing_history if not h.get("success", False)]),
            "average_quality_score": sum(
                h.get("quality_score", 0) for h in self.processing_history
            )
            / max(len(self.processing_history), 1),
        }

    async def cleanup(self):
        logger.info("🧹 Cleaning up supervisor and all agents...")
        for name, agent in self.agents.items():
            try:
                await agent.cleanup()
                logger.info("✅ Agent %s cleaned up", name)
            except Exception as exc:
                logger.error("❌ Error cleaning up agent %s: %s", name, exc)
        self.agents.clear()
        self.processing_history.clear()
        self.is_initialized = False
        logger.info("✅ Supervisor cleanup completed")

    # ----- LangGraph Node Implementations -------------------------------------------------

    async def _entry_node(self, state: WorkflowState) -> Dict[str, Any]:
        logger.info("🚦 Entry node activated for %s", state.document_id)
        shortcut = self.doc_type_shortcuts.get(state.doc_type or "", {})
        skip_nodes = set(shortcut.get("skip", set()))
        metadata = {
            "doc_type": state.doc_type,
            "shortcut_applied": bool(shortcut),
        }
        return {
            "stage": "entry",
            "start_timestamp": datetime.utcnow(),
            "skip_nodes": skip_nodes,
            "metadata": {**state.metadata, **metadata},
        }

    async def _science_node(self, state: WorkflowState) -> Dict[str, Any]:
        if "science_extraction" in state.skip_nodes:
            logger.info("⏭️ Skipping science extraction due to doc_type route")
            return {
                "stage": "science_extraction_skipped",
                "science_result": state.science_result or {},
                "completed_nodes": {**state.completed_nodes, "science_extraction": "skipped"},
            }

        payload = {"document_text": state.document_text}
        result = await self._run_agent_with_retry("science_extraction", state, payload)
        return self._merge_results(state, "science_extraction", result, "science_result")

    async def _context_node(self, state: WorkflowState) -> Dict[str, Any]:
        if "context_rag" in state.skip_nodes:
            logger.info("⏭️ Skipping context retrieval due to doc_type route")
            return {
                "stage": "context_skipped",
                "context_result": state.context_result or {},
                "completed_nodes": {**state.completed_nodes, "context_rag": "skipped"},
            }

        payload = {
            "enhanced_data": state.intermediate_results,
            "st_louis_context": state.st_louis_context or {},
        }
        result = await self._run_agent_with_retry("context_rag", state, payload)
        return self._merge_results(state, "context_rag", result, "context_result")

    async def _clinical_node(self, state: WorkflowState) -> Dict[str, Any]:
        if "clinical_synthesis" in state.skip_nodes:
            logger.info("⏭️ Skipping clinical synthesis due to doc_type route")
            return {
                "stage": "clinical_skipped",
                "clinical_result": state.clinical_result or {},
                "completed_nodes": {**state.completed_nodes, "clinical_synthesis": "skipped"},
            }

        if state.science_result is None and "science_extraction" not in state.skip_nodes:
            logger.debug("Waiting for science extraction before clinical synthesis")
            return {"stage": "clinical_waiting_science"}

        payload = {
            "science_data": state.science_result or {},
            "document_text": state.document_text,
            "context_result": state.context_result,
        }
        result = await self._run_agent_with_retry("clinical_synthesis", state, payload)
        return self._merge_results(state, "clinical_synthesis", result, "clinical_result")

    async def _education_node(self, state: WorkflowState) -> Dict[str, Any]:
        if "educational_content" in state.skip_nodes:
            logger.info("⏭️ Skipping educational content due to doc_type route")
            return {
                "stage": "education_skipped",
                "educational_result": state.educational_result or {},
                "completed_nodes": {**state.completed_nodes, "educational_content": "skipped"},
            }

        if state.clinical_result is None and "clinical_synthesis" not in state.skip_nodes:
            logger.debug("Waiting for clinical synthesis before education")
            return {"stage": "education_waiting_clinical"}

        payload = {
            "document_text": state.document_text,
            "science_data": state.science_result or {},
            "clinical_content": state.clinical_result or {},
        }
        result = await self._run_agent_with_retry("educational_content", state, payload)
        return self._merge_results(state, "educational_content", result, "educational_result")

    async def _rebellion_node(self, state: WorkflowState) -> Dict[str, Any]:
        if "rebellion_framework" in state.skip_nodes:
            logger.info("⏭️ Skipping rebellion framework due to doc_type route")
            return {
                "stage": "rebellion_skipped",
                "rebellion_result": state.rebellion_result or {},
                "completed_nodes": {**state.completed_nodes, "rebellion_framework": "skipped"},
            }

        if state.clinical_result is None and "clinical_synthesis" not in state.skip_nodes:
            logger.debug("Waiting for clinical synthesis before rebellion framework")
            return {"stage": "rebellion_waiting_clinical"}

        payload = {
            "document_text": state.document_text,
            "science_data": state.science_result or {},
            "clinical_content": state.clinical_result or {},
        }
        result = await self._run_agent_with_retry("rebellion_framework", state, payload)
        return self._merge_results(state, "rebellion_framework", result, "rebellion_result")

    async def _founder_node(self, state: WorkflowState) -> Dict[str, Any]:
        if "founder_voice" in state.skip_nodes:
            logger.info("⏭️ Skipping founder voice due to doc_type route")
            return {
                "stage": "founder_skipped",
                "founder_voice_result": state.founder_voice_result or {},
                "completed_nodes": {**state.completed_nodes, "founder_voice": "skipped"},
            }

        if state.clinical_result is None and "clinical_synthesis" not in state.skip_nodes:
            logger.debug("Waiting for clinical synthesis before founder voice")
            return {"stage": "founder_waiting_clinical"}

        payload = {
            "clinical_data": state.clinical_result or {},
            "enhanced_data": state.intermediate_results,
            "document_id": state.document_id,
            "document_text": state.document_text,
        }
        result = await self._run_agent_with_retry("founder_voice", state, payload)
        return self._merge_results(state, "founder_voice", result, "founder_voice_result")

    async def _marketing_node(self, state: WorkflowState) -> Dict[str, Any]:
        if "marketing_seo" in state.skip_nodes:
            if not state.marketing_completed:
                logger.info("⏭️ Skipping marketing due to doc_type route")
            return {
                "stage": "marketing_skipped",
                "marketing_result": state.marketing_result or {},
                "marketing_completed": True,
                "completed_nodes": {**state.completed_nodes, "marketing_seo": "skipped"},
            }

        required = [
            state.educational_result or {},
            state.rebellion_result or {},
            state.founder_voice_result or {},
        ]
        if not all(required):
            logger.debug("Waiting for creative fan-out before marketing")
            return {"stage": "marketing_waiting_creatives"}

        if state.marketing_completed:
            return {"stage": "marketing_done"}

        payload = {"final_context": state.intermediate_results}
        result = await self._run_agent_with_retry("marketing_seo", state, payload)
        merged = self._merge_results(state, "marketing_seo", result, "marketing_result")
        merged.update({"marketing_completed": True})
        return merged

    async def _validation_node(self, state: WorkflowState) -> Dict[str, Any]:
        if "validation" in state.skip_nodes:
            if not state.validation_completed:
                logger.info("⏭️ Skipping validation due to doc_type route")
            return {
                "stage": "validation_skipped",
                "validation_result": state.validation_result or {},
                "validation_completed": True,
                "completed_nodes": {**state.completed_nodes, "validation": "skipped"},
                "end_timestamp": datetime.utcnow(),
            }

        marketing_needed = "marketing_seo" not in state.skip_nodes
        if marketing_needed and not state.marketing_completed:
            logger.debug("Waiting for marketing completion before validation")
            return {"stage": "validation_waiting_marketing"}

        if state.validation_completed:
            return {"stage": "validation_done", "end_timestamp": datetime.utcnow()}

        payload = {"complete_output": state.intermediate_results}
        result = await self._run_agent_with_retry("validation", state, payload)
        merged = self._merge_results(state, "validation", result, "validation_result")
        merged.update({"validation_completed": True, "end_timestamp": datetime.utcnow()})
        return merged

    # ----- Helper methods -----------------------------------------------------------------

    def _merge_results(
        self,
        state: WorkflowState,
        node_name: str,
        result: Dict[str, Any],
        target_field: str,
    ) -> Dict[str, Any]:
        if not result:
            logger.warning("⚠️ %s returned empty results", node_name)
            return {
                "stage": f"{node_name}_empty",
                target_field: getattr(state, target_field) or {},
                "completed_nodes": {**state.completed_nodes, node_name: "empty"},
            }

        merged_results = {**state.intermediate_results, **result}
        return {
            "stage": f"{node_name}_completed",
            target_field: result,
            "intermediate_results": merged_results,
            "completed_nodes": {**state.completed_nodes, node_name: "done"},
        }

    async def _run_agent_with_retry(
        self,
        agent_name: str,
        state: WorkflowState,
        payload: Dict[str, Any],
        success_predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Dict[str, Any]:
        agent = self.agents.get(agent_name)
        if not agent:
            logger.error("Agent %s not found", agent_name)
            return {}

        max_attempts = self.retry_policy["max_attempts"]
        delay = self.retry_policy["base_delay"]
        factor = self.retry_policy["backoff_factor"]

        last_result: Dict[str, Any] = {}
        for attempt in range(1, max_attempts + 1):
            agent_context = self._build_agent_context(state, payload, agent_name, attempt)
            logger.info("🔄 Executing %s attempt %d/%d", agent_name, attempt, max_attempts)
            result = await agent.execute(agent_context)
            state.record_attempt(agent_name)

            if result and (success_predicate is None or success_predicate(result)):
                return result

            last_result = result or {}
            if attempt < max_attempts:
                await asyncio.sleep(delay)
                delay *= factor

        logger.warning("⚠️ %s exhausted retries", agent_name)
        return last_result

    def _build_agent_context(
        self,
        state: WorkflowState,
        payload: Dict[str, Any],
        agent_name: str,
        attempt: int,
    ) -> Dict[str, Any]:
        base_context = {
            "document_id": state.document_id,
            "document_text": state.document_text,
            "client_insights": state.client_insights,
            "founder_insights": state.founder_insights,
            "st_louis_context": state.st_louis_context,
            "processing_stage": state.stage,
            "intermediate_results": dict(state.intermediate_results),
            "cache_prefix": f"{state.cache_prefix}:{agent_name}",
            "cache_chunk_id": state.cache_chunk_id,
            "retry_attempt": attempt,
        }
        base_context.update(payload)
        return base_context

    def _finalize_output(self, state: WorkflowState) -> Dict[str, Any]:
        processing_time = None
        if state.start_timestamp and state.end_timestamp:
            processing_time = (state.end_timestamp - state.start_timestamp).total_seconds()

        quality_score = 0.0
        confidence_score = 0.0
        validation_passed = False
        if state.validation_result:
            quality_score = state.validation_result.get("quality_scores", {}).get(
                "overall_quality", 0.0
            )
            confidence_score = state.validation_result.get("confidence_scoring", {}).get(
                "confidence_score", 0.0
            )
            validation_passed = quality_score >= self.quality_thresholds["minimum_quality"]

        final_output = {
            "document_id": state.document_id,
            "document_text": state.document_text,
            "processing_timestamp": datetime.utcnow().isoformat(),
            "processing_time_seconds": processing_time,
            "supervisor_status": "completed" if validation_passed else "completed_with_issues",
            "agent_outputs": state.intermediate_results,
            "quality_score": quality_score,
            "confidence_score": confidence_score,
            "validation_passed": validation_passed,
            "retry_counts": state.attempt_counters,
            "completed_nodes": state.completed_nodes,
            "metadata": state.as_dict().get("metadata", {}),
        }

        self.processing_history.append(
            {
                "document_id": state.document_id,
                "start_time": state.start_timestamp.isoformat() if state.start_timestamp else None,
                "end_time": state.end_timestamp.isoformat() if state.end_timestamp else None,
                "processing_time": processing_time,
                "success": validation_passed,
                "quality_score": quality_score,
            }
        )

        logger.info(
            "✅ Document %s processed. Quality %.2f Confidence %.2f",
            state.document_id,
            quality_score,
            confidence_score,
        )
        return final_output
