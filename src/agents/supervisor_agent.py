"""Supervisor agent orchestrating the LangGraph-based workflow."""

from __future__ import annotations

import asyncio
import copy
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Iterable

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
from .workflow_state import WorkflowState, create_initial_state, record_attempt, as_dict

logger = logging.getLogger(__name__)

PIPELINE_AGENTS: List[str] = [
    "context_rag",
    "science_extraction",
    "clinical_synthesis",
    "educational_content",
    "rebellion_framework",
    "founder_voice",
    "marketing_seo",
    "validation",
]


class SupervisorAgent(BaseAgent):
    """Supervisor agent that coordinates the multi-agent workflow with LangGraph."""

    def __init__(self):
        super().__init__(
            name="EnlitensSupervisor",
            role="Multi-Agent System Orchestrator",
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

        plan_data = self._generate_execution_plan(payload)

        state = create_initial_state(
            document_id=payload["document_id"],
            document_text=payload["document_text"],
            doc_type=payload.get("doc_type"),
            client_insights=payload.get("client_insights"),
            founder_insights=payload.get("founder_insights"),
            st_louis_context=payload.get("st_louis_context"),
            cache_prefix=payload.get("cache_prefix", payload.get("document_id", "doc")),
            cache_chunk_id=payload.get("cache_chunk_id", f"{payload.get('document_id', 'doc')}:root"),
        )

        state["skip_nodes"].update(plan_data["skip_nodes"])
        metadata = state.get("metadata", {})
        metadata.update(
            {
                "execution_plan": plan_data["plan"],
                "plan_created_at": datetime.utcnow().isoformat(),
                "plan_notes": plan_data["notes"],
                "plan_status": "pending",
            }
        )
        state["metadata"] = metadata

        final_state = await self.workflow_graph.ainvoke(state)
        final_state = await self._ensure_plan_completion(final_state)
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
        logger.info("🚦 Entry node activated for %s", state["document_id"])
        shortcut = self.doc_type_shortcuts.get(state.get("doc_type") or "", {})
        skip_nodes = set(shortcut.get("skip", set()))
        metadata = {
            "doc_type": state.get("doc_type"),
            "shortcut_applied": bool(shortcut),
        }
        if skip_nodes:
            state["skip_nodes"].update(skip_nodes)
            for node in skip_nodes:
                self._mark_plan_step(state, node, "skipped", "doc_type shortcut applied")
        self._mark_plan_started(state)
        return {
            "stage": "entry",
            "start_timestamp": datetime.utcnow(),
            "skip_nodes": skip_nodes,
            "metadata": self._metadata_with_plan(state, metadata),
        }

    async def _science_node(self, state: WorkflowState) -> Dict[str, Any]:
        if "science_extraction" in state.get("skip_nodes", set()):
            logger.info("⏭️ Skipping science extraction due to doc_type route")
            self._mark_plan_step(state, "science_extraction", "skipped", "Skip flag detected")
            return {
                "stage": "science_extraction_skipped",
                "science_result": state.get("science_result") or {},
                "completed_nodes": {**state.get("completed_nodes", {}), "science_extraction": "skipped"},
                "metadata": self._metadata_with_plan(state),
            }

        payload = {"document_text": state["document_text"]}
        result = await self._run_agent_with_retry("science_extraction", state, payload)
        return self._merge_results(state, "science_extraction", result, "science_result")

    async def _context_node(self, state: WorkflowState) -> Dict[str, Any]:
        if "context_rag" in state.get("skip_nodes", set()):
            logger.info("⏭️ Skipping context retrieval due to doc_type route")
            self._mark_plan_step(state, "context_rag", "skipped", "Skip flag detected")
            return {
                "stage": "context_skipped",
                "context_result": state.get("context_result") or {},
                "completed_nodes": {**state.get("completed_nodes", {}), "context_rag": "skipped"},
                "metadata": self._metadata_with_plan(state),
            }

        payload = {
            "enhanced_data": state.get("intermediate_results", {}),
            "st_louis_context": state.get("st_louis_context") or {},
        }
        result = await self._run_agent_with_retry("context_rag", state, payload)
        return self._merge_results(state, "context_rag", result, "context_result")

    async def _clinical_node(self, state: WorkflowState) -> Dict[str, Any]:
        if "clinical_synthesis" in state.get("skip_nodes", set()):
            logger.info("⏭️ Skipping clinical synthesis due to doc_type route")
            self._mark_plan_step(state, "clinical_synthesis", "skipped", "Skip flag detected")
            return {
                "stage": "clinical_skipped",
                "clinical_result": state.get("clinical_result") or {},
                "completed_nodes": {**state.get("completed_nodes", {}), "clinical_synthesis": "skipped"},
                "metadata": self._metadata_with_plan(state),
            }

        if state.get("science_result") is None and "science_extraction" not in state.get("skip_nodes", set()):
            logger.debug("Waiting for science extraction before clinical synthesis")
            return {"stage": "clinical_waiting_science"}

        payload = {
            "science_data": state.get("science_result") or {},
            "document_text": state["document_text"],
            "context_result": state.get("context_result"),
        }
        result = await self._run_agent_with_retry("clinical_synthesis", state, payload)
        return self._merge_results(state, "clinical_synthesis", result, "clinical_result")

    async def _education_node(self, state: WorkflowState) -> Dict[str, Any]:
        if "educational_content" in state.get("skip_nodes", set()):
            logger.info("⏭️ Skipping educational content due to doc_type route")
            self._mark_plan_step(state, "educational_content", "skipped", "Skip flag detected")
            return {
                "stage": "education_skipped",
                "educational_result": state.get("educational_result") or {},
                "completed_nodes": {**state.get("completed_nodes", {}), "educational_content": "skipped"},
                "metadata": self._metadata_with_plan(state),
            }

        if state.get("clinical_result") is None and "clinical_synthesis" not in state.get("skip_nodes", set()):
            logger.debug("Waiting for clinical synthesis before education")
            return {"stage": "education_waiting_clinical"}

        payload = {
            "document_text": state["document_text"],
            "science_data": state.get("science_result") or {},
            "clinical_content": state.get("clinical_result") or {},
        }
        result = await self._run_agent_with_retry("educational_content", state, payload)
        return self._merge_results(state, "educational_content", result, "educational_result")

    async def _rebellion_node(self, state: WorkflowState) -> Dict[str, Any]:
        if "rebellion_framework" in state.get("skip_nodes", set()):
            logger.info("⏭️ Skipping rebellion framework due to doc_type route")
            self._mark_plan_step(state, "rebellion_framework", "skipped", "Skip flag detected")
            return {
                "stage": "rebellion_skipped",
                "rebellion_result": state.get("rebellion_result") or {},
                "completed_nodes": {**state.get("completed_nodes", {}), "rebellion_framework": "skipped"},
                "metadata": self._metadata_with_plan(state),
            }

        if state.get("clinical_result") is None and "clinical_synthesis" not in state.get("skip_nodes", set()):
            logger.debug("Waiting for clinical synthesis before rebellion framework")
            return {"stage": "rebellion_waiting_clinical"}

        payload = {
            "document_text": state["document_text"],
            "science_data": state.get("science_result") or {},
            "clinical_content": state.get("clinical_result") or {},
        }
        result = await self._run_agent_with_retry("rebellion_framework", state, payload)
        return self._merge_results(state, "rebellion_framework", result, "rebellion_result")

    async def _founder_node(self, state: WorkflowState) -> Dict[str, Any]:
        if "founder_voice" in state.get("skip_nodes", set()):
            logger.info("⏭️ Skipping founder voice due to doc_type route")
            self._mark_plan_step(state, "founder_voice", "skipped", "Skip flag detected")
            return {
                "stage": "founder_skipped",
                "founder_voice_result": state.get("founder_voice_result") or {},
                "completed_nodes": {**state.get("completed_nodes", {}), "founder_voice": "skipped"},
                "metadata": self._metadata_with_plan(state),
            }

        if state.get("clinical_result") is None and "clinical_synthesis" not in state.get("skip_nodes", set()):
            logger.debug("Waiting for clinical synthesis before founder voice")
            return {"stage": "founder_waiting_clinical"}

        payload = {
            "clinical_data": state.get("clinical_result") or {},
            "enhanced_data": state.get("intermediate_results", {}),
            "document_id": state["document_id"],
            "document_text": state["document_text"],
        }
        result = await self._run_agent_with_retry("founder_voice", state, payload)
        return self._merge_results(state, "founder_voice", result, "founder_voice_result")

    async def _marketing_node(self, state: WorkflowState) -> Dict[str, Any]:
        if "marketing_seo" in state.get("skip_nodes", set()):
            if not state.get("marketing_completed", False):
                logger.info("⏭️ Skipping marketing due to doc_type route")
            self._mark_plan_step(state, "marketing_seo", "skipped", "Skip flag detected")
            return {
                "stage": "marketing_skipped",
                "marketing_result": state.get("marketing_result") or {},
                "marketing_completed": True,
                "completed_nodes": {**state.get("completed_nodes", {}), "marketing_seo": "skipped"},
                "metadata": self._metadata_with_plan(state),
            }

        required = [
            state.get("educational_result") or {},
            state.get("rebellion_result") or {},
            state.get("founder_voice_result") or {},
        ]
        if not all(required):
            logger.debug("Waiting for creative fan-out before marketing")
            return {"stage": "marketing_waiting_creatives"}

        if state.get("marketing_completed", False):
            return {"stage": "marketing_done"}

        payload = {"final_context": state.get("intermediate_results", {})}
        result = await self._run_agent_with_retry("marketing_seo", state, payload)
        merged = self._merge_results(state, "marketing_seo", result, "marketing_result")
        merged.update({"marketing_completed": True})
        return merged

    async def _validation_node(self, state: WorkflowState) -> Dict[str, Any]:
        if "validation" in state.get("skip_nodes", set()):
            if not state.get("validation_completed", False):
                logger.info("⏭️ Skipping validation due to doc_type route")
            self._mark_plan_step(state, "validation", "skipped", "Skip flag detected")
            return {
                "stage": "validation_skipped",
                "validation_result": state.get("validation_result") or {},
                "validation_completed": True,
                "completed_nodes": {**state.get("completed_nodes", {}), "validation": "skipped"},
                "end_timestamp": datetime.utcnow(),
                "metadata": self._metadata_with_plan(state),
            }

        marketing_needed = "marketing_seo" not in state.get("skip_nodes", set())
        if marketing_needed and not state.get("marketing_completed", False):
            logger.debug("Waiting for marketing completion before validation")
            return {"stage": "validation_waiting_marketing"}

        if state.get("validation_completed", False):
            return {"stage": "validation_done", "end_timestamp": datetime.utcnow()}

        payload = {"complete_output": state.get("intermediate_results", {})}
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
            self._mark_plan_step(state, node_name, "needs_follow_up", "Empty result returned")
            return {
                "stage": f"{node_name}_empty",
                target_field: state.get(target_field) or {},
                "completed_nodes": {**state.get("completed_nodes", {}), node_name: "empty"},
                "metadata": self._metadata_with_plan(state),
            }

        merged_results = {**state.get("intermediate_results", {}), **result}
        self._mark_plan_step(state, node_name, "completed")
        return {
            "stage": f"{node_name}_completed",
            target_field: result,
            "intermediate_results": merged_results,
            "completed_nodes": {**state.get("completed_nodes", {}), node_name: "done"},
            "metadata": self._metadata_with_plan(state),
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
        self._mark_plan_step(state, agent_name, "in_progress")
        for attempt in range(1, max_attempts + 1):
            agent_context = self._build_agent_context(state, payload, agent_name, attempt)
            logger.info("🔄 Executing %s attempt %d/%d", agent_name, attempt, max_attempts)
            result = await agent.execute(agent_context)
            record_attempt(state, agent_name)

            if result and (success_predicate is None or success_predicate(result)):
                return result

            last_result = result or {}
            if attempt < max_attempts:
                await asyncio.sleep(delay)
                delay *= factor

        logger.warning("⚠️ %s exhausted retries", agent_name)
        self._mark_plan_step(state, agent_name, "failed", "Retries exhausted")
        return last_result

    def _build_agent_context(
        self,
        state: WorkflowState,
        payload: Dict[str, Any],
        agent_name: str,
        attempt: int,
    ) -> Dict[str, Any]:
        base_context = {
            "document_id": state["document_id"],
            "document_text": state["document_text"],
            "client_insights": state.get("client_insights"),
            "founder_insights": state.get("founder_insights"),
            "st_louis_context": state.get("st_louis_context"),
            "processing_stage": state.get("stage"),
            "intermediate_results": dict(state.get("intermediate_results", {})),
            "cache_prefix": f"{state.get('cache_prefix')}:{agent_name}",
            "cache_chunk_id": state.get("cache_chunk_id"),
            "retry_attempt": attempt,
        }
        plan_step = None
        for step in state.get("metadata", {}).get("execution_plan", []):
            if step.get("agent") == agent_name:
                plan_step = step
                break
        if plan_step:
            base_context["plan_step"] = plan_step
        base_context.update(payload)
        return base_context

    def _finalize_output(self, state: WorkflowState) -> Dict[str, Any]:
        processing_time = None
        if state.get("start_timestamp") and state.get("end_timestamp"):
            processing_time = (state.get("end_timestamp") - state.get("start_timestamp")).total_seconds()

        quality_score = 0.0
        confidence_score = 0.0
        validation_passed = False
        if state.get("validation_result"):
            quality_score = state.get("validation_result").get("quality_scores", {}).get(
                "overall_quality", 0.0
            )
            confidence_score = state.get("validation_result").get("confidence_scoring", {}).get(
                "confidence_score", 0.0
            )
            validation_passed = quality_score >= self.quality_thresholds["minimum_quality"]

        final_output = {
            "document_id": state["document_id"],
            "document_text": state["document_text"],
            "processing_timestamp": datetime.utcnow().isoformat(),
            "processing_time_seconds": processing_time,
            "supervisor_status": "completed" if validation_passed else "completed_with_issues",
            "agent_outputs": state.get("intermediate_results", {}),
            "quality_score": quality_score,
            "confidence_score": confidence_score,
            "validation_passed": validation_passed,
            "retry_metadata": state.get("validation_result", {}).get("retry_metadata", {}),
            "retry_counts": state.get("attempt_counters", {}),
            "completed_nodes": state.get("completed_nodes", {}),
            "metadata": as_dict(state).get("metadata", {}),
        }

        self.processing_history.append(
            {
                "document_id": state["document_id"],
                "start_time": state.get("start_timestamp").isoformat() if state.get("start_timestamp") else None,
                "end_time": state.get("end_timestamp").isoformat() if state.get("end_timestamp") else None,
                "processing_time": processing_time,
                "success": validation_passed,
                "quality_score": quality_score,
            }
        )

        logger.info(
            "✅ Document %s processed. Quality %.2f Confidence %.2f",
            state["document_id"],
            quality_score,
            confidence_score,
        )
        return final_output

    # ----- Planning helpers ---------------------------------------------------------------

    def _generate_execution_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        doc_type = payload.get("doc_type") or "default"
        shortcut_skips = set(self.doc_type_shortcuts.get(doc_type, {}).get("skip", set()))
        plan: List[Dict[str, Any]] = []
        notes: Dict[str, Any] = {}

        def add_step(agent: str, description: str) -> None:
            if agent in shortcut_skips or agent not in PIPELINE_AGENTS:
                return
            plan.append(
                {
                    "step": len(plan) + 1,
                    "agent": agent,
                    "description": description,
                    "status": "pending",
                }
            )

        is_complex = self._is_complex_query(payload)
        notes["complexity_reason"] = self._complexity_reason(payload, is_complex)

        if is_complex or payload.get("st_louis_context") or payload.get("client_insights"):
            add_step(
                "context_rag",
                "Gather constitution-aligned context and supporting research",
            )

        add_step("science_extraction", "Extract constitution-filtered scientific facts")
        add_step("clinical_synthesis", "Compose constitution-aligned clinical synthesis")

        creative_needed = doc_type not in {"science_only", "validation_only"}
        if creative_needed:
            add_step("educational_content", "Translate findings into educational framing")
            add_step("rebellion_framework", "Map insights to rebellion framework guidance")
            add_step("founder_voice", "Apply founder voice tone and narrative")
            add_step("marketing_seo", "Craft marketing-forward summary and SEO hooks")

        add_step("validation", "Run constitutional validation and quality checks")

        planned_agents = {step["agent"] for step in plan}
        skip_nodes = (set(PIPELINE_AGENTS) - planned_agents) | shortcut_skips

        notes["planned_agents"] = sorted(planned_agents)
        notes["skipped_agents"] = sorted(skip_nodes)

        return {"plan": plan, "skip_nodes": skip_nodes, "notes": notes}

    def _is_complex_query(self, payload: Dict[str, Any]) -> bool:
        document_text = payload.get("document_text") or ""
        word_count = len(document_text.split())
        question_count = document_text.count("?")
        has_multiple_sections = "\n\n" in document_text or "###" in document_text
        return (
            word_count > 400
            or question_count > 1
            or has_multiple_sections
            or bool(payload.get("client_insights"))
        )

    def _complexity_reason(self, payload: Dict[str, Any], is_complex: bool) -> str:
        if not is_complex:
            return "baseline_plan"
        reasons: List[str] = []
        document_text = payload.get("document_text") or ""
        if len(document_text.split()) > 400:
            reasons.append("lengthy_source")
        if document_text.count("?") > 1:
            reasons.append("multi_question")
        if "\n\n" in document_text or "###" in document_text:
            reasons.append("multi_section")
        if payload.get("client_insights"):
            reasons.append("client_context_present")
        if not reasons:
            reasons.append("heuristic_complexity")
        return ",".join(reasons)

    def _mark_plan_started(self, state: WorkflowState) -> None:
        metadata = state.get("metadata", {})
        if metadata.get("plan_status") == "pending":
            metadata = copy.deepcopy(metadata)
            metadata["plan_status"] = "in_progress"
            state["metadata"] = metadata

    def _mark_plan_step(
        self,
        state: WorkflowState,
        agent_name: str,
        status: str,
        note: Optional[str] = None,
    ) -> None:
        if not agent_name:
            return
        metadata = state.get("metadata", {})
        plan = metadata.get("execution_plan")
        if not plan:
            return
        updated = False
        new_plan: List[Dict[str, Any]] = []
        for step in plan:
            if step.get("agent") == agent_name:
                if step.get("status") == status and not note:
                    new_plan.append(step)
                    continue
                step_copy = copy.deepcopy(step)
                step_copy["status"] = status
                if note:
                    step_copy["note"] = note
                step_copy["updated_at"] = datetime.utcnow().isoformat()
                new_plan.append(step_copy)
                updated = True
            else:
                new_plan.append(step)
        if updated:
            new_metadata = copy.deepcopy(metadata)
            new_metadata["execution_plan"] = new_plan
            state["metadata"] = new_metadata

    def _metadata_with_plan(
        self, state: WorkflowState, extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        metadata = copy.deepcopy(state.get("metadata", {}))
        if extra:
            metadata.update(extra)
        return metadata

    async def _ensure_plan_completion(self, state: WorkflowState) -> WorkflowState:
        unfinished_agents = self._unfinished_plan_agents(state)
        if unfinished_agents:
            logger.info("🔁 Executing follow-up tasks for incomplete plan steps: %s", unfinished_agents)
        for agent_name in unfinished_agents:
            follow_up_result = await self._execute_follow_up(agent_name, state)
            if follow_up_result:
                self._apply_node_result(state, follow_up_result)
        metadata = state.get("metadata", {})
        plan = metadata.get("execution_plan") or []
        remaining = [step for step in plan if step.get("status") not in {"completed", "skipped"}]
        metadata = copy.deepcopy(metadata)
        metadata["plan_status"] = "completed" if not remaining else "incomplete"
        metadata["plan_completed_at"] = datetime.utcnow().isoformat()
        if remaining:
            metadata["plan_remaining_agents"] = [step.get("agent") for step in remaining]
        state["metadata"] = metadata
        return state

    def _unfinished_plan_agents(self, state: WorkflowState) -> List[str]:
        metadata = state.get("metadata", {})
        plan = metadata.get("execution_plan") or []
        return [step.get("agent") for step in plan if step.get("status") not in {"completed", "skipped"}]

    async def _execute_follow_up(
        self, agent_name: str, state: WorkflowState
    ) -> Optional[Dict[str, Any]]:
        handlers: Dict[str, Callable[[WorkflowState], Any]] = {
            "context_rag": self._context_node,
            "science_extraction": self._science_node,
            "clinical_synthesis": self._clinical_node,
            "educational_content": self._education_node,
            "rebellion_framework": self._rebellion_node,
            "founder_voice": self._founder_node,
            "marketing_seo": self._marketing_node,
            "validation": self._validation_node,
        }
        handler = handlers.get(agent_name)
        if not handler:
            logger.warning("No follow-up handler registered for %s", agent_name)
            return None
        try:
            result = await handler(state)
            if result and result.get("stage", "").endswith("completed"):
                self._mark_plan_step(state, agent_name, "completed", "Follow-up execution")
            return result
        except Exception as exc:
            logger.error("Follow-up execution for %s failed: %s", agent_name, exc)
            self._mark_plan_step(state, agent_name, "failed", f"Follow-up error: {exc}")
            return None

    def _apply_node_result(self, state: WorkflowState, result: Dict[str, Any]) -> None:
        if not result:
            return
        for key, value in result.items():
            if key == "skip_nodes" and isinstance(value, Iterable):
                state["skip_nodes"].update(value)
            elif key == "completed_nodes" and isinstance(value, dict):
                state.setdefault("completed_nodes", {}).update(value)
            elif key == "intermediate_results" and isinstance(value, dict):
                state.setdefault("intermediate_results", {}).update(value)
            else:
                state[key] = value
