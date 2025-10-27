"""
Supervisor Agent for the Enlitens Multi-Agent System.

This agent coordinates and orchestrates the work of all specialized agents,
ensuring high-quality output and proper resource management.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from .base_agent import BaseAgent
from .science_extraction_agent import ScienceExtractionAgent
from .clinical_synthesis_agent import ClinicalSynthesisAgent
from .founder_voice_agent import FounderVoiceAgent
from .context_rag_agent import ContextRAGAgent
from .marketing_seo_agent import MarketingSEOAgent
from .validation_agent import ValidationAgent

logger = logging.getLogger(__name__)

@dataclass
class ProcessingContext:
    """Context information passed between agents."""
    document_id: str
    document_text: str
    client_insights: Optional[Dict[str, Any]] = None
    founder_insights: Optional[Dict[str, Any]] = None
    st_louis_context: Optional[Dict[str, Any]] = None
    processing_stage: str = "initial"
    intermediate_results: Dict[str, Any] = None

    def __post_init__(self):
        if self.intermediate_results is None:
            self.intermediate_results = {}

class SupervisorAgent(BaseAgent):
    """
    Supervisor agent that orchestrates the entire multi-agent system.
    """

    def __init__(self):
        super().__init__(
            name="EnlitensSupervisor",
            role="Multi-Agent System Orchestrator",
            model="qwen3:32b"
        )
        self.agents: Dict[str, BaseAgent] = {}
        self.processing_history: List[Dict[str, Any]] = []
        self.max_concurrent_agents = 2  # Limit concurrent processing

        # Quality and retry configuration
        self.quality_thresholds = {
            "minimum_quality": 0.6,
            "good_quality": 0.8,
            "excellent_quality": 0.9
        }

        self.retry_config = {
            "max_retries": 3,
            "retry_delay": 3,  # seconds between retries
            "quality_improvement_threshold": 0.1,
            "confidence_threshold": 0.7
        }

    async def initialize(self) -> bool:
        """Initialize the supervisor and all specialized agents."""
        try:
            logger.info("Initializing Supervisor Agent and all specialized agents...")

            # Initialize specialized agents
            self.agents = {
                "science_extraction": ScienceExtractionAgent(),
                "clinical_synthesis": ClinicalSynthesisAgent(),
                "founder_voice": FounderVoiceAgent(),
                "context_rag": ContextRAGAgent(),
                "marketing_seo": MarketingSEOAgent(),
                "validation": ValidationAgent()
            }

            # Initialize all agents
            init_tasks = []
            for agent_name, agent in self.agents.items():
                init_tasks.append(self._initialize_agent_with_retry(agent, agent_name))

            results = await asyncio.gather(*init_tasks, return_exceptions=True)

            # Check if all initializations succeeded
            failed_agents = []
            for i, result in enumerate(results):
                agent_name = list(self.agents.keys())[i]
                if isinstance(result, Exception):
                    logger.error(f"Failed to initialize agent {agent_name}: {result}")
                    failed_agents.append(agent_name)

            if failed_agents:
                logger.error(f"Failed to initialize agents: {failed_agents}")
                return False

            self.is_initialized = True
            logger.info("✅ Supervisor and all specialized agents initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize supervisor: {e}")
            return False

    async def _initialize_agent_with_retry(self, agent: BaseAgent, agent_name: str, max_retries: int = 3) -> bool:
        """Initialize an agent with retry logic."""
        for attempt in range(max_retries):
            try:
                success = await agent.initialize()
                if success:
                    logger.info(f"✅ Agent {agent_name} initialized on attempt {attempt + 1}")
                    return True
                else:
                    logger.warning(f"Agent {agent_name} initialization failed on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Agent {agent_name} initialization error on attempt {attempt + 1}: {e}")

            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

        return False

    async def process_document(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Process a single document through the complete multi-agent pipeline.
        """
        logger.info(f"🎯 Supervisor starting document processing: {context.document_id}")
        logger.info(f"📝 Document text length: {len(context.document_text)} characters")

        start_time = datetime.now()

        try:
            # Stage 1: Science and Research Extraction
            logger.info(f"🔬 Stage 1: Starting science extraction")
            context.processing_stage = "science_extraction"
            science_results = await self._execute_agent_sequentially(
                "science_extraction", context,
                {"document_text": context.document_text}
            )
            logger.info(f"✅ Stage 1: Science extraction completed")

            # Stage 2: Clinical Synthesis
            context.processing_stage = "clinical_synthesis"
            context.intermediate_results.update(science_results)
            clinical_results = await self._execute_agent_sequentially(
                "clinical_synthesis", context,
                {"science_data": science_results}
            )

            # Stage 3: Founder Voice Integration
            context.processing_stage = "founder_voice"
            context.intermediate_results.update(clinical_results)
            founder_results = await self._execute_agent_sequentially(
                "founder_voice", context,
                {"clinical_data": clinical_results}
            )

            # Stage 4: Context and RAG Enhancement
            context.processing_stage = "context_rag"
            context.intermediate_results.update(founder_results)
            rag_results = await self._execute_agent_sequentially(
                "context_rag", context,
                {"enhanced_data": context.intermediate_results}
            )

            # Stage 5: Marketing and SEO Content Generation
            context.processing_stage = "marketing_seo"
            context.intermediate_results.update(rag_results)
            marketing_results = await self._execute_agent_sequentially(
                "marketing_seo", context,
                {"final_context": context.intermediate_results}
            )

            # Stage 6: Final Validation and Quality Assurance
            context.processing_stage = "validation"
            final_results = await self._execute_agent_sequentially(
                "validation", context,
                {"complete_output": context.intermediate_results}
            )

            # Check quality and retry if needed
            quality_score = final_results.get("quality_scores", {}).get("overall_quality", 0)
            confidence_score = final_results.get("confidence_scoring", {}).get("confidence_score", 0)

            retry_count = 0
            while (quality_score < self.quality_thresholds["minimum_quality"] or
                   confidence_score < self.retry_config["confidence_threshold"]) and \
                  retry_count < self.retry_config["max_retries"]:

                retry_count += 1
                logger.info(f"🔄 Quality retry {retry_count}/{self.retry_config['max_retries']} for {context.document_id}")
                logger.info(f"   Quality: {quality_score:.2f}, Confidence: {confidence_score:.2f}")

                # Retry with enhanced context and quality improvement prompts
                improved_results = await self._retry_with_quality_improvement(
                    context, final_results, retry_count
                )

                if improved_results:
                    context.intermediate_results.update(improved_results)
                    final_results = await self._execute_agent_sequentially(
                        "validation", context,
                        {"complete_output": context.intermediate_results}
                    )

                    new_quality_score = final_results.get("quality_scores", {}).get("overall_quality", 0)
                    new_confidence_score = final_results.get("confidence_scoring", {}).get("confidence_score", 0)

                    # Check if quality improved enough
                    if (new_quality_score - quality_score < self.retry_config["quality_improvement_threshold"] and
                        new_confidence_score - confidence_score < self.retry_config["quality_improvement_threshold"]):
                        logger.warning(f"⚠️ Insufficient quality improvement after retry {retry_count}")
                        break

                    quality_score = new_quality_score
                    confidence_score = new_confidence_score
                    logger.info(f"✅ Quality improved: {quality_score:.2f}, Confidence: {confidence_score:.2f}")

                await asyncio.sleep(self.retry_config["retry_delay"])

            # Compile final results
            processing_time = datetime.now() - start_time
            validation_passed = quality_score >= self.quality_thresholds["minimum_quality"]

            final_output = {
                "document_id": context.document_id,
                "processing_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": processing_time.total_seconds(),
                "supervisor_status": "completed" if validation_passed else "completed_with_issues",
                "agent_outputs": context.intermediate_results,
                "quality_score": quality_score,
                "confidence_score": confidence_score,
                "validation_passed": validation_passed,
                "retry_count": retry_count,
                "final_validation": final_results.get("final_validation", {}),
                "quality_breakdown": final_results.get("quality_scores", {}),
                "confidence_analysis": final_results.get("confidence_scoring", {})
            }

            # Store processing history
            self.processing_history.append({
                "document_id": context.document_id,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "processing_time": processing_time.total_seconds(),
                "success": True,
                "quality_score": final_results.get("quality_score", 0)
            })

            logger.info(f"✅ Document {context.document_id} processed successfully in {processing_time.total_seconds():.2f}s")
            return final_output

        except Exception as e:
            processing_time = datetime.now() - start_time
            logger.error(f"❌ Document {context.document_id} processing failed after {processing_time.total_seconds():.2f}s: {e}")

            # Store failure in history
            self.processing_history.append({
                "document_id": context.document_id,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "processing_time": processing_time.total_seconds(),
                "success": False,
                "error": str(e)
            })

            return {
                "document_id": context.document_id,
                "processing_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": processing_time.total_seconds(),
                "supervisor_status": "failed",
                "error": str(e)
            }

    async def _execute_agent_sequentially(self, agent_name: str, context: ProcessingContext,
                                        additional_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single agent with proper context."""
        agent = self.agents.get(agent_name)
        if not agent:
            logger.error(f"Agent {agent_name} not found")
            return {}

        try:
            # Prepare agent context
            agent_context = {
                "document_id": context.document_id,
                "document_text": context.document_text,
                "client_insights": context.client_insights,
                "founder_insights": context.founder_insights,
                "st_louis_context": context.st_louis_context,
                "processing_stage": context.processing_stage,
                "intermediate_results": context.intermediate_results.copy()
            }
            agent_context.update(additional_context)

            logger.info(f"🔄 Executing agent: {agent_name}")
            logger.info(f"📋 Agent context keys: {list(agent_context.keys())}")

            result = await agent.execute(agent_context)

            if result:
                logger.info(f"✅ Agent {agent_name} completed successfully")
                logger.info(f"📊 Agent {agent_name} result keys: {list(result.keys())}")
                return result
            else:
                logger.warning(f"⚠️ Agent {agent_name} returned empty results")
                return {}

        except Exception as e:
            logger.error(f"❌ Agent {agent_name} execution failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {}

    async def _retry_with_quality_improvement(self, context: ProcessingContext,
                                            validation_results: Dict[str, Any],
                                            retry_count: int) -> Dict[str, Any]:
        """Retry processing with quality improvement prompts."""
        try:
            logger.info(f"🔄 Attempting quality improvement retry {retry_count}")

            # Identify which agents need improvement based on validation results
            quality_scores = validation_results.get("quality_scores", {})
            issues = validation_results.get("final_validation", {}).get("recommendations", [])

            # Determine which content areas need improvement
            improvement_areas = []
            if quality_scores.get("clinical_accuracy", 1) < 0.7:
                improvement_areas.append("clinical_content")
            if quality_scores.get("founder_voice_authenticity", 1) < 0.7:
                improvement_areas.append("founder_voice")
            if quality_scores.get("marketing_effectiveness", 1) < 0.7:
                improvement_areas.append("marketing_seo")
            if quality_scores.get("completeness", 1) < 0.8:
                improvement_areas.append("all_sections")

            if not improvement_areas:
                improvement_areas = ["clinical_content", "marketing_content"]  # Default retry areas

            improved_results = {}

            # Retry specific agents with enhanced prompts
            for area in improvement_areas:
                if area == "clinical_content":
                    enhanced_results = await self._retry_clinical_content(context, issues, retry_count)
                    if enhanced_results:
                        improved_results.update(enhanced_results)

                elif area == "founder_voice":
                    enhanced_results = await self._retry_founder_voice(context, issues, retry_count)
                    if enhanced_results:
                        improved_results.update(enhanced_results)

                elif area == "marketing_seo":
                    enhanced_results = await self._retry_marketing_content(context, issues, retry_count)
                    if enhanced_results:
                        improved_results.update(enhanced_results)

                elif area == "all_sections":
                    # Retry all agents with quality improvement
                    enhanced_results = await self._retry_all_sections(context, issues, retry_count)
                    if enhanced_results:
                        improved_results.update(enhanced_results)

            return improved_results

        except Exception as e:
            logger.error(f"❌ Quality improvement retry failed: {e}")
            return {}

    async def _retry_clinical_content(self, context: ProcessingContext, issues: List[str],
                                    retry_count: int) -> Dict[str, Any]:
        """Retry clinical content with enhanced prompts."""
        try:
            # Enhanced prompt for clinical improvement
            improvement_prompts = [
                "Focus more on neuroscience evidence and brain mechanisms",
                "Include specific clinical interventions and protocols",
                "Add more client-centered language and outcomes",
                "Ensure all content is evidence-based and clinically sound"
            ]

            if retry_count <= len(improvement_prompts):
                enhancement = improvement_prompts[retry_count - 1]
            else:
                enhancement = "Improve clinical accuracy and evidence-based content"

            # Re-execute clinical synthesis with enhanced context
            clinical_agent = self.agents.get("clinical_synthesis")
            if clinical_agent:
                enhanced_context = asdict(context)
                enhanced_context.update({
                    "quality_issues": issues,
                    "enhancement_focus": enhancement,
                    "retry_attempt": retry_count
                })

                result = await clinical_agent.execute(enhanced_context)
                return {"clinical_content": result.get("clinical_content", {}),
                       "educational_content": result.get("educational_content", {})}

        except Exception as e:
            logger.error(f"Clinical content retry failed: {e}")

        return {}

    async def _retry_founder_voice(self, context: ProcessingContext, issues: List[str],
                                 retry_count: int) -> Dict[str, Any]:
        """Retry founder voice content with enhanced prompts."""
        try:
            # Enhanced prompts for voice improvement
            voice_improvements = [
                "Make language more direct and authentic like Liz Wooten",
                "Increase rebellious tone against traditional therapy",
                "Add more hopeful and empowering messaging",
                "Include more St. Louis local relevance and context"
            ]

            if retry_count <= len(voice_improvements):
                enhancement = voice_improvements[retry_count - 1]
            else:
                enhancement = "Enhance founder voice authenticity and St. Louis relevance"

            # Re-execute founder voice agent with enhanced context
            founder_agent = self.agents.get("founder_voice")
            if founder_agent:
                enhanced_context = asdict(context)
                enhanced_context.update({
                    "quality_issues": issues,
                    "enhancement_focus": enhancement,
                    "retry_attempt": retry_count
                })

                result = await founder_agent.execute(enhanced_context)
                return result

        except Exception as e:
            logger.error(f"Founder voice retry failed: {e}")

        return {}

    async def _retry_marketing_content(self, context: ProcessingContext, issues: List[str],
                                     retry_count: int) -> Dict[str, Any]:
        """Retry marketing content with enhanced prompts."""
        try:
            # Enhanced prompts for marketing improvement
            marketing_improvements = [
                "Add stronger calls-to-action and conversion elements",
                "Include more local SEO terms for St. Louis",
                "Improve value propositions and benefits",
                "Add more specific client pain point solutions"
            ]

            if retry_count <= len(marketing_improvements):
                enhancement = marketing_improvements[retry_count - 1]
            else:
                enhancement = "Enhance marketing effectiveness and conversion optimization"

            # Re-execute marketing agent with enhanced context
            marketing_agent = self.agents.get("marketing_seo")
            if marketing_agent:
                enhanced_context = asdict(context)
                enhanced_context.update({
                    "quality_issues": issues,
                    "enhancement_focus": enhancement,
                    "retry_attempt": retry_count
                })

                result = await marketing_agent.execute(enhanced_context)
                return result

        except Exception as e:
            logger.error(f"Marketing content retry failed: {e}")

        return {}

    async def _retry_all_sections(self, context: ProcessingContext, issues: List[str],
                                retry_count: int) -> Dict[str, Any]:
        """Retry all sections with comprehensive quality improvement."""
        try:
            # Comprehensive retry with all agents
            improved_results = {}

            # Retry each agent with quality enhancement
            for agent_name in ["science_extraction", "clinical_synthesis", "founder_voice", "marketing_seo"]:
                agent = self.agents.get(agent_name)
                if agent:
                    enhanced_context = asdict(context)
                    enhanced_context.update({
                        "quality_issues": issues,
                        "enhancement_focus": f"Improve quality and address: {', '.join(issues[:3])}",
                        "retry_attempt": retry_count,
                        "comprehensive_retry": True
                    })

                    result = await agent.execute(enhanced_context)
                    if result:
                        improved_results.update(result)

            return improved_results

        except Exception as e:
            logger.error(f"Comprehensive retry failed: {e}")
            return {}

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method for the supervisor."""
        # Convert dict context to ProcessingContext
        processing_context = ProcessingContext(**context)
        return await self.process_document(processing_context)

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate supervisor output."""
        required_fields = ["document_id", "processing_timestamp", "supervisor_status"]
        return all(field in output for field in required_fields)

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "supervisor": self.get_status(),
            "agents": {name: agent.get_status() for name, agent in self.agents.items()},
            "processing_history": self.processing_history[-10:],  # Last 10 entries
            "total_processed": len([h for h in self.processing_history if h.get("success", False)]),
            "total_failed": len([h for h in self.processing_history if not h.get("success", False)]),
            "average_quality_score": sum(h.get("quality_score", 0) for h in self.processing_history) / max(len(self.processing_history), 1)
        }

    async def cleanup(self):
        """Clean up supervisor and all agents."""
        logger.info("🧹 Cleaning up supervisor and all agents...")

        # Clean up all agents
        for agent_name, agent in self.agents.items():
            try:
                await agent.cleanup()
                logger.info(f"✅ Agent {agent_name} cleaned up")
            except Exception as e:
                logger.error(f"❌ Error cleaning up agent {agent_name}: {e}")

        self.agents.clear()
        self.processing_history.clear()
        self.is_initialized = False
        logger.info("✅ Supervisor cleanup completed")