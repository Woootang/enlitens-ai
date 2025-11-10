"""
Context Curator Agent
Coordinates the 3 pre-processing agents to create optimized context for the main agent.
"""

import logging
from typing import Any, Dict, List, Optional

from src.agents.profile_matcher_agent import ProfileMatcherAgent
from src.agents.health_report_synthesizer_agent import HealthReportSynthesizerAgent
from src.agents.voice_guide_generator_agent import VoiceGuideGeneratorAgent

logger = logging.getLogger(__name__)


class ContextCuratorAgent:
    """
    Master coordinator for intelligent context curation.
    
    Orchestrates 3 pre-processing agents to create optimized, high-signal context
    for the main synthesis agent.
    """
    
    def __init__(
        self,
        personas_dir: str = "enlitens_client_profiles/profiles",
        transcripts_path: str = "enlitens_knowledge_base/transcripts.txt",
        framework_path: str = "enlitens_knowledge_base/enlitens_interview_framework.txt"
    ):
        self.profile_matcher = ProfileMatcherAgent(personas_dir=personas_dir)
        self.health_synthesizer = HealthReportSynthesizerAgent()
        self.voice_generator = VoiceGuideGeneratorAgent(
            transcripts_path=transcripts_path,
            framework_path=framework_path
        )
        
        # Cache voice guide (generated once, reused for all documents)
        self.voice_guide_cache: Optional[str] = None
        
    async def curate_context(
        self,
        paper_text: str,
        entities: Dict[str, List[str]],
        health_report_text: str,
        llm_client: Any
    ) -> Dict[str, Any]:
        """
        Curate optimized context for a research paper.
        
        Args:
            paper_text: Full text of the research paper
            entities: NER extracted entities
            health_report_text: Full St. Louis health report
            llm_client: LLM client for agent operations
            
        Returns:
            Dict containing curated context:
            - selected_personas: List of 10 relevant persona objects
            - personas_text: Formatted text of selected personas
            - health_brief: Synthesized health context
            - voice_guide: Liz's voice style guide
            - token_estimate: Estimated total tokens
        """
        logger.info("="*80)
        logger.info("🎯 CONTEXT CURATION: Intelligent Pre-Processing")
        logger.info("="*80)
        
        # Step 1: Select top 10 relevant personas
        logger.info("📊 Agent 1: Profile Matcher - Selecting relevant personas...")
        selected_personas = await self.profile_matcher.select_top_personas(
            paper_text=paper_text,
            entities=entities,
            llm_client=llm_client,
            top_k=10
        )
        
        personas_text = self.profile_matcher.get_selected_personas_text(selected_personas)
        logger.info(f"✅ Selected {len(selected_personas)} personas (~{len(personas_text)//4} tokens)")
        
        # Step 2: Synthesize health report
        logger.info("🏥 Agent 2: Health Report Synthesizer - Creating targeted brief...")
        health_brief = await self.health_synthesizer.synthesize_health_context(
            health_report_text=health_report_text,
            selected_personas=selected_personas,
            llm_client=llm_client
        )
        logger.info(f"✅ Health brief synthesized (~{len(health_brief)//4} tokens)")
        
        # Step 3: Generate voice guide (cached after first generation)
        if self.voice_guide_cache is None:
            logger.info("🎙️ Agent 3: Voice Guide Generator - Creating Liz's style guide...")
            self.voice_guide_cache = await self.voice_generator.generate_voice_guide(llm_client)
            logger.info(f"✅ Voice guide generated (~{len(self.voice_guide_cache)//4} tokens)")
        else:
            logger.info("✅ Using cached voice guide")
        
        voice_guide = self.voice_guide_cache
        
        # Calculate token estimates (rough: 1 token ≈ 4 characters)
        token_estimate = {
            'personas': len(personas_text) // 4,
            'health_brief': len(health_brief) // 4,
            'voice_guide': len(voice_guide) // 4,
            'total_curated': (len(personas_text) + len(health_brief) + len(voice_guide)) // 4
        }
        
        logger.info("="*80)
        logger.info(f"📊 CURATION COMPLETE - Token Breakdown:")
        logger.info(f"   Personas: ~{token_estimate['personas']:,} tokens")
        logger.info(f"   Health Brief: ~{token_estimate['health_brief']:,} tokens")
        logger.info(f"   Voice Guide: ~{token_estimate['voice_guide']:,} tokens")
        logger.info(f"   Total Curated Context: ~{token_estimate['total_curated']:,} tokens")
        logger.info("="*80)
        
        return {
            'selected_personas': selected_personas,
            'personas_text': personas_text,
            'health_brief': health_brief,
            'voice_guide': voice_guide,
            'token_estimate': token_estimate
        }
    
    def format_curated_context_for_prompt(self, curated_context: Dict[str, Any]) -> str:
        """
        Format curated context into a single string for the main agent's prompt.
        
        Returns a structured, ready-to-use context block.
        """
        return f"""
# CURATED CONTEXT FOR THIS RESEARCH PAPER

## RELEVANT CLIENT PERSONAS
{curated_context['personas_text']}

## LOCAL HEALTH CONTEXT (St. Louis)
{curated_context['health_brief']}

## LIZ'S VOICE GUIDE
{curated_context['voice_guide']}

---
This context has been intelligently curated to maximize relevance and quality.
Use ALL of this information to inform your output.
"""

