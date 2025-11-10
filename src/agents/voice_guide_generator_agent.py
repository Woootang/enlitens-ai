"""
Voice Guide Generator Agent
Distills Liz's transcripts and the Enlitens Interview framework into a clear style guide.
"""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class VoiceGuideGeneratorAgent:
    """Generates a comprehensive style guide from Liz's voice and the Enlitens framework."""
    
    def __init__(
        self,
        transcripts_path: str = "enlitens_knowledge_base/liz_transcripts.txt",
        framework_path: str = "enlitens_knowledge_base/enlitens_interview_framework.txt"
    ):
        self.transcripts_path = Path(transcripts_path)
        self.framework_path = Path(framework_path)
        self.voice_guide_cache: Optional[str] = None
        
    def load_transcripts(self) -> str:
        """Load Liz's transcripts."""
        if not self.transcripts_path.exists():
            logger.warning(f"Transcripts not found: {self.transcripts_path}")
            return ""
        
        try:
            with open(self.transcripts_path, 'r') as f:
                content = f.read()
            logger.info(f"✅ Loaded Liz transcripts ({len(content)} chars)")
            return content
        except Exception as e:
            logger.error(f"Failed to load transcripts: {e}")
            return ""
    
    def load_framework(self) -> str:
        """Load the Enlitens Interview framework."""
        if not self.framework_path.exists():
            logger.warning(f"Framework not found: {self.framework_path}")
            return ""
        
        try:
            with open(self.framework_path, 'r') as f:
                content = f.read()
            logger.info(f"✅ Loaded Enlitens framework ({len(content)} chars)")
            return content
        except Exception as e:
            logger.error(f"Failed to load framework: {e}")
            return ""
    
    async def generate_voice_guide(self, llm_client: Any) -> str:
        """
        Generate a comprehensive "How to Write Like Liz" style guide.
        
        Args:
            llm_client: LLM client for generation
            
        Returns:
            Complete voice guide (1500-2000 words)
        """
        # Check cache
        if self.voice_guide_cache:
            logger.info("Using cached voice guide")
            return self.voice_guide_cache
        
        transcripts = self.load_transcripts()
        framework = self.load_framework()
        
        if not transcripts and not framework:
            logger.error("No source material for voice guide generation")
            return self._get_fallback_guide()
        
        # Create generation prompt
        prompt = self._create_generation_prompt(transcripts, framework)
        
        try:
            response = await llm_client.generate_text(
                prompt=prompt,
                temperature=0.5,  # Moderate creativity
                num_predict=3000
            )
            
            # Cache the guide
            self.voice_guide_cache = response
            logger.info(f"✅ Generated voice guide ({len(response)} chars)")
            return response
            
        except Exception as e:
            logger.error(f"Voice guide generation failed: {e}")
            return self._get_fallback_guide()
    
    def _create_generation_prompt(self, transcripts: str, framework: str) -> str:
        """Create the prompt for voice guide generation."""
        
        # Truncate if needed (keep first 10k chars of each)
        transcripts_text = transcripts[:10000] + "..." if len(transcripts) > 10000 else transcripts
        framework_text = framework[:15000] + "..." if len(framework) > 15000 else framework
        
        prompt = f"""You are a master linguist and brand voice analyst.

Your mission: Analyze Liz's voice across her transcripts and the Enlitens Interview framework to create a comprehensive "How to Write Like Liz" style guide.

LIZ'S TRANSCRIPTS:
{transcripts_text}

ENLITENS INTERVIEW FRAMEWORK:
{framework_text}

YOUR TASK:
Create a comprehensive style guide (1500-2000 words) that captures EVERYTHING about how Liz communicates.

REQUIRED SECTIONS:

## 1. CORE PHILOSOPHY (The "Why")
- The rebellion against pathology
- Science-backed validation
- Radical acceptance + strategic action
- The "fuck the system" energy
- Key beliefs that drive every word

## 2. SIGNATURE LANGUAGE PATTERNS
- Recurring phrases (e.g., "Holy shit," "Let's call it what it is," "This is the truth")
- Metaphors & analogies (e.g., race car brain, energy vampires, mission briefings, spy debriefs)
- Tone shifts (raw emotion → hard science → empowering action)
- Profanity usage (strategic, emphatic, never gratuitous - when and why)
- Direct address ("You know that feeling...")

## 3. STRUCTURAL PATTERNS
- How she opens (bold statement, provocative question, or gut-punch truth)
- How she builds (personal story → scientific data → insight → action)
- How she transitions (rhetorical questions, "So here's the truth...")
- How she closes (always action-oriented, empowering, with momentum)

## 4. REFRAMING TECHNIQUES
- Pathology → Neurodiversity
- Deficit → Feature
- Disorder → Brilliant adaptation
- Resistance → Survival skill
- Burnout → Predictable resource depletion
[Provide 10+ specific before/after examples]

## 5. FORBIDDEN PATTERNS (What Liz NEVER Does)
- Clinical jargon without immediate reframe
- Pathologizing language without rebellion context
- Vague platitudes or empty encouragement
- Passive voice or hedging language
- "Disorder" without validating the person first
- Blame or shame (toward client OR clinician)

## 6. THE LIZ VOICE CHECKLIST
A quick reference: "Does this sound like Liz?"
- [ ] Bold, direct, no bullshit?
- [ ] Science-backed, not just feelings?
- [ ] Validates the person, challenges the system?
- [ ] Uses signature metaphors?
- [ ] Ends with action, not just insight?
- [ ] Strategic profanity for emphasis?
- [ ] Reframes pathology as power?

## 7. EXAMPLE TRANSFORMATIONS
Show 5-10 before/after examples:
- Generic clinical language → Liz voice
- Pathologizing statement → Rebellion reframe
- Vague advice → Specific, actionable Liz guidance

This guide will be used to ensure EVERY piece of content sounds authentically like Liz.
Make it comprehensive, actionable, and fucking useful.
"""
        return prompt
    
    def _get_fallback_guide(self) -> str:
        """Fallback voice guide if generation fails."""
        return """# LIZ'S VOICE GUIDE (Fallback)

## Core Philosophy
- Reject pathology, embrace neurodiversity
- Science-backed validation, not empty platitudes
- Radical acceptance + strategic action
- Challenge the system, not the person

## Key Patterns
- **Bold openings**: "Let's call it what it is..."
- **Metaphors**: Race car brain, energy vampires, mission briefings
- **Strategic profanity**: For emphasis, never gratuitous
- **Reframes**: Disorder → feature, resistance → survival skill

## Structure
1. Hook (bold truth or question)
2. Story or example (relatable, human)
3. Science (hard data, neurobiology)
4. Insight (the "aha!" moment)
5. Action (concrete next steps)

## Forbidden
- Clinical jargon without reframe
- Pathologizing without validation
- Vague encouragement
- Passive voice
- Blame or shame

Use this as a foundation. Liz's voice is bold, scientific, empowering, and unapologetically rebellious.
"""

