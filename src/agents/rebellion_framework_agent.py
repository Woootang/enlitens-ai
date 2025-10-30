"""
Rebellion Framework Agent - Extracts content for Enlitens' proprietary Rebellion Framework.
"""

import logging
from typing import Dict, Any
from .base_agent import BaseAgent
from src.synthesis.ollama_client import OllamaClient
from src.models.enlitens_schemas import RebellionFramework

logger = logging.getLogger(__name__)

class RebellionFrameworkAgent(BaseAgent):
    """Agent specialized in applying the Rebellion Framework to research."""

    def __init__(self):
        super().__init__(
            name="RebellionFramework",
            role="Rebellion Framework Application",
            model="qwen2.5-32b-instruct-q4_k_m"
        )
        self.ollama_client = None

    async def initialize(self) -> bool:
        """Initialize the rebellion framework agent."""
        try:
            self.ollama_client = OllamaClient(default_model=self.model)
            self.is_initialized = True
            logger.info(f"✅ {self.name} agent initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            return False

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Rebellion Framework to research content."""
        try:
            document_text = context.get("document_text", "")[:8000]
            research_content = context.get("science_data", {}).get("research_content", {})
            clinical_content = context.get("clinical_content", {})

            prompt = f"""
You are applying Enlitens' proprietary Rebellion Framework to neuroscience research.
The Rebellion Framework challenges traditional deficit-based narratives and reframes neurodivergence as adaptation and strength.

STRICT RULES:
✓ Base all reframings on actual research findings provided
✓ Use neuroscience evidence to challenge deficit-based narratives
✓ Distinguish between research-supported claims and creative reframing
✓ Mark speculative strengths as "Potential strength:" when not explicitly stated in research
✗ DO NOT fabricate neuroscience findings to support reframing
✗ DO NOT generate practice statistics or client testimonials
✗ DO NOT add claims not supported by the research context

DOCUMENT TEXT:
{document_text}

RESEARCH FINDINGS:
{research_content.get('findings', [])}

CLINICAL APPLICATIONS:
{clinical_content.get('interventions', [])}

Apply the Rebellion Framework to extract content for ALL sections below (3-10 items per section):

1. NARRATIVE DECONSTRUCTION: Identify traditional mental health narratives that this research challenges or complicates. What deficit-based assumptions can we challenge? How does this research suggest alternative framings? Examples: "ADHD is a disorder" → "ADHD is a different cognitive style with unique strengths"

2. SENSORY PROFILING: Extract sensory processing insights from the research. How do different brains process sensory information? What sensory patterns emerge? Include interoception, exteroception, proprioception findings. Relate to how clients might experience their sensory world differently.

3. EXECUTIVE FUNCTION: Identify executive function insights. How do different brains handle planning, organization, emotional regulation, impulse control, working memory, task switching? What adaptations or compensations emerge? Frame as differences, not deficits.

4. SOCIAL PROCESSING: Extract social cognition and connection insights. How do different brains process social information, relationships, communication, emotional cues? What are the neuroscience mechanisms? How can we support authentic social connection?

5. STRENGTHS SYNTHESIS: Synthesize neurodivergent strengths from the research. What advantages, unique capabilities, or positive traits emerge? How can "symptoms" be reframed as adaptations or strengths in different contexts? Be specific and genuine.

6. REBELLION THEMES: Identify themes that support the "rebellion" against traditional narratives. What assumptions does this research challenge? How does it support neurodiversity-affirming perspectives? What radical reframings are possible?

7. AHA MOMENTS: Create "aha moment" insights for clients. Powerful realizations that shift perspective from "I'm broken" to "My brain works differently." Validating insights that help clients understand themselves. Format: "You're not [negative interpretation], your brain is [neuroscience truth]"

REBELLION FRAMEWORK PRINCIPLES:
- Challenge deficit-based narratives
- Reframe differences as adaptations, not disorders
- Validate client experiences through neuroscience
- Emphasize neuroplasticity and possibility
- Honor neurodiversity and different ways of being
- Support authentic self-understanding
- Each item should shift perspective from shame to science

IMPORTANT:
- Each section needs 3-10 items
- Be specific to the research content
- Use empowering, validating language
- Ground in actual neuroscience findings
- Make it personally relevant to clients
- Challenge traditional mental health narratives

Return as JSON with these EXACT field names:
{{"narrative_deconstruction": [list], "sensory_profiling": [list], "executive_function": [list], "social_processing": [list], "strengths_synthesis": [list], "rebellion_themes": [list], "aha_moments": [list]}}
"""

            cache_kwargs = self._cache_kwargs(context)
            result = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=RebellionFramework,
                temperature=0.6,  # LOWERED from 0.75: Research shows 0.6 optimal for creative but grounded content
                max_retries=3,
                **cache_kwargs,
            )

            if result:
                return {
                    "rebellion_framework": result.model_dump(),
                    "framework_quality": "high"
                }
            else:
                return {"rebellion_framework": RebellionFramework().model_dump()}

        except Exception as e:
            logger.error(f"Rebellion framework application failed: {e}")
            return {"rebellion_framework": RebellionFramework().model_dump()}

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate the rebellion framework content."""
        rebellion_content = output.get("rebellion_framework", {})

        has_content = any([
            rebellion_content.get("narrative_deconstruction"),
            rebellion_content.get("aha_moments"),
            rebellion_content.get("strengths_synthesis")
        ])

        return has_content

    async def cleanup(self):
        """Clean up resources."""
        logger.info(f"Cleaning up {self.name} agent")
