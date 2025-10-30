"""
Educational Content Agent - Extracts and generates educational materials.
"""

import logging
from typing import Dict, Any

from .base_agent import BaseAgent
from src.models.enlitens_schemas import EducationalContent
from src.synthesis.few_shot_library import FEW_SHOT_LIBRARY
from src.synthesis.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class EducationalContentAgent(BaseAgent):
    """Agent specialized in generating educational content for clients."""

    def __init__(self):
        super().__init__(
            name="EducationalContent",
            role="Client Education Material Generation",
            model="/home/antons-gs/enlitens-ai/models/mistral-7b-instruct"
        )
        self.ollama_client = None

    async def initialize(self) -> bool:
        """Initialize the educational content agent."""
        try:
            self.ollama_client = OllamaClient(default_model=self.model)
            if not await self.ollama_client.check_connection():
                raise RuntimeError(
                    f"vLLM server is not reachable at {self.ollama_client.base_url}. Please run stable_run.sh or start the vLLM server."
                )
            self.is_initialized = True
            logger.info(f"✅ {self.name} agent initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            return False

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate educational content from research."""
        try:
            document_text = context.get("document_text", "")[:8000]
            research_content = context.get("science_data", {}).get("research_content", {})
            clinical_content = context.get("clinical_content", {})

            few_shot_block = FEW_SHOT_LIBRARY.render_for_prompt(
                task="educational_content",
                query=document_text,
                k=2,
            )

            exemplars = (
                "FEW-SHOT EXEMPLARS (match tone, show client-level clarity):\n"
                f"{few_shot_block}\n\n" if few_shot_block else ""
            )

            prompt = f"""
You are an educational content specialist creating client-friendly educational materials.
Your goal is to translate complex neuroscience research into accessible educational content.

STRICT RULES:
✓ Base all explanations on the research findings provided
✓ Use analogies and examples to make concepts accessible
✓ Clearly mark hypothetical examples as "[HYPOTHETICAL EXAMPLE]"
✓ When citing research, use exact findings from the source
✗ DO NOT add neuroscience facts from your training data not in the source
✗ DO NOT generate practice statistics or client testimonials
✗ DO NOT fabricate research findings or statistics

{exemplars}
DOCUMENT TEXT:
{document_text}

RESEARCH FINDINGS:
{research_content.get('findings', [])}

CLINICAL APPLICATIONS:
{clinical_content.get('interventions', [])}

Create comprehensive educational content for ALL sections below (5-10 items per section):

1. EXPLANATIONS: Clear, accessible explanations of neuroscience concepts from the research. Break down complex ideas into understandable language. Explain HOW things work in the brain.

2. EXAMPLES: Concrete, relatable examples that illustrate neuroscience concepts. Use everyday situations clients would recognize. Show concepts in action.

3. ANALOGIES: Creative analogies and metaphors that make neuroscience accessible. Compare brain processes to familiar things (e.g., "Your amygdala is like a smoke alarm").

4. DEFINITIONS: Simple definitions of technical terms, brain regions, processes, and neuroscience jargon. Make terminology accessible without dumbing down.

5. PROCESSES: Step-by-step explanations of how brain processes work. Explain sequences like "How anxiety develops in the brain" or "How memory consolidation happens."

6. COMPARISONS: Compare and contrast different concepts, brain regions, or processes. Show how things are similar or different. Help clients understand relationships.

7. VISUAL AIDS: Descriptions of diagrams, charts, or visual aids that would help explain concepts. Describe what should be illustrated and why it would help learning.

8. LEARNING OBJECTIVES: Clear learning objectives for each topic. What should clients understand after learning this? What should they be able to do with this knowledge?

EXTRACTION GUIDELINES:
- Make everything client-accessible (8th grade reading level)
- Use concrete examples from daily life
- Explain "why this matters" for each concept
- Avoid jargon or explain it when necessary
- Each item should be substantial (2-4 sentences minimum)
- Focus on practical understanding, not academic detail
- Each section needs 5-10 items minimum

Return as JSON with these EXACT field names:
{{"explanations": [list], "examples": [list], "analogies": [list], "definitions": [list], "processes": [list], "comparisons": [list], "visual_aids": [list], "learning_objectives": [list]}}
"""

            cache_kwargs = self._cache_kwargs(context)
            result = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=EducationalContent,
                temperature=0.35,
                max_retries=3,
                enforce_grammar=True,
                **cache_kwargs,
            )

            if result:
                return {
                    "educational_content": result.model_dump(),
                    "generation_quality": "high"
                }

            return {"educational_content": EducationalContent().model_dump()}

        except Exception as e:
            logger.error(f"Educational content generation failed: {e}")
            return {"educational_content": EducationalContent().model_dump()}

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate the generated educational content."""
        educational_content = output.get("educational_content", {})

        has_content = any([
            educational_content.get("explanations"),
            educational_content.get("examples"),
            educational_content.get("analogies")
        ])

        return has_content

    async def cleanup(self):
        """Clean up resources."""
        logger.info(f"Cleaning up {self.name} agent")
