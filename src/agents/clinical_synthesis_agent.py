"""
Clinical Synthesis Agent - Synthesizes clinical applications from research.
"""

import logging
from typing import Dict, Any
from .base_agent import BaseAgent
from src.synthesis.ollama_client import OllamaClient
from src.models.enlitens_schemas import ClinicalContent

logger = logging.getLogger(__name__)

class ClinicalSynthesisAgent(BaseAgent):
    """Agent specialized in synthesizing clinical applications."""

    def __init__(self):
        super().__init__(
            name="ClinicalSynthesis",
            role="Clinical Application Synthesis",
            model="qwen3:32b"
        )
        self.ollama_client = None

    async def initialize(self) -> bool:
        """Initialize the clinical synthesis agent."""
        try:
            self.ollama_client = OllamaClient(default_model=self.model)
            self.is_initialized = True
            logger.info(f"✅ {self.name} agent initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            return False

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize clinical applications from research."""
        try:
            # Get context data
            science_data = context.get("science_data", {})
            research_content = science_data.get("research_content", {})
            document_text = context.get("document_text", "")[:5000]  # First 5000 chars

            prompt = f"""
You are a clinical psychologist extracting therapeutic applications from neuroscience research.

STRICT RULES:
✓ Extract ONLY clinical applications that can be reasonably inferred from the research
✓ Base all interventions on stated research findings
✓ Clearly distinguish between stated protocols and potential applications
✗ DO NOT fabricate therapeutic protocols not supported by the research
✗ DO NOT add clinical practices from your training data
✗ DO NOT generate practice statistics or client outcome claims

DOCUMENT EXCERPT:
{document_text}

RESEARCH FINDINGS:
{research_content.get('findings', [])}

METHODOLOGIES:
{research_content.get('methodologies', [])}

CLINICAL IMPLICATIONS:
{research_content.get('implications', [])}

Extract and synthesize clinical content from this research. Base all clinical applications on the research findings above. For potential applications not explicitly stated, use prefix "Potential application:"

Generate content for ALL fields below:

1. INTERVENTIONS: List 3-8 specific therapeutic interventions or techniques that could be applied based on this research. Include evidence-based practices, neuroscience-informed approaches, and practical therapy techniques.

2. ASSESSMENTS: List 3-8 assessment methods or evaluation tools that would be relevant. Include diagnostic tools, measurement instruments, observational methods, and screening approaches.

3. OUTCOMES: List 3-8 expected clinical outcomes or treatment goals. What changes should clients experience? Include behavioral, cognitive, emotional, and neurobiological outcomes.

4. PROTOCOLS: List 3-8 treatment protocols or structured approaches. Include session structures, treatment phases, or therapeutic frameworks that could be used.

5. GUIDELINES: List 3-8 clinical guidelines or best practices. Include recommendations for implementation, ethical considerations, and evidence-based standards.

6. CONTRAINDICATIONS: List 3-8 contraindications or situations where caution is needed. When should these approaches NOT be used? Who should avoid them?

7. SIDE EFFECTS: List 3-8 potential side effects or risks to monitor. What could go wrong? What should therapists watch for?

8. MONITORING: List 3-8 monitoring approaches or progress indicators. How should therapists track progress? What metrics matter?

IMPORTANT:
- Extract liberally - if research mentions anxiety, include anxiety assessment tools and interventions
- Infer practical applications from theoretical findings
- Each list should have 3-8 items minimum
- Use plain language, no numbering or bullet points in strings
- If a field seems not applicable, still provide related content (e.g., general best practices)

Return as JSON with these EXACT field names:
{{"interventions": [list], "assessments": [list], "outcomes": [list], "protocols": [list], "guidelines": [list], "contraindications": [list], "side_effects": [list], "monitoring": [list]}}
"""

            result = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=ClinicalContent,
                temperature=0.3,  # LOWERED from 0.7: Research shows 0.3 optimal for factual clinical content
                max_retries=3
            )

            if result:
                return {
                    "clinical_content": result.model_dump(),
                    "synthesis_quality": "high"
                }
            else:
                return {"clinical_content": ClinicalContent().model_dump()}

        except Exception as e:
            logger.error(f"Clinical synthesis failed: {e}")
            return {"clinical_content": ClinicalContent().model_dump()}

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate the synthesized clinical content."""
        clinical_content = output.get("clinical_content", {})

        # Check for correct field names matching ClinicalContent schema
        has_content = any([
            clinical_content.get("interventions"),
            clinical_content.get("assessments"),
            clinical_content.get("outcomes")
        ])

        return has_content

    async def cleanup(self):
        """Clean up resources."""
        logger.info(f"Cleaning up {self.name} agent")
