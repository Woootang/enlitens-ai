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
            science_data = context.get("science_data", {})
            research_content = science_data.get("research_content", {})
            
            prompt = f"""
You are a clinical psychologist translating neuroscience research into therapy applications.

RESEARCH FINDINGS:
{research_content.get('key_findings', [])}

NEUROSCIENCE MECHANISMS:
{research_content.get('neuroscience_mechanisms', [])}

Synthesize clinical content:
1. Treatment approaches (3-5 therapy techniques)
2. Assessment methods (how to evaluate clients)
3. Intervention strategies (specific interventions)
4. Client education points (what to teach clients)
5. Progress indicators (signs of improvement)

Return as JSON with these exact fields:
- treatment_approaches: [list of strings]
- assessment_methods: [list of strings]
- intervention_strategies: [list of strings]
- client_education: [list of strings]
- progress_indicators: [list of strings]
"""

            result = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=ClinicalContent,
                temperature=0.4,
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
        
        has_content = any([
            clinical_content.get("treatment_approaches"),
            clinical_content.get("intervention_strategies")
        ])
        
        return has_content

    async def cleanup(self):
        """Clean up resources."""
        logger.info(f"Cleaning up {self.name} agent")
