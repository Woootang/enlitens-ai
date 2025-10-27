"""
Science Extraction Agent - Extracts scientific findings and neuroscience content.
"""

import logging
from typing import Dict, Any
from .base_agent import BaseAgent
from src.synthesis.ollama_client import OllamaClient
from src.models.enlitens_schemas import ResearchContent

logger = logging.getLogger(__name__)

class ScienceExtractionAgent(BaseAgent):
    """Agent specialized in extracting scientific and research content."""

    def __init__(self):
        super().__init__(
            name="ScienceExtraction",
            role="Scientific Content Extraction",
            model="qwen3:32b"
        )
        self.ollama_client = None

    async def initialize(self) -> bool:
        """Initialize the science extraction agent."""
        try:
            self.ollama_client = OllamaClient(default_model=self.model)
            self.is_initialized = True
            logger.info(f"✅ {self.name} agent initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            return False

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract scientific content from the document."""
        try:
            document_text = context.get("document_text", "")
            
            prompt = f"""
You are a neuroscience research analyst extracting key scientific findings.

DOCUMENT TEXT (first 5000 chars):
{document_text[:5000]}

Extract the following research content:
1. Key findings (3-5 major discoveries or conclusions)
2. Methodologies (research methods used)
3. Neuroscience mechanisms (brain processes explained)
4. Clinical implications (how this applies to therapy)
5. Evidence strength (quality of evidence)

Return as JSON with these exact fields:
- key_findings: [list of strings]
- methodologies: [list of strings]
- neuroscience_mechanisms: [list of strings]
- clinical_implications: [list of strings]
- evidence_strength: [list of strings]
"""

            result = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=ResearchContent,
                temperature=0.3,
                max_retries=3
            )

            if result:
                return {
                    "research_content": result.model_dump(),
                    "extraction_quality": "high"
                }
            else:
                return {"research_content": ResearchContent().model_dump()}

        except Exception as e:
            logger.error(f"Science extraction failed: {e}")
            return {"research_content": ResearchContent().model_dump()}

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate the extracted scientific content."""
        research_content = output.get("research_content", {})
        
        # Check if we have at least some content
        has_content = any([
            research_content.get("key_findings"),
            research_content.get("methodologies"),
            research_content.get("neuroscience_mechanisms")
        ])
        
        return has_content

    async def cleanup(self):
        """Clean up resources."""
        logger.info(f"Cleaning up {self.name} agent")
