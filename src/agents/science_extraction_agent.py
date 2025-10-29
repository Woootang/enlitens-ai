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

            # Use more text for better extraction
            text_excerpt = document_text[:8000] if len(document_text) > 8000 else document_text

            prompt = f"""
You are a neuroscience research analyst extracting comprehensive scientific content.

DOCUMENT TEXT:
{text_excerpt}

Extract ALL research content from this document. Be thorough and liberal in extraction - include both explicit statements and reasonable inferences. Even if information seems limited, extract what you can and infer related content.

Generate content for ALL fields below:

1. FINDINGS: List 5-15 key research findings, discoveries, or conclusions. Include main results, secondary findings, interesting observations, and supporting evidence. Be comprehensive!

2. STATISTICS: List 5-15 statistical data points, numbers, percentages, effect sizes, sample sizes, p-values, confidence intervals, or quantitative results. Include any numerical data mentioned.

3. METHODOLOGIES: List 5-15 research methods, study designs, experimental procedures, data collection methods, analysis techniques, participant selection methods, or measurement approaches used.

4. LIMITATIONS: List 3-10 study limitations, weaknesses, potential confounds, sample restrictions, methodological concerns, or areas needing further research. If not stated, infer reasonable limitations.

5. FUTURE_DIRECTIONS: List 3-10 suggestions for future research, unanswered questions, areas needing investigation, or next steps mentioned or implied by the research.

6. IMPLICATIONS: List 5-15 clinical implications, practical applications, therapeutic relevance, real-world applications, or how this research informs practice. Be creative in inferring applications!

7. CITATIONS: List 5-15 key papers cited, author names mentioned, referenced studies, or important sources. Extract any citation information present.

8. REFERENCES: List 5-15 bibliographic details, reference list entries, or source materials. Extract any reference information present.

IMPORTANT EXTRACTION RULES:
- Extract liberally - related content counts!
- If explicit information is missing, infer reasonable content from what IS present
- Break down complex findings into multiple list items
- Each list should have AT LEAST 3 items, preferably 5-15
- Use clear, complete sentences
- No numbering or bullet points in strings
- Better to over-extract than under-extract!

Return as JSON with these EXACT field names:
{{"findings": [list], "statistics": [list], "methodologies": [list], "limitations": [list], "future_directions": [list], "implications": [list], "citations": [list], "references": [list]}}
"""

            result = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=ResearchContent,
                temperature=0.6,  # Increased for better generation
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
