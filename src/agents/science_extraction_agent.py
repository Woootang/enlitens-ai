"""
Science Extraction Agent - Extracts scientific findings and neuroscience content.
"""

import logging
import math
from collections import Counter
from typing import Any, Dict, List, Tuple

from .base_agent import BaseAgent
from src.models.enlitens_schemas import ResearchContent
from src.synthesis.few_shot_library import FEW_SHOT_LIBRARY
from src.synthesis.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class ScienceExtractionAgent(BaseAgent):
    """Agent specialized in extracting scientific and research content."""

    def __init__(self):
        super().__init__(
            name="ScienceExtraction",
            role="Scientific Content Extraction",
            model="qwen2.5-32b-instruct-q4_k_m"
        )
        self.ollama_client = None
        self.self_consistency_temperatures = [0.1, 0.2, 0.3]
        self.vote_threshold_ratio = 0.5

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

            few_shot_block = FEW_SHOT_LIBRARY.render_for_prompt(
                task="science_extraction",
                query=text_excerpt,
                k=2,
            )

            exemplars = (
                "FEW-SHOT EXEMPLARS (follow citation style and refusal patterns):\n"
                f"{few_shot_block}\n\n" if few_shot_block else ""
            )

            prompt = f"""
You are a neuroscience research analyst extracting comprehensive scientific content from academic papers.

STRICT RULES:
✓ Extract ONLY information explicitly stated in the source document
✓ Quote exact statistics and data points - NO approximations
✓ Include source context for all citations (author, year, page if available)
✓ Mark any inferred implications as "Potential implication:" to distinguish from stated findings
✗ DO NOT fabricate statistics, citations, or research findings
✗ DO NOT add information from your training data
✗ DO NOT approximate numbers - use exact figures from the text

{exemplars}
DOCUMENT TEXT:
{text_excerpt}

Extract ALL research content from this document. Be thorough but accurate - only include what is explicitly stated or can be directly inferred from the text. If a required field cannot be filled because the evidence is missing, respond "Refusal: insufficient grounding in provided sources."

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
- Extract only what is explicitly stated in the source document
- Quote exact statistics with context - NO rounding or approximation
- For citations, include exact author names and years as stated in document
- For implications, distinguish between stated conclusions and potential applications
- Each list should have AT LEAST 3 items from the source document
- Use clear, complete sentences with exact quotes where appropriate
- No numbering or bullet points in strings
- Quality over quantity - accuracy is critical!

STATISTICS FORMAT:
- "According to [Author] ([Year]), [exact quoted statistic with numbers]"
- Include context: sample size, methodology, significance levels as stated

Return as JSON with these EXACT field names:
{{"findings": [list], "statistics": [list], "methodologies": [list], "limitations": [list], "future_directions": [list], "implications": [list], "citations": [list], "references": [list]}}
"""

            samples = await self._run_self_consistency_sampling(prompt)

            if samples:
                aggregated, vote_stats = self._aggregate_samples(samples)
                return {
                    "research_content": aggregated,
                    "extraction_quality": "high",
                    "self_consistency": vote_stats,
                }

            return {"research_content": ResearchContent().model_dump()}

        except Exception as e:
            logger.error(f"Science extraction failed: {e}")
            return {"research_content": ResearchContent().model_dump()}

    async def _run_self_consistency_sampling(self, prompt: str) -> List[ResearchContent]:
        """Sample multiple generations to improve factual reliability."""

        samples: List[ResearchContent] = []
        for temperature in self.self_consistency_temperatures:
            result = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=ResearchContent,
                temperature=temperature,
                max_retries=3,
                enforce_grammar=True,
            )
            if result:
                samples.append(result)
            else:
                logger.debug(
                    "ScienceExtraction self-consistency sample failed at temperature %.2f",
                    temperature,
                )
        return samples

    def _aggregate_samples(
        self, samples: List[ResearchContent]
    ) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
        """Aggregate list outputs using majority voting."""

        if not samples:
            return ResearchContent().model_dump(), {"num_samples": 0}

        sample_dicts = [sample.model_dump() for sample in samples]
        vote_threshold = max(1, math.ceil(len(samples) * self.vote_threshold_ratio))
        aggregated: Dict[str, List[str]] = {key: [] for key in sample_dicts[0].keys()}
        field_votes: Dict[str, Dict[str, int]] = {}

        for field in aggregated:
            counter: Counter[str] = Counter()
            original_map: Dict[str, str] = {}
            for sample in sample_dicts:
                for item in sample.get(field, []) or []:
                    normalized = self._normalize_value(item)
                    counter[normalized] += 1
                    original_map.setdefault(normalized, item)

            selected = [
                original_map[key]
                for key, count in counter.most_common()
                if count >= vote_threshold
            ]

            if not selected:
                richest_sample = max(
                    sample_dicts,
                    key=lambda sample: len(sample.get(field, []) or []),
                )
                selected = richest_sample.get(field, []) or []

            aggregated[field] = selected
            field_votes[field] = dict(counter)

        return aggregated, {
            "num_samples": len(samples),
            "vote_threshold": vote_threshold,
            "field_vote_counts": field_votes,
        }

    @staticmethod
    def _normalize_value(value: str) -> str:
        if not isinstance(value, str):
            return str(value)
        return " ".join(value.lower().split())

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate the extracted scientific content."""
        research_content = output.get("research_content", {})

        # Check if we have at least some content matching ResearchContent schema
        has_content = any([
            research_content.get("findings"),
            research_content.get("methodologies"),
            research_content.get("implications")
        ])

        return has_content

    async def cleanup(self):
        """Clean up resources."""
        logger.info(f"Cleaning up {self.name} agent")
