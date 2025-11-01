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
from src.utils.enlitens_constitution import EnlitensConstitution

logger = logging.getLogger(__name__)

class ScienceExtractionAgent(BaseAgent):
    """Agent specialized in extracting scientific and research content."""

    def __init__(self):
        super().__init__(
            name="ScienceExtraction",
            role="Scientific Content Extraction",
        )
        self.ollama_client = None
        self.self_consistency_temperatures = [0.1, 0.2, 0.3]
        self.vote_threshold_ratio = 0.5
        self.constitution = EnlitensConstitution()
        self._prompt_principles = ["ENL-001", "ENL-002", "ENL-003", "ENL-009"]

    async def initialize(self) -> bool:
        """Initialize the science extraction agent."""
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

            constitution_block = self.constitution.render_prompt_section(
                self._prompt_principles,
                include_exemplars=True,
                header="APPLY THESE ENLITENS PRINCIPLES DURING EXTRACTION",
            )

            prompt = f"""
You are the Enlitens Science Extraction Agent.

Mission: surface research facts that advance a neurodiversity-affirming analysis while dismantling pathology-first language.

{constitution_block}

STRICT EXTRACTION RULES:
• Pull ONLY evidence stated in the excerpt. Quote or paraphrase with precision and cite as given.
• Highlight context, systems, and environment anywhere a finding could be misread as individual deficit.
• When legacy terminology (DSM/ADOS/"disorder") appears, reframe it using constitution-approved language and note the legacy frame.
• Never invent statistics, citations, or references. If evidence is missing respond with "Refusal: insufficient grounding".

{exemplars}

DOCUMENT TEXT (truncated to relevant span):
{text_excerpt}

Return JSON with these EXACT fields:
{{"findings": [str], "statistics": [str], "methodologies": [str], "limitations": [str], "future_directions": [str], "implications": [str], "citations": [str], "references": [str]}}

Tone: factual, precise, and constitutionally aligned from the start.
"""

            cache_kwargs = self._cache_kwargs(context)
            primary_result = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=ResearchContent,
                temperature=0.3,  # LOWERED from 0.6: Research shows 0.3 optimal for factual extraction
                max_retries=3,
                **cache_kwargs,
            )
            samples: List[ResearchContent] = []
            if primary_result:
                samples.append(primary_result)

            samples.extend(await self._run_self_consistency_sampling(prompt))

            if samples:
                aggregated, vote_stats = self._aggregate_samples(samples)
                sanitized = self._apply_constitutional_filters(aggregated)
                return {
                    "research_content": sanitized,
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

    def _apply_constitutional_filters(
        self, aggregated: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Apply constitution sanitation to every extracted list."""

        cleaned: Dict[str, List[str]] = {}
        for field, values in aggregated.items():
            cleaned[field] = self.constitution.sanitize_list(values)
        return cleaned

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
