"""Clinical Synthesis Agent - Synthesizes clinical applications from research."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from .base_agent import BaseAgent
from src.models.enlitens_schemas import ClinicalContent
from src.synthesis.few_shot_library import FEW_SHOT_LIBRARY
from src.synthesis.ollama_client import OllamaClient
from src.utils.enlitens_constitution import EnlitensConstitution

logger = logging.getLogger(__name__)


class ClinicalOutline(BaseModel):
    """Structured outline that the synthesis stage elaborates on."""

    thesis: str = Field(..., description="Core rebellious thesis")
    sections: List[str] = Field(default_factory=list, description="Major sections to cover")
    client_strengths: List[str] = Field(default_factory=list, description="Strengths or adaptive capacities to spotlight")
    key_system_levers: List[str] = Field(default_factory=list, description="System-level actions or critiques")
    rallying_cry: str = Field(default="", description="Bold, precise call-to-action")

class ClinicalSynthesisAgent(BaseAgent):
    """Agent specialized in synthesizing clinical applications."""

    @staticmethod
    def _model_to_json(model: BaseModel, *, indent: int = 2) -> str:
        """Serialize a pydantic model to UTF-8 safe JSON.

        Pydantic v2 removed the ``ensure_ascii`` argument from ``model_dump_json``.
        This helper mirrors the previous behaviour by routing through ``model_dump``
        before handing control to ``json.dumps`` so downstream consumers continue
        receiving human-readable UTF-8 JSON payloads.
        """

        return json.dumps(model.model_dump(), indent=indent, ensure_ascii=False)

    def __init__(self):
        super().__init__(
            name="ClinicalSynthesis",
            role="Clinical Application Synthesis",
        )
        self.ollama_client = None
        self.constitution = EnlitensConstitution()
        self._prompt_principles = ["ENL-002", "ENL-005", "ENL-007", "ENL-008", "ENL-010"]

    async def initialize(self) -> bool:
        """Initialize the clinical synthesis agent."""
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
        """Synthesize clinical applications from research."""
        try:
            science_data = context.get("science_data", {}) or {}
            research_content = science_data.get("research_content", {}) or {}
            sanitized_research = self.constitution.sanitize_mapping(research_content)
            research_payload = json.dumps(sanitized_research, ensure_ascii=False, indent=2)
            document_text = (context.get("document_text", "") or "")[:5000]

            constitution_block = self.constitution.render_prompt_section(
                self._prompt_principles,
                include_exemplars=True,
                header="ENLITENS CONSTITUTION – CLINICAL SYNTHESIS",
            )

            few_shot_block = FEW_SHOT_LIBRARY.render_for_prompt(
                task="clinical_synthesis",
                query=document_text,
                k=2,
            )

            outline_prompt = f"""
You are planning an Enlitens-aligned clinical synthesis. Construct a JSON outline that captures the rebellious thesis, strength-first arcs, and system-level moves required by the constitution.

{constitution_block}

RESEARCH SNAPSHOT:
{research_payload}

DOCUMENT EXCERPT (for tone and lived detail):
{document_text}

Return JSON with fields {{"thesis": str, "sections": [str], "client_strengths": [str], "key_system_levers": [str], "rallying_cry": str}}.
"""

            outline = await self.ollama_client.generate_structured_response(
                prompt=outline_prompt,
                response_model=ClinicalOutline,
                temperature=0.2,
                max_retries=3,
                enforce_grammar=True,
            )

            if outline is None:
                outline = ClinicalOutline(
                    thesis="Context rewrites the story; the client is never the pathology.",
                    sections=[
                        "Strength Lens",
                        "Context Pressures",
                        "System Disruption",
                        "Future Autonomy",
                    ],
                    client_strengths=["Adaptive pattern recognition", "High-fidelity empathy"],
                    key_system_levers=["Workplace redesign", "Trauma-informed school supports"],
                    rallying_cry="We torch the lie that clients are broken and re-engineer the water they swim in.",
                )

            outline_json = self._model_to_json(outline)
            exemplars = (
                "FEW-SHOT EXEMPLARS (mirror structure, mark speculation clearly):\n"
                f"{few_shot_block}\n\n" if few_shot_block else ""
            )

            final_prompt = f"""
You are the Enlitens Clinical Synthesis Agent operating in two stages.

Stage 1 (already complete): Outline drafted below.
Stage 2: Expand that outline into the ClinicalContent schema while:
• Championing the Enlitens worldview with strengths-first storytelling and contextual analogies.
• Pairing every challenge with a systemic critique or environmental redesign lever.
• Using bold, precise voice – strategic profanity acceptable when it sharpens the point.
• Ensuring future autonomy is explicit: clients graduate with actionable roadmaps.

{constitution_block}

OUTLINE TO HONOUR:
{outline_json}

RESEARCH CONTENT (cleaned):
{research_payload}

{exemplars}

Return JSON strictly matching {{"interventions": [str], "assessments": [str], "outcomes": [str], "protocols": [str], "guidelines": [str], "contraindications": [str], "side_effects": [str], "monitoring": [str]}}. Every list must contain 3-8 items and embed contextual analogies plus citations in plain language when relevant.
"""

            cache_kwargs = self._cache_kwargs(context)
            result = await self.ollama_client.generate_structured_response(
                prompt=final_prompt,
                response_model=ClinicalContent,
                temperature=0.25,
                max_retries=3,
                enforce_grammar=True,
                **cache_kwargs,
            )

            if result:
                processed = self._post_process_output(result.model_dump(), outline)
                return {
                    "clinical_content": processed,
                    "synthesis_quality": "high",
                    "synthesis_outline": outline.model_dump(),
                }

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

    def _post_process_output(
        self, clinical_data: Dict[str, Any], outline: ClinicalOutline
    ) -> Dict[str, Any]:
        """Apply constitution-aligned post-processing to synthesis output."""

        sanitized = self.constitution.sanitize_mapping(clinical_data)

        # Ensure all expected keys exist as lists
        defaults = ClinicalContent().model_dump()
        for field, default_value in defaults.items():
            if field not in sanitized or sanitized[field] is None:
                sanitized[field] = default_value

        # Inject strengths-first reminder if missing
        strengths_line = None
        if outline.client_strengths:
            joined = ", ".join(outline.client_strengths)
            strengths_line = f"Strengths spotlight: {joined} are leveraged as active therapy fuel."
        if strengths_line:
            interventions = sanitized.get("interventions", [])
            if strengths_line not in interventions:
                interventions.append(self.constitution.sanitize_language(strengths_line))
                sanitized["interventions"] = interventions

        # Guarantee system accountability appears in guidelines
        if outline.key_system_levers:
            lever_line = "System redesign targets: " + ", ".join(outline.key_system_levers)
            guidelines = sanitized.get("guidelines", [])
            if lever_line not in guidelines:
                guidelines.append(self.constitution.sanitize_language(lever_line))
                sanitized["guidelines"] = guidelines

        # Embed autonomy rallying cry into outcomes if absent
        if outline.rallying_cry:
            outcomes = sanitized.get("outcomes", [])
            clean_rally = self.constitution.sanitize_language(outline.rallying_cry)
            if clean_rally not in outcomes:
                outcomes.append(clean_rally)
                sanitized["outcomes"] = outcomes

        return sanitized
