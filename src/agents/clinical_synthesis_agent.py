"""Clinical Synthesis Agent - Synthesizes clinical applications from research."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from .base_agent import BaseAgent
from src.models.enlitens_schemas import ClinicalContent
from src.synthesis.few_shot_library import FEW_SHOT_LIBRARY
from src.synthesis.ollama_client import OllamaClient
from src.utils.enlitens_constitution import EnlitensConstitution
from src.utils.prompt_briefing import compose_document_brief
from src.monitoring.error_telemetry import TelemetrySeverity, log_with_telemetry

logger = logging.getLogger(__name__)
TELEMETRY_AGENT = "clinical_synthesis_agent"


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
            log_with_telemetry(
                logger.error,
                "Failed to initialize %s: %s",
                self.name,
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Agent initialization failed",
                details={"error": str(e)},
            )
            return False

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize clinical applications from research."""
        try:
            science_data = context.get("science_data", {}) or {}
            research_content = science_data.get("research_content", {}) or {}
            sanitized_research = self.constitution.sanitize_mapping(research_content)
            research_payload = json.dumps(sanitized_research, ensure_ascii=False, indent=2)
            document_text = context.get("document_text", "") or ""
            retrieved_passages = context.get("retrieved_passages") or []
            document_brief = await compose_document_brief(
                document_text=document_text,
                retrieved_passages=retrieved_passages,
                ollama_client=self.ollama_client,
            )
            retrieved_block = self._render_retrieved_passages_block(
                retrieved_passages,
                raw_client_context=context.get("raw_client_context"),
                raw_founder_context=context.get("raw_founder_context"),
            )
            client_profile_bundle = context.get("client_profiles") or {}
            client_profile_data = client_profile_bundle.get("client_profiles") if isinstance(client_profile_bundle, dict) else {}
            client_profiles = client_profile_data.get("profiles") if isinstance(client_profile_data, dict) else []
            external_profile_sources = client_profile_data.get("external_sources") if isinstance(client_profile_data, dict) else []
            profile_block = self._render_client_profiles_block(client_profiles)
            external_research_block = self._render_external_sources_block(external_profile_sources)
            intake_registry_block = self._render_intake_registry_block(context.get("intake_registry") or {})
            health_priority_block = self._render_health_priority_block(context.get("health_report_summary") or {})

            constitution_block = self.constitution.render_prompt_section(
                self._prompt_principles,
                include_exemplars=True,
                header="ENLITENS CONSTITUTION – CLINICAL SYNTHESIS",
            )

            few_shot_block = FEW_SHOT_LIBRARY.render_for_prompt(
                task="clinical_synthesis",
                query=document_brief or document_text,
                k=2,
            )

            outline_prompt = f"""
You are planning an Enlitens-aligned clinical synthesis. Construct a JSON outline that captures the rebellious thesis, strength-first arcs, and system-level moves required by the constitution.

{constitution_block}

CLIENT PROFILES (fictional scenarios to ground clinical moves):
{profile_block}

EXTERNAL REGIONAL DATA (cite later with [Ext #] tags when used):
{external_research_block}

INTAKE REGISTRY HIGHLIGHTS:
{intake_registry_block}

HEALTH REPORT SIGNALS:
{health_priority_block}

RETRIEVED PASSAGES FOR CITATION:
{retrieved_block}

RESEARCH SNAPSHOT:
{research_payload}

DOCUMENT BRIEF (for tone and lived detail):
{document_brief}

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
• Weaving in external research and local realities, citing [Ext #] when you lean on those sources.

{constitution_block}

OUTLINE TO HONOUR:
{outline_json}

CLIENT PROFILES (fictional personas guiding specificity):
{profile_block}

EXTERNAL REGIONAL DATA (cite using [Ext #] tags when referenced):
{external_research_block}

INTAKE REGISTRY HIGHLIGHTS:
{intake_registry_block}

HEALTH REPORT SIGNALS:
{health_priority_block}

RESEARCH CONTENT (cleaned):
{research_payload}

DOCUMENT BRIEF (tone + late-stage insights):
{document_brief}

{exemplars}

RETRIEVED PASSAGES WITH SOURCE TAGS:
{retrieved_block}

Return JSON strictly matching {{"interventions": [str], "assessments": [str], "outcomes": [str], "protocols": [str], "guidelines": [str], "contraindications": [str], "side_effects": [str], "monitoring": [str]}}. Every list must contain 3-8 items and embed contextual analogies plus citations in plain language when relevant.
If you use a claim from the passages above, include the corresponding [Source #] tag in the string so downstream validation can trace it.
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
                processed = self._post_process_output(
                    result.model_dump(),
                    outline,
                    client_profiles=client_profiles,
                    external_sources=external_profile_sources,
                    intake_registry=context.get("intake_registry") or {},
                    health_summary=context.get("health_report_summary") or {},
                )
                return {
                    "clinical_content": processed,
                    "synthesis_quality": "high",
                    "synthesis_outline": outline.model_dump(),
                }

            fallback_processed = self._post_process_output(
                ClinicalContent().model_dump(),
                outline,
                client_profiles=client_profiles,
                external_sources=external_profile_sources,
                intake_registry=context.get("intake_registry") or {},
                health_summary=context.get("health_report_summary") or {},
            )
            return {
                "clinical_content": fallback_processed,
                "synthesis_quality": "needs_review",
                "synthesis_outline": outline.model_dump(),
            }

        except Exception as e:
            log_with_telemetry(
                logger.error,
                "Clinical synthesis failed: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Clinical synthesis failed",
                doc_id=context.get("document_id"),
                details={"error": str(e)},
            )
            return {"clinical_content": ClinicalContent().model_dump()}

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate the synthesized clinical content."""
        clinical_content = output.get("clinical_content")

        if not isinstance(clinical_content, dict):
            raise ValueError(
                "ClinicalSynthesisAgent validation failed: expected 'clinical_content' payload to be a mapping."
            )

        required_counts = {
            "interventions": 3,
            "assessments": 3,
            "outcomes": 3,
            "protocols": 3,
            "guidelines": 3,
            "contraindications": 3,
            "side_effects": 3,
            "monitoring": 3,
        }

        for field, minimum in required_counts.items():
            items = clinical_content.get(field)
            if not isinstance(items, list):
                raise ValueError(
                    f"ClinicalSynthesisAgent validation failed: field '{field}' must be a list; got {type(items).__name__}."
                )
            if len(items) < minimum:
                raise ValueError(
                    f"ClinicalSynthesisAgent validation failed: expected at least {minimum} items in '{field}', got {len(items)}."
                )

        return True

    async def cleanup(self):
        """Clean up resources."""
        logger.info(f"Cleaning up {self.name} agent")

    def _render_client_profiles_block(self, profiles: Sequence[Any]) -> str:
        if not profiles:
            return "- No fictional client profiles available; note the gap and borrow regional context explicitly."
        lines: List[str] = []
        for profile in profiles[:3]:
            if isinstance(profile, dict):
                name = profile.get("profile_name", "Unnamed persona")
                quote = profile.get("intake_reference", "")
                region = ", ".join(profile.get("regional_touchpoints", [])[:3])
                needs = ", ".join(profile.get("unmet_needs", [])[:2])
            else:
                name = getattr(profile, "profile_name", "Unnamed persona")
                quote = getattr(profile, "intake_reference", "")
                region = ", ".join(getattr(profile, "regional_touchpoints", [])[:3])
                needs = ", ".join(getattr(profile, "unmet_needs", [])[:2])
            lines.append(
                f"- {name}: {quote} | Regions: {region or 'unspecified'} | Needs: {needs or 'clarify during synthesis'}"
            )
        return "\n".join(lines)

    def _render_external_sources_block(self, sources: Sequence[Any]) -> str:
        if not sources:
            return "- No external regional sources harvested yet."
        lines: List[str] = []
        for source in sources[:6]:
            if isinstance(source, dict):
                label = source.get("label", "[Ext]")
                title = source.get("title", "Unnamed source")
                summary = source.get("summary", source.get("snippet", ""))
            else:
                label = getattr(source, "label", "[Ext]")
                title = getattr(source, "title", "Unnamed source")
                summary = getattr(source, "summary", getattr(source, "snippet", ""))
            lines.append(f"- {label} {title}: {summary}")
        return "\n".join(lines)

    def _render_intake_registry_block(self, intake_registry: Dict[str, Any]) -> str:
        if not isinstance(intake_registry, dict) or not intake_registry:
            return "- Intake registry unavailable."
        lines: List[str] = []
        top_locations = intake_registry.get("top_locations") or []
        if top_locations:
            lines.append(
                "Top intake localities: " + ", ".join(name for name, _ in top_locations[:5])
            )
        top_themes = intake_registry.get("top_themes") or []
        if top_themes:
            lines.append(
                "Dominant intake themes: " + ", ".join(theme for theme, _ in top_themes[:5])
            )
        return "\n".join(f"- {line}" for line in lines) if lines else "- Intake registry parsed but no high-signal entries."

    def _render_health_priority_block(self, health_summary: Dict[str, Any]) -> str:
        if not isinstance(health_summary, dict) or not health_summary:
            return "- Health report data unavailable."
        lines: List[str] = []
        for label, _ in (health_summary.get("priority_signals") or [])[:5]:
            lines.append(f"- Priority signal: {label}")
        mentions = health_summary.get("regional_mentions") or {}
        if mentions:
            ordered = sorted(mentions.items(), key=lambda item: (-item[1], item[0]))
            focus = ", ".join(name for name, _ in ordered[:5])
            lines.append(f"- Hotspot regions: {focus}")
        return "\n".join(lines) if lines else "- Health report parsed but no actionable signals."

    def _post_process_output(
        self,
        clinical_data: Dict[str, Any],
        outline: ClinicalOutline,
        *,
        client_profiles: Sequence[Any],
        external_sources: Sequence[Any],
        intake_registry: Dict[str, Any],
        health_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply constitution-aligned post-processing and guarantee minimum density."""

        sanitized = self.constitution.sanitize_mapping(clinical_data)

        defaults = ClinicalContent().model_dump()
        for field, default_value in defaults.items():
            value = sanitized.get(field)
            if value is None or not isinstance(value, list):
                sanitized[field] = list(default_value)

        strength_line = None
        if outline.client_strengths:
            joined = ", ".join(outline.client_strengths)
            strength_line = self.constitution.sanitize_language(
                f"Strengths spotlight: {joined} are leveraged as active therapy fuel."
            )

        lever_line = None
        if outline.key_system_levers:
            lever_line = self.constitution.sanitize_language(
                "System redesign targets: " + ", ".join(outline.key_system_levers)
            )

        rally_line = None
        if outline.rallying_cry:
            rally_line = self.constitution.sanitize_language(outline.rallying_cry)

        profile_supports = self._collect_profile_items(client_profiles, "support_recommendations")
        profile_needs = self._collect_profile_items(client_profiles, "unmet_needs")
        profile_cautions = self._collect_profile_items(client_profiles, "cautionary_flags")
        profile_masking = self._collect_profile_items(client_profiles, "masking_signals")
        external_lines = self._collect_external_summaries(external_sources)
        health_lines = self._collect_health_signals(health_summary)
        intake_lines = self._collect_intake_signals(intake_registry)

        sanitized["interventions"] = self._ensure_minimum_list(
            sanitized.get("interventions"),
            additions=(profile_supports + ([strength_line] if strength_line else [])),
        )

        sanitized["assessments"] = self._ensure_minimum_list(
            sanitized.get("assessments"),
            additions=profile_masking + profile_needs + intake_lines,
        )

        sanitized["outcomes"] = self._ensure_minimum_list(
            sanitized.get("outcomes"),
            additions=(health_lines + ([rally_line] if rally_line else [])),
        )

        sanitized["protocols"] = self._ensure_minimum_list(
            sanitized.get("protocols"),
            additions=external_lines + profile_supports,
        )

        additional_guidelines = profile_needs + ([lever_line] if lever_line else [])
        sanitized["guidelines"] = self._ensure_minimum_list(
            sanitized.get("guidelines"),
            additions=additional_guidelines,
        )

        sanitized["contraindications"] = self._ensure_minimum_list(
            sanitized.get("contraindications"),
            additions=profile_cautions,
        )

        sanitized["side_effects"] = self._ensure_minimum_list(
            sanitized.get("side_effects"),
            additions=[
                "Monitor for compassion fatigue when clients shoulder systemic change alone",
                "Watch for sensory crashes when new routines lack decompression buffers",
            ],
        )

        sanitized["monitoring"] = self._ensure_minimum_list(
            sanitized.get("monitoring"),
            additions=[
                "Schedule regular narrative check-ins with fictional personas to ensure masking load decreases",
                "Align monitoring metrics with regional access barriers flagged in health data",
            ],
        )

        return sanitized

    def _collect_profile_items(
        self, profiles: Sequence[Any], field: str, max_items: int = 12
    ) -> List[str]:
        items: List[str] = []
        for profile in profiles or []:
            value: Any = None
            if isinstance(profile, dict):
                value = profile.get(field)
            else:
                value = getattr(profile, field, None)
            if isinstance(value, list):
                for entry in value:
                    text = str(entry).strip()
                    if text:
                        items.append(text)
            elif isinstance(value, str) and value.strip():
                items.append(value.strip())
        return items[:max_items]

    def _collect_external_summaries(self, sources: Sequence[Any]) -> List[str]:
        summaries: List[str] = []
        for source in sources or []:
            if isinstance(source, dict):
                label = source.get("label", "[Ext]")
                summary = source.get("summary") or source.get("snippet")
            else:
                label = getattr(source, "label", "[Ext]")
                summary = getattr(source, "summary", getattr(source, "snippet", ""))
            if summary:
                summaries.append(f"{summary} ({label})")
        return summaries

    def _collect_health_signals(self, health_summary: Dict[str, Any]) -> List[str]:
        signals: List[str] = []
        if not isinstance(health_summary, dict):
            return signals
        for label, _ in (health_summary.get("priority_signals") or [])[:6]:
            signals.append(f"Health priority flagged: {label} [Source F1]")
        mentions = health_summary.get("regional_mentions") or {}
        for name in list(mentions.keys())[:4]:
            signals.append(f"Coordinate care with public health teams active in {name} [Source F1]")
        return signals

    def _collect_intake_signals(self, intake_registry: Dict[str, Any]) -> List[str]:
        signals: List[str] = []
        if not isinstance(intake_registry, dict):
            return signals
        for theme, _ in (intake_registry.get("top_themes") or [])[:6]:
            signals.append(f"Screen for intake theme: {theme}")
        for location, _ in (intake_registry.get("top_locations") or [])[:4]:
            signals.append(f"Assess resource accessibility within {location}")
        return signals

    def _ensure_minimum_list(
        self,
        current: Optional[Sequence[Any]],
        *,
        additions: Sequence[str],
        min_items: int = 3,
        max_items: int = 8,
    ) -> List[str]:
        unique: List[str] = []
        seen: set[str] = set()
        for item in current or []:
            text = self.constitution.sanitize_language(str(item))
            if not text:
                continue
            lowered = text.lower()
            if lowered not in seen:
                seen.add(lowered)
                unique.append(text)
        for addition in additions:
            clean = self.constitution.sanitize_language(str(addition))
            if not clean:
                continue
            lowered = clean.lower()
            if lowered not in seen:
                seen.add(lowered)
                unique.append(clean)
            if len(unique) >= max_items:
                break
        while len(unique) < min_items:
            filler = self.constitution.sanitize_language("Maintain fictional scenario guardrail – fill with contextual insight")
            if filler.lower() in seen:
                break
            unique.append(filler)
            seen.add(filler.lower())
        return unique[:max_items]
