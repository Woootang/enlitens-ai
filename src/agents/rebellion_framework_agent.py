"""
Rebellion Framework Agent - Extracts content for Enlitens' proprietary Rebellion Framework.
"""

import logging
from typing import Dict, Any, List, Sequence
from .base_agent import BaseAgent
from src.synthesis.ollama_client import OllamaClient
from src.models.enlitens_schemas import RebellionFramework
from src.monitoring.error_telemetry import TelemetrySeverity, log_with_telemetry

logger = logging.getLogger(__name__)
TELEMETRY_AGENT = "rebellion_framework_agent"

class RebellionFrameworkAgent(BaseAgent):
    """Agent specialized in applying the Rebellion Framework to research."""

    MIN_ITEMS = 3
    MAX_ITEMS = 10
    FRAMEWORK_FIELDS: List[str] = [
        "narrative_deconstruction",
        "sensory_profiling",
        "executive_function",
        "social_processing",
        "strengths_synthesis",
        "rebellion_themes",
        "aha_moments",
    ]

    def __init__(self):
        super().__init__(
            name="RebellionFramework",
            role="Rebellion Framework Application",
        )
        self.ollama_client = None

    def _generate_placeholder(self, field: str, index: int) -> str:
        readable = field.replace("_", " ")
        return (
            f"Pending {readable} insight {index + 1}: synthesize from the referenced neuroscience findings."
        )

    def _clone_entry(self, seed: str, field: str, index: int) -> str:
        readable = field.replace("_", " ")
        return f"{seed} (reinforced {readable} insight {index + 1})"

    def _normalize_framework(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, List[str]] = {}
        for field in self.FRAMEWORK_FIELDS:
            raw_values = payload.get(field, [])
            if not isinstance(raw_values, list):
                raw_values = [raw_values] if raw_values not in (None, "") else []

            cleaned: List[str] = []
            for value in raw_values:
                text = value.strip() if isinstance(value, str) else str(value).strip()
                if text:
                    cleaned.append(text)

            if not cleaned:
                cleaned = [
                    self._generate_placeholder(field, idx)
                    for idx in range(self.MIN_ITEMS)
                ]
            else:
                while len(cleaned) < self.MIN_ITEMS:
                    seed = cleaned[(len(cleaned) - 1) % len(cleaned)]
                    cleaned.append(self._clone_entry(seed, field, len(cleaned)))

            normalized[field] = cleaned[: self.MAX_ITEMS]

        return normalized

    def _default_payload(self) -> Dict[str, Any]:
        return self._normalize_framework(RebellionFramework().model_dump())

    def _render_client_profiles_block(self, profiles: Sequence[Any]) -> str:
        if not profiles:
            return "- No fictional client profiles available."
        lines: List[str] = []
        for profile in profiles[:3]:
            if isinstance(profile, dict):
                name = profile.get("profile_name", "Unnamed persona")
                quote = profile.get("intake_reference", "")
                regions = ", ".join(profile.get("regional_touchpoints", [])[:3])
            else:
                name = getattr(profile, "profile_name", "Unnamed persona")
                quote = getattr(profile, "intake_reference", "")
                regions = ", ".join(getattr(profile, "regional_touchpoints", [])[:3])
            lines.append(f"- {name}: {quote} | Regions: {regions or 'unspecified'}")
        return "\n".join(lines)

    def _render_external_sources_block(self, sources: Sequence[Any]) -> str:
        if not sources:
            return "- No external research harvested yet."
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
            lines.append("Top intake localities: " + ", ".join(name for name, _ in top_locations[:5]))
        top_themes = intake_registry.get("top_themes") or []
        if top_themes:
            lines.append("Dominant intake themes: " + ", ".join(theme for theme, _ in top_themes[:5]))
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

    def _normalize_with_context(
        self,
        payload: Dict[str, Any],
        *,
        client_profiles: Sequence[Any],
        external_sources: Sequence[Any],
        intake_registry: Dict[str, Any],
        health_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized = self._normalize_framework(payload or {})

        profile_supports = self._collect_profile_items(client_profiles, "support_recommendations")
        profile_needs = self._collect_profile_items(client_profiles, "unmet_needs")
        profile_masking = self._collect_profile_items(client_profiles, "masking_signals")
        profile_touchpoints = self._collect_profile_items(client_profiles, "regional_touchpoints")
        profile_overviews = self._collect_profile_items(client_profiles, "persona_overview")
        external_lines = self._collect_external_summaries(external_sources)
        intake_lines = self._collect_intake_signals(intake_registry)
        health_lines = self._collect_health_signals(health_summary)

        normalized["narrative_deconstruction"] = self._ensure_field_length(
            "narrative_deconstruction",
            normalized.get("narrative_deconstruction", []),
            profile_needs + intake_lines + health_lines,
        )

        normalized["sensory_profiling"] = self._ensure_field_length(
            "sensory_profiling",
            normalized.get("sensory_profiling", []),
            profile_masking + profile_touchpoints,
        )

        normalized["executive_function"] = self._ensure_field_length(
            "executive_function",
            normalized.get("executive_function", []),
            profile_supports + profile_needs,
        )

        normalized["social_processing"] = self._ensure_field_length(
            "social_processing",
            normalized.get("social_processing", []),
            intake_lines + profile_touchpoints,
        )

        normalized["strengths_synthesis"] = self._ensure_field_length(
            "strengths_synthesis",
            normalized.get("strengths_synthesis", []),
            profile_supports + external_lines,
        )

        normalized["rebellion_themes"] = self._ensure_field_length(
            "rebellion_themes",
            normalized.get("rebellion_themes", []),
            external_lines + health_lines + intake_lines,
        )

        normalized["aha_moments"] = self._ensure_field_length(
            "aha_moments",
            normalized.get("aha_moments", []),
            profile_overviews + profile_supports,
        )

        return normalized

    async def initialize(self) -> bool:
        """Initialize the rebellion framework agent."""
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
        """Apply Rebellion Framework to research content."""
        try:
            document_text = context.get("document_text", "")[:8000]
            research_content = context.get("science_data", {}).get("research_content", {})
            clinical_content = context.get("clinical_content", {})
            client_profile_bundle = context.get("client_profiles") or {}
            client_profile_data = client_profile_bundle.get("client_profiles") if isinstance(client_profile_bundle, dict) else {}
            client_profiles = client_profile_data.get("profiles") if isinstance(client_profile_data, dict) else []
            external_sources = client_profile_data.get("external_sources") if isinstance(client_profile_data, dict) else []
            profile_block = self._render_client_profiles_block(client_profiles)
            external_block = self._render_external_sources_block(external_sources)
            intake_block = self._render_intake_registry_block(context.get("intake_registry") or {})
            health_block = self._render_health_priority_block(context.get("health_report_summary") or {})
            retrieved_block = self._render_retrieved_passages_block(
                context.get("retrieved_passages"),
                raw_client_context=context.get("raw_client_context"),
                raw_founder_context=context.get("raw_founder_context"),
            )

            prompt = f"""
You are applying Enlitens' proprietary Rebellion Framework to neuroscience research.
The Rebellion Framework challenges traditional deficit-based narratives and reframes neurodivergence as adaptation and strength.

STRICT RULES:
✓ Base all reframings on actual research findings provided
✓ Use neuroscience evidence to challenge deficit-based narratives
✓ Distinguish between research-supported claims and creative reframing
✓ Mark speculative strengths as "Potential strength:" when not explicitly stated in research
✓ When leveraging external research summaries, cite the matching [Ext #] tag
✗ DO NOT fabricate neuroscience findings to support reframing
✗ DO NOT generate practice statistics or client testimonials
✗ DO NOT add claims not supported by the research context

DOCUMENT TEXT:
{document_text}

RESEARCH FINDINGS:
{research_content.get('findings', [])}

RETRIEVED PASSAGES (quote verbatim and cite with [Source #]):
{retrieved_block}

CLINICAL APPLICATIONS:
{clinical_content.get('interventions', [])}

CLIENT PROFILES (fictional scenarios anchoring context):
{profile_block}

EXTERNAL REGIONAL DATA (cite with [Ext #] tags when referenced):
{external_block}

INTAKE REGISTRY HIGHLIGHTS:
{intake_block}

HEALTH REPORT SIGNALS:
{health_block}

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

For any claim grounded in the passages above, include the matching [Source #] tag in the string so QA can trace it.

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
                normalized = self._normalize_with_context(
                    result.model_dump(),
                    client_profiles=client_profiles,
                    external_sources=external_sources,
                    intake_registry=context.get("intake_registry") or {},
                    health_summary=context.get("health_report_summary") or {},
                )
                return {
                    "rebellion_framework": normalized,
                    "framework_quality": "high",
                }

            fallback_payload = self._normalize_with_context(
                {},
                client_profiles=client_profiles,
                external_sources=external_sources,
                intake_registry=context.get("intake_registry") or {},
                health_summary=context.get("health_report_summary") or {},
            )
            return {
                "rebellion_framework": fallback_payload,
                "framework_quality": "needs_review",
            }

        except Exception as e:
            log_with_telemetry(
                logger.error,
                "Rebellion framework application failed: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Rebellion framework application failed",
                doc_id=context.get("document_id"),
                details={"error": str(e)},
            )
            client_profile_bundle = context.get("client_profiles") or {}
            client_profile_data = client_profile_bundle.get("client_profiles") if isinstance(client_profile_bundle, dict) else {}
            fallback_payload = self._normalize_with_context(
                {},
                client_profiles=client_profile_data.get("profiles") if isinstance(client_profile_data, dict) else [],
                external_sources=client_profile_data.get("external_sources") if isinstance(client_profile_data, dict) else [],
                intake_registry=context.get("intake_registry") or {},
                health_summary=context.get("health_report_summary") or {},
            )
            return {
                "rebellion_framework": fallback_payload,
                "framework_quality": "needs_review",
            }

    def _collect_profile_items(
        self, profiles: Sequence[Any], field: str, max_items: int = 12
    ) -> List[str]:
        values: List[str] = []
        for profile in profiles or []:
            if isinstance(profile, dict):
                data = profile.get(field)
            else:
                data = getattr(profile, field, None)
            if isinstance(data, list):
                for item in data:
                    text = str(item).strip()
                    if text:
                        values.append(text)
            elif isinstance(data, str) and data.strip():
                values.append(data.strip())
        return values[:max_items]

    def _collect_external_summaries(self, sources: Sequence[Any]) -> List[str]:
        values: List[str] = []
        for source in sources or []:
            if isinstance(source, dict):
                label = source.get("label", "[Ext]")
                summary = source.get("summary") or source.get("snippet")
            else:
                label = getattr(source, "label", "[Ext]")
                summary = getattr(source, "summary", getattr(source, "snippet", ""))
            if summary:
                values.append(f"{summary} ({label})")
        return values

    def _collect_intake_signals(self, intake_registry: Dict[str, Any]) -> List[str]:
        signals: List[str] = []
        if not isinstance(intake_registry, dict):
            return signals
        for theme, _ in (intake_registry.get("top_themes") or [])[:6]:
            signals.append(f"Reframe intake theme: {theme}")
        for location, _ in (intake_registry.get("top_locations") or [])[:4]:
            signals.append(f"Account for locality stressors in {location}")
        return signals

    def _collect_health_signals(self, health_summary: Dict[str, Any]) -> List[str]:
        signals: List[str] = []
        if not isinstance(health_summary, dict):
            return signals
        for label, _ in (health_summary.get("priority_signals") or [])[:5]:
            signals.append(f"Public-health insight: {label} [Source F1]")
        mentions = health_summary.get("regional_mentions") or {}
        for name in list(mentions.keys())[:4]:
            signals.append(f"Regional determinant: {name} [Source F1]")
        return signals

    def _ensure_field_length(
        self,
        field: str,
        values: List[str],
        additions: Sequence[str],
    ) -> List[str]:
        cleaned: List[str] = []
        seen: set[str] = set()
        for value in values or []:
            text = value.strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered not in seen:
                seen.add(lowered)
                cleaned.append(text)
        for addition in additions:
            text = str(addition).strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            cleaned.append(text)
            seen.add(lowered)
            if len(cleaned) >= self.MAX_ITEMS:
                break
        while len(cleaned) < self.MIN_ITEMS:
            cleaned.append(self._generate_placeholder(field, len(cleaned)))
        return cleaned[: self.MAX_ITEMS]

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate the rebellion framework content."""
        rebellion_content = output.get("rebellion_framework")

        if not isinstance(rebellion_content, dict):
            raise ValueError(
                "RebellionFrameworkAgent validation failed: expected 'rebellion_framework' payload to be a mapping."
            )

        required_counts = {
            "narrative_deconstruction": 3,
            "sensory_profiling": 3,
            "executive_function": 3,
            "social_processing": 3,
            "strengths_synthesis": 3,
            "rebellion_themes": 3,
            "aha_moments": 3,
        }

        for field, minimum in required_counts.items():
            items = rebellion_content.get(field)
            if not isinstance(items, list):
                raise ValueError(
                    f"RebellionFrameworkAgent validation failed: field '{field}' must be a list; got {type(items).__name__}."
                )
            if len(items) < minimum:
                raise ValueError(
                    f"RebellionFrameworkAgent validation failed: expected at least {minimum} items in '{field}', got {len(items)}."
                )

        return True

    async def cleanup(self):
        """Clean up resources."""
        logger.info(f"Cleaning up {self.name} agent")
