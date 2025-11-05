"""Client Profile Agent - Links intake language to retrieved research citations."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from functools import lru_cache
from itertools import chain
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pydantic import ValidationError

from .base_agent import BaseAgent
from src.data.locality_loader import LocalityRecord, load_locality_reference
from src.models.enlitens_schemas import (
    ClientProfile,
    ClientProfileSet,
    ExternalResearchSource,
)
from src.models.prediction_error import PredictionErrorEntry
from src.orchestration.research_orchestrator import (
    ExternalResearchOrchestrator,
    ResearchHit,
    ResearchQuery,
)
from src.synthesis.ollama_client import OllamaClient
from src.utils.enlitens_constitution import EnlitensConstitution
from src.monitoring.error_telemetry import TelemetrySeverity, log_with_telemetry

logger = logging.getLogger(__name__)
TELEMETRY_AGENT = "client_profile_agent"


def _canonicalize_label(value: str) -> str:
    text = re.sub(r"[^\w\s]", " ", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().lower()


@lru_cache(maxsize=1)
def _locality_index() -> Dict[str, LocalityRecord]:
    records = load_locality_reference()
    index: Dict[str, LocalityRecord] = {}
    for record in records.values():
        key = _canonicalize_label(record.name)
        if key:
            index[key] = record
    return index


def _match_locality(label: str) -> Optional[LocalityRecord]:
    if not label:
        return None

    index = _locality_index()
    canonical = _canonicalize_label(label)
    record = index.get(canonical)
    if record:
        return record

    probes: List[str] = []
    for delimiter in ("-", "—", "–", "|", "/"):
        if delimiter in label:
            probes.append(label.split(delimiter, 1)[0].strip())
    if "(" in label:
        probes.append(label.split("(", 1)[0].strip())

    for probe in probes:
        candidate = index.get(_canonicalize_label(probe))
        if candidate:
            return candidate

    return None


@lru_cache(maxsize=1)
def _global_asset_index() -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for record in _locality_index().values():
        canonical_locality = _canonicalize_label(record.name)
        for asset in chain(
            record.landmark_schools,
            record.youth_sports_leagues,
            record.community_centers,
            record.health_resources,
            record.signature_eateries,
        ):
            canonical_asset = _canonicalize_label(asset)
            if not canonical_asset:
                continue
            entry = index.setdefault(
                canonical_asset,
                {"label": asset, "localities": set()},
            )
            entry["localities"].add(canonical_locality)
    return index


def _assets_for_records(records: Sequence[LocalityRecord]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for record in records:
        for asset in chain(
            record.landmark_schools,
            record.youth_sports_leagues,
            record.community_centers,
            record.health_resources,
            record.signature_eateries,
        ):
            canonical = _canonicalize_label(asset)
            if canonical and canonical not in mapping:
                mapping[canonical] = asset
    return mapping


class ClientProfileAgent(BaseAgent):
    """Agent that grounds research passages in real client intake language."""

    def __init__(self) -> None:
        super().__init__(
            name="ClientProfiles",
            role="Intake-grounded client persona synthesis",
        )
        self.ollama_client: Optional[OllamaClient] = None
        self.constitution = EnlitensConstitution()
        self._prompt_principles: Sequence[str] = ("ENL-002", "ENL-003", "ENL-010")
        self.research_orchestrator: Optional[ExternalResearchOrchestrator] = None
        self._fallback_disclaimer = (
            "FICTIONAL COMPOSITE CASE STUDY – INTERNAL TRAINING USE ONLY. NOT A REAL CLIENT."
        )

    async def initialize(self) -> bool:
        try:
            self.ollama_client = OllamaClient(default_model=self.model)
            if not await self.ollama_client.check_connection():
                raise RuntimeError(
                    "Language model backend unavailable for ClientProfileAgent."
                )
            if self.research_orchestrator is None:
                self.research_orchestrator = ExternalResearchOrchestrator.from_settings()
            self.is_initialized = True
            logger.info("✅ %s agent initialized", self.name)
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            log_with_telemetry(
                logger.error,
                "Failed to initialize %s: %s",
                self.name,
                exc,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Agent initialization failed",
                details={"error": str(exc)},
            )
            return False

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        document_id = context.get("document_id") or "document"
        external_sources: List[ExternalResearchSource] = []

        try:
            if self.ollama_client is None:
                raise RuntimeError("ClientProfileAgent not initialized")

            retrieved_passages = context.get("retrieved_passages") or []
            raw_client_context = context.get("raw_client_context")
            client_insights = context.get("client_insights") or {}
            st_louis_context = context.get("st_louis_context") or {}
            intake_registry = context.get("intake_registry") or {}
            transcript_registry = context.get("transcript_registry") or {}
            regional_atlas = context.get("regional_atlas") or {}
            health_summary = context.get("health_report_summary") or {}
            document_localities = context.get("document_locality_matches") or []

            intake_quotes = self._collect_intake_quotes(client_insights, raw_client_context)
            intake_segments = self._select_relevant_intake_segments(
                intake_registry=intake_registry,
                document_localities=document_localities,
                retrieved_passages=retrieved_passages,
                fallback_quotes=intake_quotes,
            )
            intake_block = self._render_intake_block(intake_quotes, raw_client_context)
            intake_registry_block = self._render_intake_registry_block(intake_segments)
            retrieved_block = self._render_retrieved_passages_block(
                retrieved_passages,
                raw_client_context=raw_client_context,
                raw_founder_context=context.get("raw_founder_context"),
                max_passages=6,
            )
            regional_focus_block = self._render_regional_focus_block(
                document_localities=document_localities,
                intake_registry=intake_registry,
                regional_atlas=regional_atlas,
                transcript_registry=transcript_registry,
            )
            health_priority_block = self._render_health_priority_block(
                health_summary=health_summary,
                document_localities=document_localities,
            )
            research_focus_block = self._render_research_focus_block(retrieved_passages)

            external_research_context = await self._gather_external_research(
                context=context,
                intake_segments=intake_segments,
                document_localities=document_localities,
            )
            external_sources = external_research_context.get("sources", [])
            external_block = external_research_context.get("prompt_block", "No external research harvested.")

            constitution_block = self.constitution.render_prompt_section(
                self._prompt_principles,
                include_exemplars=True,
                header="ENLITENS CONSTITUTION – CLIENT PROFILE GUARANTEES",
            )

            st_louis_block = (
                json.dumps(st_louis_context, ensure_ascii=False, indent=2)
                if st_louis_context
                else "No explicit St. Louis context provided. If absent, note the gap but still tie benefits to community realities."
            )

            schema_hint = (
                "{\n"
                "  \"profiles\": [\n"
                "    {\n"
                "      \"profile_name\": str,\n"
                "      \"fictional_disclaimer\": \"FICTIONAL COMPOSITE CASE STUDY – INTERNAL TRAINING USE ONLY. NOT A REAL CLIENT.\",\n"
                "      \"intake_reference\": str,\n"
                "      \"persona_overview\": str,\n"
                "      \"research_reference\": str,\n"
                "      \"benefit_explanation\": str,\n"
                "      \"prediction_errors\": [\n"
                "        {\n"
                "          \"trigger_context\": str,\n"
                "          \"surprising_pivot\": str,\n"
                "          \"intended_cognitive_effect\": str\n"
                "        }\n"
                "      ],\n"
                "      \"st_louis_alignment\": str | null,\n"
                "      \"regional_touchpoints\": [str],\n"
                "      \"masking_signals\": [str],\n"
                "      \"unmet_needs\": [str],\n"
                "      \"support_recommendations\": [str],\n"
                "      \"cautionary_flags\": [str]\n"
                "    }\n"
                "  ],\n"
                "  \"shared_thread\": str | null,\n"
                "  \"external_sources\": [\n"
                "    {\n"
                "      \"label\": \"[Ext #]\",\n"
                "      \"title\": str,\n"
                "      \"url\": str,\n"
                "      \"publisher\": str | null,\n"
                "      \"published_at\": str | null,\n"
                "      \"summary\": str,\n"
                "      \"verification_status\": \"verified\" | \"needs_review\" | \"flagged\"\n"
                "    }\n"
                "  ]\n"
                "}"
            )

            prompt = f"""
You are the Enlitens Client Profile Agent. Create three fictional personas that weave intake language, St. Louis locality, and cited research into actionable case studies without promising clinical outcomes.

MANDATORY CONTEXT REMINDERS:
• Always acknowledge the Delmar Divide dynamics when relevant and note cross-river (Illinois/Missouri) perspectives when present.
• Speak with respect toward rural or conservative caregivers—never stereotype or dismiss their values.
• Cite every research-derived insight with [Source #] or [Ext #] tags and keep tone empathetic and community-grounded.

{constitution_block}

INTAKE LANGUAGE TO HONOUR (reuse phrases verbatim inside quotes):
{intake_block}

ST. LOUIS CONTEXT SNAPSHOT:
{st_louis_block}

REGIONAL ATLAS SIGNALS (match personas to these neighbourhoods/towns):
{regional_focus_block}

HEALTH REPORT PRIORITIES TO RESPECT:
{health_priority_block}

INTAKE COHORT SNAPSHOT (ground profiles in these segments):
{intake_registry_block}

EXTERNAL RESEARCH & LOCAL DATA (cite using [Ext #] tags when referenced):
{external_block}

RESEARCH FOCUS NOTES:
{research_focus_block}

RETRIEVED PASSAGES WITH SOURCE TAGS:
{retrieved_block}

OUTPUT REQUIREMENTS:
1. Produce exactly THREE distinct profiles in JSON.
2. Start each profile with a "fictional_disclaimer" field containing exactly: "FICTIONAL COMPOSITE CASE STUDY – INTERNAL TRAINING USE ONLY. NOT A REAL CLIENT." Place it before any narrative content.
3. Anchor each persona to a *unique* municipality/neighbourhood or micro-community listed above. Avoid conflicting demographics.
4. Each "intake_reference" must reuse the exact client phrasing inside quotes. If no quote exists, say "No direct intake quote provided" and flag the gap.
5. "research_reference" AND "benefit_explanation" must each cite at least one [Source #] tag from the passages above.
6. Reference external data with [Ext #] tags when you draw from the external research block and clarify fictionalization when extrapolating community detail.
7. Populate "prediction_errors" with 2–4 locally grounded surprises that cite [Source #] or [Ext #] tags and describe the intended cognitive effect.
8. In "persona_overview", create clearly labeled subsections for: "Neighborhood & Daily Geography", "Family & Intergenerational History", "Economic Context & Access Gaps", "Sensory & Community Experiences", and "Local Supports (schools, leagues, churches, eateries)"—each grounded in the provided data.
9. Use "support_recommendations" for focus areas only—never promise outcomes, cures, or success rates.
10. Explicitly remind the reader within each profile that all personas are fictional composites when citing or paraphrasing sources.
11. Return JSON exactly in this shape (no commentary). Keep persona names short but evocative ("Bevo Mill sensory de-masker"):
{schema_hint}
"""

            response = await self._structured_generation(
                prompt=prompt,
                context=context,
                intake_quotes=intake_quotes,
                retrieved_passages=retrieved_passages,
                st_louis_context=st_louis_context,
                intake_segments=intake_segments,
                external_sources=external_sources,
                document_id=document_id,
            )

            response = self._post_process_profiles(
                response=response,
                document_id=document_id,
                document_localities=document_localities,
                external_sources=external_sources,
                regional_atlas=regional_atlas,
            )

            payload = response.model_dump()
            source_tags = self._collect_source_tags(response)
            metrics = self._summarize_profile_metrics(
                response,
                document_localities=document_localities,
            )
            metrics.update(
                {
                    "source_tag_count": len(source_tags),
                    "source_tags": sorted(source_tags),
                }
            )
            log_with_telemetry(
                logger.info,
                "Client profiles generated successfully",
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MINOR,
                impact="Client profile set ready",
                doc_id=document_id,
                details={"metrics": metrics},
            )
            return {
                "client_profiles": payload,
                "client_profile_summary": {
                    "intake_quotes_used": [p.intake_reference for p in response.profiles],
                    "source_tags": sorted(source_tags),
                    "external_sources_used": [
                        {
                            "label": src.label,
                            "url": str(src.url),
                            "verification_status": src.verification_status,
                        }
                        for src in response.external_sources
                    ],
                    "shared_thread": response.shared_thread,
                },
            }
        except Exception as exc:
            log_with_telemetry(
                logger.error,
                "ClientProfileAgent failed: %s",
                exc,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Client profile generation failed",
                doc_id=context.get("document_id"),
                details={"error": str(exc)},
            )
            fallback = self._fallback_profiles(
                intake_quotes=intake_quotes,
                intake_segments=intake_segments,
                retrieved_passages=retrieved_passages,
                st_louis_context=st_louis_context,
                regional_atlas=regional_atlas,
                document_localities=document_localities,
                external_sources=external_sources,
                document_id=document_id,
            )
            fallback_metrics = self._summarize_profile_metrics(
                fallback,
                document_localities=document_localities,
            )
            fallback_source_tags = self._collect_source_tags(fallback)
            fallback_metrics.update(
                {
                    "source_tag_count": len(fallback_source_tags),
                    "source_tags": sorted(fallback_source_tags),
                }
            )
            log_with_telemetry(
                logger.warning,
                "ClientProfileAgent returned fallback profiles after exception",
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Fallback client profiles used",
                doc_id=context.get("document_id"),
                details={"metrics": fallback_metrics},
            )
            return {
                "client_profiles": fallback.model_dump(),
                "client_profile_summary": {
                    "intake_quotes_used": [p.intake_reference for p in fallback.profiles],
                    "source_tags": sorted(fallback_source_tags),
                    "external_sources_used": [
                        {
                            "label": src.label,
                            "url": str(src.url),
                            "verification_status": src.verification_status,
                        }
                        for src in fallback.external_sources
                    ],
                    "shared_thread": fallback.shared_thread,
                },
            }

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        payload = output.get("client_profiles")
        if not isinstance(payload, dict):
            raise ValueError("ClientProfileAgent expected 'client_profiles' mapping in output")
        try:
            profiles = ClientProfileSet.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"ClientProfileAgent returned invalid schema: {exc}") from exc

        if len(profiles.profiles) != 3:
            raise ValueError("ClientProfileAgent must return exactly three profiles")

        name_set: set[str] = set()
        for profile in profiles.profiles:
            if profile.profile_name.lower() in name_set:
                raise ValueError("Client profile names must be unique")
            name_set.add(profile.profile_name.lower())
            if "[Source" not in profile.research_reference:
                raise ValueError("Each research_reference must cite a [Source #]")
            if "[Source" not in profile.benefit_explanation:
                raise ValueError("Each benefit_explanation must cite a [Source #]")
            if not any(marker in profile.intake_reference for marker in ('"', "'", "“", "”")):
                raise ValueError("Intake references must include quoted client language")
            if "fictional" not in profile.fictional_disclaimer.lower():
                raise ValueError("Fictional disclaimer must clearly state the scenario is fictional")
            if len(profile.prediction_errors) < 2:
                raise ValueError("Each profile must include at least two prediction_errors entries")

        for source in profiles.external_sources:
            if "[Ext" not in source.label:
                raise ValueError("External sources must have [Ext #] labels")

        return True

    async def _structured_generation(
        self,
        *,
        prompt: str,
        context: Dict[str, Any],
        intake_quotes: List[str],
        retrieved_passages: Sequence[Dict[str, Any]],
        st_louis_context: Dict[str, Any],
        intake_segments: Sequence[Dict[str, Any]] | None = None,
        external_sources: Sequence[ExternalResearchSource] | None = None,
        document_id: str = "document",
    ) -> ClientProfileSet:
        if self.ollama_client is None:
            raise RuntimeError("ClientProfileAgent not initialized")

        cache_kwargs = self._cache_kwargs(context, suffix="profiles")
        citation_map = context.get("citation_map") or {}
        source_segments = context.get("source_segments") or []

        profile_temperature = getattr(self.settings.llm, "profile_temperature", 0.45)
        if profile_temperature is None:
            profile_temperature = 0.45
        profile_top_p = getattr(self.settings.llm, "profile_top_p", None)

        try:
            raw_response = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=ClientProfileSet,
                temperature=profile_temperature,
                top_p=profile_top_p,
                max_retries=3,
                enforce_grammar=True,
                **cache_kwargs,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            log_with_telemetry(
                logger.warning,
                "Structured generation raised error: %s",
                exc,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MINOR,
                impact="Client profile structured generation error",
                doc_id=context.get("document_id"),
            )
            raw_response = None

        if isinstance(raw_response, ClientProfileSet):
            return raw_response

        source_tags: List[str] = []
        if isinstance(source_segments, list):
            for segment in source_segments:
                tag = segment.get("tag") if isinstance(segment, dict) else None
                if tag and tag not in source_tags:
                    source_tags.append(str(tag))
        if not source_tags:
            source_tags = [f"[Source {idx}]" for idx, _ in enumerate(retrieved_passages, start=1)]
        if not source_tags and isinstance(citation_map, dict):
            source_tags = [str(tag) for tag in citation_map.keys()]

        normalized = self._normalize_partial_profiles(
            raw_response,
            source_tags=source_tags,
            citation_map=citation_map,
        )
        if normalized is not None:
            try:
                return ClientProfileSet.model_validate(normalized)
            except ValidationError as exc:
                log_with_telemetry(
                    logger.warning,
                    "Partial profile payload failed validation: %s",
                    exc,
                    agent=TELEMETRY_AGENT,
                    severity=TelemetrySeverity.MINOR,
                    impact="Client profile partial validation failed",
                    doc_id=context.get("document_id"),
                )

        log_with_telemetry(
            logger.warning,
            "⚠️ ClientProfileAgent falling back to deterministic profiles",
            agent=TELEMETRY_AGENT,
            severity=TelemetrySeverity.MINOR,
            impact="Client profiles fallback",
            doc_id=context.get("document_id"),
        )
        document_localities = context.get("document_locality_matches") or []
        regional_atlas = context.get("regional_atlas") or {}
        fallback_profiles = self._fallback_profiles(
            intake_quotes=intake_quotes,
            intake_segments=intake_segments or [],
            retrieved_passages=retrieved_passages,
            st_louis_context=st_louis_context,
            source_tags=source_tags,
            citation_map=citation_map,
            client_insights=context.get("client_insights") or {},
            regional_atlas=regional_atlas,
            document_localities=document_localities,
            external_sources=list(external_sources or []),
            document_id=document_id,
        )
        fallback_metrics = self._summarize_profile_metrics(
            fallback_profiles,
            document_localities=document_localities,
        )
        fallback_source_tags = self._collect_source_tags(fallback_profiles)
        fallback_metrics.update(
            {
                "source_tag_count": len(fallback_source_tags),
                "source_tags": sorted(fallback_source_tags),
            }
        )
        log_with_telemetry(
            logger.warning,
            "⚠️ ClientProfileAgent falling back to deterministic profiles",
            agent=TELEMETRY_AGENT,
            severity=TelemetrySeverity.MINOR,
            impact="Client profiles fallback",
            doc_id=context.get("document_id"),
            details={"metrics": fallback_metrics},
        )
        return fallback_profiles

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_partial_profiles(
        self,
        payload: Any,
        *,
        source_tags: Sequence[str],
        citation_map: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        if payload is None:
            return None
        if isinstance(payload, ClientProfileSet):
            return payload.model_dump()
        if not isinstance(payload, dict):
            return None

        allowed_keys = {
            "profile_name",
            "fictional_disclaimer",
            "intake_reference",
            "persona_overview",
            "research_reference",
            "benefit_explanation",
            "prediction_errors",
            "st_louis_alignment",
            "local_geography",
            "community_connections",
            "regional_touchpoints",
            "masking_signals",
            "unmet_needs",
            "support_recommendations",
            "cautionary_flags",
        }

        normalized: Dict[str, Any] = {}
        shared = payload.get("shared_thread")
        if shared is None or isinstance(shared, str):
            normalized["shared_thread"] = shared

        raw_profiles = payload.get("profiles")
        if not raw_profiles:
            return None

        def normalize_prediction_error_items(items: Any) -> List[Dict[str, str]]:
            normalized: List[Dict[str, str]] = []
            if isinstance(items, PredictionErrorEntry):
                return [items.model_dump()]
            if isinstance(items, list):
                for item in items:
                    normalized.extend(normalize_prediction_error_items(item))
                return normalized
            if isinstance(items, dict):
                entry: Dict[str, str] = {}
                for key in (
                    "trigger_context",
                    "surprising_pivot",
                    "intended_cognitive_effect",
                ):
                    value = items.get(key)
                    if value:
                        entry[key] = str(value).strip()
                if len(entry) == 3:
                    normalized.append(entry)
                return normalized
            if isinstance(items, tuple) and len(items) >= 3:
                entry = {
                    "trigger_context": str(items[0]).strip(),
                    "surprising_pivot": str(items[1]).strip(),
                    "intended_cognitive_effect": str(items[2]).strip(),
                }
                if all(entry.values()):
                    normalized.append(entry)
            elif isinstance(items, str):
                text = items.strip()
                if text:
                    normalized.append(
                        {
                            "trigger_context": text,
                            "surprising_pivot": f"Reframe the assumption by citing local research around {text}.",
                            "intended_cognitive_effect": "Spark curiosity instead of certainty about the intake narrative.",
                        }
                    )
            return normalized

        def flatten_profile(value: Any) -> Dict[str, Any]:
            if isinstance(value, ClientProfile):
                return value.model_dump()
            if isinstance(value, dict):
                flattened: Dict[str, Any] = {}
                for key, val in value.items():
                    if key in allowed_keys and val is not None:
                        if key == "prediction_errors":
                            flattened[key] = normalize_prediction_error_items(val)
                        else:
                            flattened[key] = val
                    elif isinstance(val, dict):
                        nested = flatten_profile(val)
                        for nested_key, nested_val in nested.items():
                            flattened.setdefault(nested_key, nested_val)
                    elif isinstance(val, list):
                        if key in allowed_keys:
                            if key == "prediction_errors":
                                flattened[key] = normalize_prediction_error_items(val)
                            else:
                                flattened[key] = [str(item).strip() for item in val if str(item).strip()]
                return flattened
            if isinstance(value, list):
                merged: Dict[str, Any] = {}
                for item in value:
                    nested = flatten_profile(item)
                    for nested_key, nested_val in nested.items():
                        merged.setdefault(nested_key, nested_val)
                return merged
            return {}

        profiles: List[Dict[str, Any]] = []
        if isinstance(raw_profiles, list):
            for item in raw_profiles:
                flattened = flatten_profile(item)
                if flattened:
                    profiles.append(flattened)
        elif isinstance(raw_profiles, dict):
            for key in sorted(raw_profiles.keys(), key=str):
                flattened = flatten_profile(raw_profiles[key])
                if flattened:
                    profiles.append(flattened)

        if not profiles:
            return None

        normalized_tags = [tag.strip() for tag in source_tags if isinstance(tag, str) and tag.strip()]

        def pick_tag(index: int) -> str:
            if normalized_tags:
                return normalized_tags[min(index, len(normalized_tags) - 1)]
            if citation_map:
                ordered = [str(tag) for tag in citation_map.keys()]
                if ordered:
                    return ordered[min(index, len(ordered) - 1)]
            return "[Source F1]"

        sanitized_profiles: List[Dict[str, Any]] = []
        for idx, profile in enumerate(profiles):
            tag = pick_tag(idx)
            profile.setdefault("fictional_disclaimer", self._fallback_disclaimer)
            if not profile.get("intake_reference"):
                profile["intake_reference"] = '"No direct intake quote provided"'
            if not profile.get("persona_overview"):
                profile["persona_overview"] = "Fictional scenario describing masking and systemic stressors in St. Louis."
            if not profile.get("research_reference"):
                snippet = citation_map.get(tag)
                summary = self._summarize_text_block(snippet, max_chars=160) if snippet else "Research insight"
                profile["research_reference"] = f"{summary} {tag}".strip()
            if not profile.get("benefit_explanation"):
                profile["benefit_explanation"] = f"{tag} clarifies how this research reframes the client's needs."
            if not profile.get("prediction_errors"):
                profile["prediction_errors"] = self._default_prediction_errors(tag)
            if profile.get("st_louis_alignment") and "[Source" not in str(profile["st_louis_alignment"]):
                profile["st_louis_alignment"] = f"{profile['st_louis_alignment']} {tag}".strip()
            if not profile.get("local_geography"):
                profile["local_geography"] = [
                    "Central West End",
                    "Tower Grove South",
                    "Delmar Loop",
                ]
            if not profile.get("community_connections"):
                profile["community_connections"] = [
                    "Carondelet YMCA",
                    "International Institute",
                    "Pageant Community Room",
                ]
            if not profile.get("regional_touchpoints"):
                profile["regional_touchpoints"] = profile["local_geography"][:3]
            if not profile.get("masking_signals"):
                profile["masking_signals"] = [
                    "Masking to sustain professional identity",
                    "Hypervigilance around social slip-ups",
                ]
            if not profile.get("unmet_needs"):
                profile["unmet_needs"] = [
                    "Predictable sensory decompression windows",
                    "Peer community normalising neurodivergent parenting",
                    "Trauma-informed professional mentorship",
                ]
            if not profile.get("support_recommendations"):
                profile["support_recommendations"] = [
                    "Map accessible co-working or library zones for deep work",
                    "Schedule restorative time anchored to neighbourhood assets",
                    "Translate research findings into scripts for family system conversations",
                ]
            if not profile.get("cautionary_flags"):
                profile["cautionary_flags"] = [
                    "Fictional scenario: do not infer outcomes",
                    "Escalate if client references acute crisis or harm",
                ]
            sanitized_profiles.append({key: profile.get(key) for key in allowed_keys if profile.get(key) is not None})

        normalized["profiles"] = sanitized_profiles

        raw_external = payload.get("external_sources")
        if raw_external:
            ext_list: List[Dict[str, Any]] = []
            iterable = raw_external if isinstance(raw_external, list) else list(raw_external.values())
            for item in iterable:
                if isinstance(item, ExternalResearchSource):
                    ext_list.append(item.model_dump())
                elif isinstance(item, dict):
                    entry: Dict[str, Any] = {}
                    for key in ("label", "title", "url", "publisher", "published_at", "summary", "verification_status"):
                        if key in item:
                            entry[key] = item[key]
                    if entry:
                        label = str(entry.get("label", "")).strip()
                        if "[Ext" not in label:
                            entry["label"] = f"[Ext {len(ext_list) + 1}]"
                        ext_list.append(entry)
            normalized["external_sources"] = ext_list

        return normalized

    def _collect_intake_quotes(
        self,
        client_insights: Dict[str, Any],
        raw_client_context: Optional[str],
    ) -> List[str]:
        quotes: List[str] = []

        def register(candidate: Optional[str]) -> None:
            if not candidate:
                return
            text = str(candidate).strip()
            if not text:
                return
            if text not in quotes:
                quotes.append(text)

        quote_keys: Sequence[str] = (
            "direct_quotes",
            "pain_point_quotes",
            "client_quotes",
            "verbatim_statements",
            "intake_quotes",
            "raw_segments",
        )
        for key in quote_keys:
            value = client_insights.get(key)
            if isinstance(value, str):
                register(value)
            elif isinstance(value, Iterable):
                for item in value:
                    if isinstance(item, dict):
                        register(item.get("quote") or item.get("text"))
                    else:
                        register(str(item))

        enhanced = client_insights.get("enhanced_analysis")
        if isinstance(enhanced, dict):
            for key in ("verbatim_quotes", "anchoring_quotes", "pain_points"):
                value = enhanced.get(key)
                if isinstance(value, str):
                    register(value)
                elif isinstance(value, Iterable):
                    for item in value:
                        if isinstance(item, dict):
                            register(item.get("quote") or item.get("text"))
                        else:
                            register(str(item))

        if raw_client_context:
            quoted = re.findall(r"[\"\“].+?[\"\”]", raw_client_context)
            if quoted:
                for match in quoted:
                    register(match)
            else:
                register(raw_client_context)

        return quotes

    def _render_intake_block(
        self, intake_quotes: List[str], raw_client_context: Optional[str]
    ) -> str:
        if intake_quotes:
            lines = []
            for quote in intake_quotes[:6]:
                trimmed = quote.strip()
                if not any(trimmed.startswith(mark) for mark in ('"', "“", "'")):
                    trimmed = f'"{trimmed}"'
                lines.append(f"- {trimmed}")
            return "\n".join(lines)

        if raw_client_context:
            sample = self._summarize_text_block(raw_client_context, max_chars=240)
            return f"- No verbatim quotes extracted. Intake narrative: \"{sample}\""

        return "- No intake language captured. Flag this gap in your JSON."

    def _fallback_profiles(
        self,
        *,
        intake_quotes: List[str],
        intake_segments: Sequence[Dict[str, Any]],
        retrieved_passages: Sequence[Dict[str, Any]],
        st_louis_context: Dict[str, Any],
        source_tags: Sequence[str] | None = None,
        citation_map: Dict[str, str] | None = None,
        client_insights: Dict[str, Any] | None = None,
        regional_atlas: Dict[str, Any] | None = None,
        document_localities: Sequence[Tuple[str, int]] | None = None,
        external_sources: Sequence[ExternalResearchSource] | None = None,
        document_id: str = "document",
    ) -> ClientProfileSet:
        """Deterministic fallback when LLM output is unavailable."""

        def pick_locations(count: int) -> List[str]:
            picks: List[str] = []
            locality_pairs = list(document_localities or [])
            locality_pairs.sort(key=lambda item: (-item[1], item[0]))
            for name, _ in locality_pairs:
                if len(picks) >= count:
                    break
                picks.append(name)
            atlas_candidates = []
            if regional_atlas:
                atlas_candidates.extend(regional_atlas.get("stl_city_neighborhoods", []))
                atlas_candidates.extend(regional_atlas.get("stl_county_municipalities", []))
                atlas_candidates.extend(regional_atlas.get("metro_east_communities", []))
            for name in atlas_candidates:
                if len(picks) >= count:
                    break
                if name not in picks:
                    picks.append(name)
            while len(picks) < count:
                picks.append(f"St. Louis locality #{len(picks) + 1}")
            return picks[:count]

        def choose_quote(index: int) -> str:
            if intake_quotes:
                quote = intake_quotes[index % len(intake_quotes)]
            elif intake_segments:
                quote = intake_segments[index % len(intake_segments)]["snippet"]
            else:
                quote = "No direct intake quote provided"
            quote = quote.strip()
            if not any(marker in quote for marker in ('"', "“", "”", "'")):
                quote = f'"{quote}"'
            return quote

        locations = pick_locations(3)
        atlas_neighbourhoods: List[str] = []
        if regional_atlas:
            atlas_neighbourhoods.extend(regional_atlas.get("stl_city_neighborhoods", [])[:10])
            atlas_neighbourhoods.extend(regional_atlas.get("stl_county_municipalities", [])[:10])
            atlas_neighbourhoods.extend(regional_atlas.get("metro_east_communities", [])[:10])
        document_locality_pairs = list(document_localities or [])
        profiles: List[ClientProfile] = []
        dataset_records = list(_locality_index().values())
        default_masking = [
            "High masking in professional settings",
            "Delayed interoceptive awareness",
            "Over-functioning to hide sensory overwhelm",
        ]
        default_unmet = [
            "Limited culturally safe peer support",
            "Fragmented executive-function scaffolding",
            "Sparse trauma-informed legal workplaces",
        ]
        default_support = [
            "Co-design weekly decompression rituals",
            "Resource mapping for neurodivergent-friendly community hubs",
            "Structured advocacy scripts to request workplace accommodations",
        ]
        default_flags = [
            "Do not infer clinical progress",
            "Escalate if acute risk language emerges",
        ]

        for idx in range(3):
            neighbourhood = locations[idx]
            quote = choose_quote(idx)
            if source_tags:
                primary_source = source_tags[idx % len(source_tags)]
                alt_source = source_tags[(idx + 1) % len(source_tags)]
            else:
                primary_source = "[Source F1]"
                alt_source = primary_source
            persona_name = f"{neighbourhood} fictional scenario"
            overview = (
                f"Fictional vignette following a resident of {neighbourhood} balancing masking, family systems, and professional pressure."
            )
            stl_alignment = (
                f"{primary_source} plus {neighbourhood} municipal data highlight how commute, school placement, and civic systems compound burnout."
            )
            seeded_localities = [(neighbourhood, 3 - idx)] + document_locality_pairs
            if dataset_records:
                offset = (idx * 3) % len(dataset_records)
                for j in range(min(3, len(dataset_records))):
                    record = dataset_records[(offset + j) % len(dataset_records)]
                    seeded_localities.append((record.name, 1))
            local_geography = self._ensure_local_geography(
                [neighbourhood],
                document_localities=seeded_localities,
                atlas_neighbourhoods=atlas_neighbourhoods,
                min_items=3,
                max_items=6,
            )
            community_connections = self._ensure_community_connections(
                [],
                local_geography=local_geography,
                min_items=3,
                max_items=10,
            )
            profiles.append(
                ClientProfile(
                    profile_name=persona_name,
                    fictional_disclaimer=self._fallback_disclaimer,
                    intake_reference=quote,
                    persona_overview=overview,
                    research_reference=f"{primary_source} evidence surfaces mechanisms relevant to this intake theme.",
                    benefit_explanation=f"{alt_source} maps interventions that resonate with this client's masking pattern.",
                    st_louis_alignment=stl_alignment,
                    local_geography=local_geography,
                    community_connections=community_connections,
                    regional_touchpoints=local_geography[:3],
                    masking_signals=list(default_masking),
                    unmet_needs=list(default_unmet),
                    support_recommendations=list(default_support),
                    cautionary_flags=list(default_flags),
                    prediction_errors=[
                        PredictionErrorEntry.model_validate(item)
                        for item in self._default_prediction_errors(primary_source, neighbourhood)
                    ],
                )
            )

        external_list = list(external_sources or [])
        return ClientProfileSet(
            profiles=profiles,
            shared_thread="Fictional personas demonstrating how regional systems intersect with neurodivergent masking",
            external_sources=external_list,
        )

    def _summarize_profile_metrics(
        self,
        profiles: ClientProfileSet,
        *,
        document_localities: Sequence[Tuple[str, int]] | None = None,
    ) -> Dict[str, Any]:
        """Build monitoring metrics for generated client profiles."""

        profile_list = list(getattr(profiles, "profiles", []) or [])
        external_sources = list(getattr(profiles, "external_sources", []) or [])

        unique_localities: List[str] = []
        unique_keys: Set[str] = set()
        prediction_errors_total = 0
        prediction_errors_by_profile: Dict[str, int] = {}
        disclaimer_failures: List[str] = []

        for profile in profile_list:
            localities = getattr(profile, "local_geography", []) or []
            for locality in localities:
                canonical = _canonicalize_label(locality)
                if canonical and canonical not in unique_keys:
                    unique_keys.add(canonical)
                    unique_localities.append(locality)

            errors = list(getattr(profile, "prediction_errors", []) or [])
            prediction_errors_total += len(errors)
            profile_name = getattr(profile, "profile_name", "profile")
            prediction_errors_by_profile[profile_name] = len(errors)

            disclaimer = str(getattr(profile, "fictional_disclaimer", "") or "")
            if "fictional" not in disclaimer.lower():
                disclaimer_failures.append(profile_name)

        external_citations_count = sum(
            1
            for source in external_sources
            if isinstance(getattr(source, "label", None), str)
            and source.label.strip().startswith("[Ext")
        )

        document_locality_labels = sorted(
            {
                str(name).strip()
                for name, _ in (document_localities or [])
                if str(name).strip()
            }
        )

        expected_prediction_errors = len(profile_list) * 2 if profile_list else 0
        alerts: List[str] = []
        if len(unique_keys) < max(3, len(profile_list)) and profile_list:
            alerts.append("Fewer than 3 unique localities referenced across profiles")
        if external_citations_count == 0 and profile_list:
            alerts.append("No external [Ext #] sources attached to profiles")
        if prediction_errors_total < expected_prediction_errors and profile_list:
            alerts.append("Prediction error count below 2 per profile expectation")
        if disclaimer_failures:
            alerts.append(
                "Fictional disclaimers missing or invalid for: "
                + ", ".join(disclaimer_failures)
            )

        if not profile_list:
            alerts.append("Client profiles missing from payload")

        metrics: Dict[str, Any] = {
            "profile_count": len(profile_list),
            "unique_localities_count": len(unique_keys),
            "unique_localities": unique_localities,
            "document_localities_detected": document_locality_labels,
            "external_citations_count": external_citations_count,
            "prediction_errors_total": prediction_errors_total,
            "prediction_errors_by_profile": prediction_errors_by_profile,
            "expected_prediction_errors": expected_prediction_errors,
            "disclaimer_status": "valid"
            if not disclaimer_failures and profile_list
            else ("invalid" if disclaimer_failures else "missing"),
            "disclaimer_failures": disclaimer_failures,
            "alerts": alerts,
        }
        if alerts:
            if any("disclaimer" in alert.lower() for alert in alerts):
                metrics["alert_level"] = "error"
            else:
                metrics["alert_level"] = "warning"
        else:
            metrics["alert_level"] = "ok"
        return metrics

    def _collect_source_tags(self, profiles: ClientProfileSet) -> List[str]:
        tags: List[str] = []
        pattern = re.compile(r"\[Source[^\]]+\]")
        for profile in profiles.profiles:
            for field in (profile.research_reference, profile.benefit_explanation, profile.st_louis_alignment or ""):
                tags.extend(pattern.findall(field))
            for entry in getattr(profile, "prediction_errors", []):
                if isinstance(entry, PredictionErrorEntry):
                    tags.extend(pattern.findall(entry.surprising_pivot))
                    tags.extend(pattern.findall(entry.intended_cognitive_effect))
        return tags

    def _default_prediction_errors(
        self,
        source_tag: str,
        locality: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        anchor = locality or "St. Louis"
        return [
            {
                "trigger_context": (
                    f"Assumes {anchor} caregivers only find support through hospital systems."
                ),
                "surprising_pivot": (
                    f"Cite {source_tag} plus library mutual-aid programs showing regulation rituals inside {anchor} third places."
                ),
                "intended_cognitive_effect": (
                    "Nudge the reader to scout overlooked civic spaces that already support neurodivergent nervous systems."
                ),
            },
            {
                "trigger_context": (
                    f"Believes every {anchor} commute drains sensory reserves beyond recovery."
                ),
                "surprising_pivot": (
                    f"Reference {source_tag} alongside local transit hacks—quiet blocks, river overlooks, or Metro rest stops—that invert that certainty."
                ),
                "intended_cognitive_effect": (
                    "Spark playful experimentation with micro-restorative pauses during the daily route."
                ),
            },
        ]

    def _select_relevant_intake_segments(
        self,
        *,
        intake_registry: Dict[str, Any],
        document_localities: Sequence[Tuple[str, int]],
        retrieved_passages: Sequence[Dict[str, Any]],
        fallback_quotes: Sequence[str],
        max_segments: int = 6,
    ) -> List[Dict[str, Any]]:
        entries = intake_registry.get("entries") if isinstance(intake_registry, dict) else None
        entries = entries or []
        locality_scores = {name.lower(): count for name, count in (document_localities or [])}
        scored: List[Dict[str, Any]] = []

        family_keywords = {
            "mother",
            "mom",
            "mum",
            "father",
            "dad",
            "parent",
            "guardian",
            "caregiver",
            "spouse",
            "husband",
            "wife",
            "partner",
            "grandmother",
            "grandfather",
            "grandparent",
            "aunt",
            "uncle",
            "cousin",
            "sibling",
            "brother",
            "sister",
            "son",
            "daughter",
        }
        employment_keywords = {
            "job",
            "work",
            "worker",
            "shift",
            "employer",
            "boss",
            "manager",
            "coworker",
            "coworkers",
            "workplace",
            "career",
            "overtime",
            "gig",
            "freelance",
            "teacher",
            "nurse",
            "retired",
            "unemployed",
            "laid off",
            "factory",
            "warehouse",
        }
        community_keywords = {
            "church",
            "library",
            "community center",
            "community centre",
            "community college",
            "community garden",
            "mutual aid",
            "food pantry",
            "food bank",
            "support group",
            "neighborhood council",
            "neighbourhood council",
            "pta",
            "ptsa",
            "temple",
            "mosque",
            "synagogue",
            "clinic",
            "county services",
            "rec center",
            "recreation center",
            "shelter",
            "nonprofit",
            "non-profit",
            "ymca",
        }
        hobby_keywords = {
            "hobby",
            "hobbies",
            "gaming",
            "video game",
            "music",
            "choir",
            "singing",
            "guitar",
            "piano",
            "running",
            "jogging",
            "cycling",
            "biking",
            "gardening",
            "garden",
            "cooking",
            "baking",
            "sports",
            "basketball",
            "soccer",
            "yoga",
            "dance",
            "dancing",
            "craft",
            "knitting",
            "reading",
            "book club",
            "writing",
            "poetry",
            "coding club",
            "art",
            "painting",
            "drawing",
        }
        emotional_keywords = {
            "anxious",
            "anxiety",
            "stressed",
            "stress",
            "overwhelmed",
            "burned out",
            "burnt out",
            "exhausted",
            "worried",
            "afraid",
            "fearful",
            "hopeful",
            "hope",
            "angry",
            "frustrated",
            "lonely",
            "isolated",
            "depressed",
            "sad",
            "relieved",
            "excited",
        }
        rural_keywords = {
            "rural",
            "farm",
            "farmland",
            "county road",
            "county line",
            "barn",
            "acre",
            "acres",
            "tractors",
            "tractor",
            "feed store",
            "grain elevator",
            "pasture",
            "small town",
            "village",
            "unincorporated",
            "agricultural",
            "homestead",
            "country life",
            "4-h",
            "co-op",
        }
        urban_keywords = {
            "downtown",
            "urban",
            "city",
            "metro",
            "subway",
            "light rail",
            "bus line",
            "transit",
            "loft",
            "apartment",
            "apartments",
            "high-rise",
            "condo",
            "streetcar",
            "sidewalk",
            "traffic",
            "siren",
            "skyscraper",
            "commuter train",
            "downtown loft",
        }

        def _contains_keyword(text: str, keywords: Sequence[str]) -> bool:
            for keyword in keywords:
                if " " in keyword or "-" in keyword:
                    if keyword in text:
                        return True
                else:
                    if re.search(rf"\b{re.escape(keyword)}\b", text):
                        return True
            return False

        for entry in entries:
            snippet = str(entry.get("snippet") or entry.get("text") or "").strip()
            if not snippet:
                continue
            themes = entry.get("themes") or []
            regional_mentions = entry.get("regional_mentions") or {}
            score = len(snippet) // 80 + len(themes)
            lowered_snippet = snippet.lower()
            lowered_themes = [str(theme).lower() for theme in themes if str(theme).strip()]
            combined_text = " ".join([lowered_snippet] + lowered_themes)
            coverage_tags: set[str] = set()

            if _contains_keyword(combined_text, family_keywords):
                score += 5
                coverage_tags.add("family_role")
            if _contains_keyword(combined_text, employment_keywords):
                score += 4
                coverage_tags.add("employment_role")
            if _contains_keyword(combined_text, community_keywords):
                score += 3
                coverage_tags.add("community_institution")
            if _contains_keyword(combined_text, hobby_keywords):
                score += 2
                coverage_tags.add("hobby_interest")
            if _contains_keyword(combined_text, emotional_keywords):
                score += 4
                coverage_tags.add("emotional_tone")

            has_rural_cue = _contains_keyword(combined_text, rural_keywords)
            has_urban_cue = _contains_keyword(combined_text, urban_keywords)
            if has_rural_cue:
                score += 3
                coverage_tags.add("rural_cue")
            if has_urban_cue:
                score += 3
                coverage_tags.add("urban_cue")

            for name, count in regional_mentions.items():
                score += int(count)
                if name and name.lower() in locality_scores:
                    score += 3
            payload = {
                "snippet": snippet,
                "themes": [str(t).strip() for t in themes if str(t).strip()],
                "regional_mentions": {str(k): int(v) for k, v in regional_mentions.items() if k},
                "coverage_tags": sorted(coverage_tags),
                "contextual_cues": {"rural": has_rural_cue, "urban": has_urban_cue},
            }
            scored.append({"score": score, "payload": payload, "entry_id": len(scored)})

        sorted_entries = sorted(scored, key=lambda item: item["score"], reverse=True)

        selected: List[Dict[str, Any]] = []
        selected_ids: set[int] = set()

        def _choose_best_with_tag(tag: str) -> None:
            if len(selected) >= max_segments:
                return
            for record in sorted_entries:
                if record["entry_id"] in selected_ids:
                    continue
                if tag in record["payload"].get("coverage_tags", []):
                    selected.append(record)
                    selected_ids.add(record["entry_id"])
                    return

        for locale_tag in ("rural_cue", "urban_cue"):
            _choose_best_with_tag(locale_tag)

        coverage_priority = (
            "family_role",
            "employment_role",
            "community_institution",
            "hobby_interest",
            "emotional_tone",
        )
        for tag in coverage_priority:
            _choose_best_with_tag(tag)

        for record in sorted_entries:
            if len(selected) >= max_segments:
                break
            if record["entry_id"] in selected_ids:
                continue
            selected.append(record)
            selected_ids.add(record["entry_id"])

        selected.sort(key=lambda item: item["score"], reverse=True)
        segments = [record["payload"] for record in selected[:max_segments]]

        if not segments:
            for quote in fallback_quotes:
                if len(segments) >= max_segments:
                    break
                cleaned = quote.strip()
                if cleaned:
                    segments.append(
                        {
                            "snippet": cleaned,
                            "themes": [],
                            "regional_mentions": {},
                            "coverage_tags": [],
                            "contextual_cues": {},
                        }
                    )

        if not segments and retrieved_passages:
            for passage in retrieved_passages[:max_segments]:
                text = self._summarize_text_block(passage.get("text"), max_chars=320)
                if text:
                    segments.append(
                        {
                            "snippet": text,
                            "themes": [],
                            "regional_mentions": {},
                            "coverage_tags": [],
                            "contextual_cues": {},
                        }
                    )

        return segments[:max_segments]

    def _render_intake_registry_block(self, segments: Sequence[Dict[str, Any]]) -> str:
        if not segments:
            return "- Intake analytics unavailable; note the gap and lean on retrieved passages."
        lines: List[str] = []
        for segment in segments[:6]:
            snippet = self._summarize_text_block(segment.get("snippet"), max_chars=220)
            themes = segment.get("themes") or []
            mentions = segment.get("regional_mentions") or {}
            theme_text = ", ".join(themes[:3]) if themes else "themes unclear"
            regional_text = ", ".join(list(mentions.keys())[:3]) if mentions else "no region tagged"
            lines.append(f"- \"{snippet}\" — themes: {theme_text}; regions: {regional_text}")
        return "\n".join(lines)

    def _render_regional_focus_block(
        self,
        *,
        document_localities: Sequence[Tuple[str, int]],
        intake_registry: Dict[str, Any],
        regional_atlas: Dict[str, Any],
        transcript_registry: Dict[str, Any],
    ) -> str:
        locality_lines: List[str] = []
        if document_localities:
            top = ", ".join(f"{name} (x{count})" for name, count in document_localities[:6])
            locality_lines.append(f"Document mentions heavily: {top}")
        top_locations = intake_registry.get("top_locations") if isinstance(intake_registry, dict) else None
        if top_locations:
            sample = ", ".join(name for name, _ in top_locations[:6])
            locality_lines.append(f"High-volume intake ZIPs: {sample}")
        transcript_themes = transcript_registry.get("top_themes") if isinstance(transcript_registry, dict) else None
        if transcript_themes:
            words = ", ".join(theme for theme, _ in transcript_themes[:6])
            locality_lines.append(f"Founder voice emphasises: {words}")
        atlas_neighbourhoods = regional_atlas.get("stl_city_neighborhoods", []) if isinstance(regional_atlas, dict) else []
        if atlas_neighbourhoods:
            locality_lines.append(
                "Atlas inventory (sample): " + ", ".join(atlas_neighbourhoods[:6])
            )
        if not locality_lines:
            return "- Regional atlas not loaded; cite any location explicitly mentioned in the PDF."
        return "\n".join(f"- {line}" for line in locality_lines)

    def _render_health_priority_block(
        self,
        *,
        health_summary: Dict[str, Any],
        document_localities: Sequence[Tuple[str, int]],
    ) -> str:
        if not health_summary:
            return "- No health report loaded; acknowledge data gap."
        lines: List[str] = []
        priority_signals = health_summary.get("priority_signals") or []
        if priority_signals:
            highlights = ", ".join(label for label, _ in priority_signals[:5])
            lines.append(f"Priority signals: {highlights}")
        regional_mentions = health_summary.get("regional_mentions") or {}
        if regional_mentions:
            ordered = sorted(regional_mentions.items(), key=lambda item: (-item[1], item[0]))
            focus = ", ".join(name for name, _ in ordered[:5])
            lines.append(f"Health report hotspots: {focus}")
        if not lines:
            return "- Health report parsed but yielded no high-signal sections."
        return "\n".join(f"- {line}" for line in lines)

    def _render_research_focus_block(
        self,
        retrieved_passages: Sequence[Dict[str, Any]],
        *,
        max_passages: int = 5,
    ) -> str:
        if not retrieved_passages:
            return "- No retrieved passages; rely on external research and intake analysis."
        lines: List[str] = []
        for idx, passage in enumerate(retrieved_passages[:max_passages], start=1):
            tag = passage.get("tag") or f"[Source {idx}]"
            snippet = self._summarize_text_block(passage.get("text"), max_chars=200)
            origin = passage.get("document_id") or passage.get("metadata", {}).get("document_id")
            doc_part = f" (doc={origin})" if origin else ""
            lines.append(f"- {tag} {snippet}{doc_part}")
        return "\n".join(lines)

    async def _gather_external_research(
        self,
        *,
        context: Dict[str, Any],
        intake_segments: Sequence[Dict[str, Any]],
        document_localities: Sequence[Tuple[str, int]],
    ) -> Dict[str, Any]:
        if not self.research_orchestrator:
            return {
                "sources": [],
                "prompt_block": "- External research connectors not configured. Cite intake and PDF sources only.",
            }

        def _format_locality_name(raw: Optional[str]) -> Optional[str]:
            if not raw:
                return None
            cleaned = raw.strip()
            if not cleaned:
                return None
            lowered = cleaned.lower()
            if any(token in lowered for token in (" missouri", " mo", " illinois", " il")):
                return cleaned
            if "metro east" in lowered:
                return f"{cleaned} Illinois"
            return f"{cleaned} Missouri"

        def _extract_locality_from_tags(tags: Any) -> Optional[str]:
            if not tags:
                return None
            first = tags[0] if isinstance(tags, (list, tuple)) and tags else None
            if isinstance(first, dict):
                candidate = first.get("name") or first.get("locality") or first.get("label")
                return _format_locality_name(candidate)
            if isinstance(first, (list, tuple)) and first:
                return _format_locality_name(str(first[0]))
            if isinstance(first, str):
                return _format_locality_name(first)
            return None

        queries: List[ResearchQuery] = []
        default_location = document_localities[0][0] if document_localities else "St. Louis"
        theme_gaps = context.get("theme_gaps") or []
        locality_records = context.get("locality_records") or {}
        locality_gap_entries = []
        if isinstance(locality_records, dict):
            raw_gaps = locality_records.get("gaps") or []
            if isinstance(raw_gaps, list):
                locality_gap_entries = [entry for entry in raw_gaps if isinstance(entry, dict)]

        for theme_gap in theme_gaps[:4]:
            if not isinstance(theme_gap, dict):
                continue
            theme_name = theme_gap.get("theme")
            if not theme_name:
                continue
            locality_from_tags = _extract_locality_from_tags(theme_gap.get("locality_tags"))
            locality = locality_from_tags or _format_locality_name(default_location)
            if not locality:
                locality = "St. Louis Missouri"
            topic = f"{locality} {theme_name.lower()} support networks"
            queries.append(
                ResearchQuery(
                    topic=topic,
                    focus="theme_gap",
                    location=locality,
                    tags=["client_profile", "theme_gap"],
                    max_results=5,
                )
            )
            log_with_telemetry(
                logger.info,
                "Queued external research query for theme gap: %s",
                topic,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MINOR,
                impact="Enriched research coverage for theme gap",
                doc_id=context.get("document_id"),
                details={
                    "topic": topic,
                    "query_reason": "theme_gap",
                    "missing_sources": theme_gap.get("missing_sources"),
                },
            )

        for gap in locality_gap_entries[:4]:
            locality_name = gap.get("name")
            if not locality_name:
                continue
            formatted_locality = _format_locality_name(locality_name) or locality_name
            leading_theme = theme_gaps[0].get("theme") if theme_gaps and isinstance(theme_gaps[0], dict) else None
            if isinstance(leading_theme, str) and leading_theme.strip():
                descriptor = leading_theme.lower()
                suffix = "support networks"
            else:
                descriptor = "community resource"
                suffix = "directories"
            topic = f"{formatted_locality} {descriptor} {suffix}"
            queries.append(
                ResearchQuery(
                    topic=topic,
                    focus="locality_gap",
                    location=formatted_locality,
                    tags=["client_profile", "locality_gap"],
                    max_results=5,
                )
            )
            log_with_telemetry(
                logger.info,
                "Queued external research query for locality gap: %s",
                topic,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MINOR,
                impact="Enriched research coverage for locality gap",
                doc_id=context.get("document_id"),
                details={
                    "topic": topic,
                    "query_reason": "locality_gap",
                    "signal_strength": gap.get("signal_strength"),
                },
            )

        for name, _ in document_localities[:3]:
            formatted = _format_locality_name(name) or name
            queries.append(
                ResearchQuery(
                    topic=f"neurodivergent burnout supports in {formatted}",
                    focus="regional_context",
                    location=formatted,
                    tags=["client_profile", "regional"],
                    max_results=5,
                )
            )

        for segment in intake_segments[:3]:
            snippet = segment.get("snippet", "") if isinstance(segment, dict) else ""
            themes = ", ".join(segment.get("themes", [])[:3]) if isinstance(segment, dict) else ""
            if snippet:
                queries.append(
                    ResearchQuery(
                        topic=f"community resources for: {themes or snippet[:80]}",
                        focus="intake_theme",
                        location=_format_locality_name(default_location) or default_location,
                        tags=["client_profile", "theme"],
                        max_results=5,
                    )
                )

        if not queries:
            default_locality = _format_locality_name(default_location) or "St. Louis Missouri"
            queries.append(
                ResearchQuery(
                    topic=f"{default_locality} neurodivergent adult support networks",
                    focus="regional_context",
                    location=default_locality,
                    tags=["fallback"],
                    max_results=6,
                )
            )

        try:
            hits: List[ResearchHit] = await self.research_orchestrator.gather(queries)
        except Exception as exc:
            log_with_telemetry(
                logger.warning,
                "External research orchestrator failed: %s",
                exc,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MINOR,
                impact="External research unreachable",
                doc_id=context.get("document_id"),
            )
            return {
                "sources": [],
                "prompt_block": "- External research service unavailable; rely on PDF citations.",
            }

        combined_hits: List[ResearchHit] = list(hits)
        minimum_expected_hits = min(len(queries), 3)
        if len(combined_hits) < minimum_expected_hits:
            fallback_queries: List[ResearchQuery] = []
            if theme_gaps:
                for theme_gap in theme_gaps[:2]:
                    if not isinstance(theme_gap, dict):
                        continue
                    theme_name = theme_gap.get("theme")
                    if not theme_name:
                        continue
                    fallback_queries.append(
                        ResearchQuery(
                            topic=f"Missouri public data on {theme_name.lower()} programs",
                            focus="retry_broad",
                            location="Missouri",
                            tags=["client_profile", "retry", "theme_gap"],
                            max_results=6,
                        )
                    )
            if not fallback_queries:
                default_locality = _format_locality_name(default_location) or "St. Louis Missouri"
                fallback_queries.append(
                    ResearchQuery(
                        topic=f"{default_locality} civic resource directories",
                        focus="retry_broad",
                        location=default_locality,
                        tags=["client_profile", "retry", "fallback"],
                        max_results=6,
                    )
                )

            log_with_telemetry(
                logger.info,
                "Retrying external research due to sparse initial hits (found=%s, expected=%s)",
                len(combined_hits),
                minimum_expected_hits,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MINOR,
                impact="Retrying external research with broader connectors",
                doc_id=context.get("document_id"),
                details={
                    "initial_hits": len(combined_hits),
                    "expected": minimum_expected_hits,
                    "retry_queries": [query.topic for query in fallback_queries],
                },
            )

            try:
                retry_hits = await self.research_orchestrator.gather(fallback_queries)
            except Exception as exc:  # pragma: no cover - defensive logging
                log_with_telemetry(
                    logger.warning,
                    "External research retry failed: %s",
                    exc,
                    agent=TELEMETRY_AGENT,
                    severity=TelemetrySeverity.MAJOR,
                    impact="External research retry failed",
                    doc_id=context.get("document_id"),
                )
                retry_hits = []

            if retry_hits:
                existing_urls = {hit.url for hit in combined_hits}
                for item in retry_hits:
                    if item.url not in existing_urls:
                        combined_hits.append(item)
                        existing_urls.add(item.url)

        if not combined_hits:
            log_with_telemetry(
                logger.warning,
                "External research remained empty after retries",
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="External research gaps persist",
                doc_id=context.get("document_id"),
                details={
                    "queries_attempted": [query.topic for query in queries],
                },
            )
            return {
                "sources": [],
                "prompt_block": "- No verified external sources surfaced; note this if context is thin.",
            }

        sources: List[ExternalResearchSource] = []
        lines: List[str] = []
        for idx, hit in enumerate(combined_hits, start=1):
            label = f"[Ext {idx}]"
            summary = self._summarize_text_block(hit.snippet or hit.summary, max_chars=220)
            verification = hit.verification_status if hasattr(hit, "verification_status") else "verified"
            source = ExternalResearchSource(
                label=label,
                title=hit.title or "External research insight",
                url=hit.url,
                publisher=hit.publisher,
                published_at=hit.published_at,
                summary=summary or "Summary unavailable",
                verification_status=verification,
            )
            sources.append(source)
            lines.append(f"- {label} {source.title} — {source.summary}")

        log_with_telemetry(
            logger.info,
            "Harvested external research hits",
            "",
            agent=TELEMETRY_AGENT,
            severity=TelemetrySeverity.MINOR,
            impact="External research appended to client profiles",
            doc_id=context.get("document_id"),
            details={"total_hits": len(sources)},
        )

        return {"sources": sources, "prompt_block": "\n".join(lines)}

    def _post_process_profiles(
        self,
        *,
        response: ClientProfileSet,
        document_id: str,
        document_localities: Sequence[Tuple[str, int]],
        external_sources: Sequence[ExternalResearchSource],
        regional_atlas: Optional[Dict[str, Any]] = None,
    ) -> ClientProfileSet:
        profiles = list(response.profiles)
        self._ensure_unique_profile_names(profiles, document_id)

        locality_names = [name for name, _ in document_localities]
        atlas_neighbourhoods = []
        if isinstance(regional_atlas, dict):
            atlas_neighbourhoods.extend(regional_atlas.get("stl_city_neighborhoods", [])[:10])
            atlas_neighbourhoods.extend(regional_atlas.get("stl_county_municipalities", [])[:10])

        for profile in profiles:
            profile.fictional_disclaimer = self._ensure_disclaimer(profile.fictional_disclaimer)
            profile.persona_overview = self._strip_outcome_language(profile.persona_overview)
            profile.research_reference = self._strip_outcome_language(profile.research_reference)
            profile.benefit_explanation = self._strip_outcome_language(profile.benefit_explanation)
            if profile.st_louis_alignment:
                profile.st_louis_alignment = self._strip_outcome_language(profile.st_louis_alignment)

            profile.local_geography = self._ensure_local_geography(
                getattr(profile, "local_geography", []),
                document_localities=document_localities,
                atlas_neighbourhoods=atlas_neighbourhoods,
                min_items=3,
                max_items=6,
            )
            profile.community_connections = self._ensure_community_connections(
                getattr(profile, "community_connections", []),
                local_geography=profile.local_geography,
                min_items=3,
                max_items=10,
            )
            profile.regional_touchpoints = self._ensure_minimum_items(
                profile.regional_touchpoints,
                fallback=(
                    profile.local_geography[:3]
                    + locality_names[:1]
                    + atlas_neighbourhoods[:4]
                    + ["Local sensory-friendly café"]
                ),
                min_items=3,
                max_items=8,
            )
            profile.masking_signals = self._ensure_minimum_items(
                profile.masking_signals,
                fallback=[
                    "Masking to sustain professional identity",
                    "Hypervigilance around social slip-ups",
                    "Compassion fatigue after caretaking",
                ],
                min_items=2,
                max_items=8,
            )
            profile.unmet_needs = self._ensure_minimum_items(
                profile.unmet_needs,
                fallback=[
                    "Predictable sensory decompression windows",
                    "Peer community normalising neurodivergent parenting",
                    "Trauma-informed professional mentorship",
                ],
                min_items=3,
                max_items=8,
            )
            profile.support_recommendations = self._ensure_minimum_items(
                profile.support_recommendations,
                fallback=[
                    "Map accessible co-working or library zones for deep work",
                    "Schedule restorative time anchored to neighbourhood assets",
                    "Translate research findings into scripts for family system conversations",
                ],
                min_items=3,
                max_items=8,
            )
            profile.cautionary_flags = self._ensure_minimum_items(
                profile.cautionary_flags,
                fallback=[
                    "Fictional scenario: do not infer outcomes",
                    "Escalate if client references acute crisis or harm",
                ],
                min_items=2,
                max_items=6,
            )
            primary_source = self._extract_first_source_tag(
                profile.research_reference,
                profile.benefit_explanation,
                profile.st_louis_alignment,
            )
            locality_hint = profile.local_geography[0] if profile.local_geography else None
            if not locality_hint and profile.regional_touchpoints:
                locality_hint = profile.regional_touchpoints[0]
            elif not locality_hint and locality_names:
                locality_hint = locality_names[0]
            profile.prediction_errors = self._ensure_prediction_errors(
                profile.prediction_errors,
                default_source=primary_source,
                locality_hint=locality_hint,
            )

        merged_sources = self._merge_external_sources(response.external_sources, external_sources)
        response.profiles = profiles
        response.external_sources = merged_sources
        return ClientProfileSet.model_validate(response.model_dump())

    def _ensure_unique_profile_names(self, profiles: Sequence[ClientProfile], document_id: str) -> None:
        seen: set[str] = set()
        for index, profile in enumerate(profiles, start=1):
            raw_name = getattr(profile, "profile_name", "")
            base = re.sub(r"[^A-Za-z0-9\s]+", "", str(raw_name or "")).strip()
            if not base:
                base = f"Scenario {index}"
            candidate = base
            while candidate.lower() in seen:
                suffix = hashlib.sha1(f"{document_id}:{base}:{len(seen)}".encode()).hexdigest()[:6]
                candidate = f"{base} · {suffix}"
            profile.profile_name = candidate
            seen.add(candidate.lower())

    def _ensure_disclaimer(self, value: Optional[str]) -> str:
        text = str(value or "").strip()
        if text and "fictional" in text.lower():
            return text
        return self._fallback_disclaimer

    def _strip_outcome_language(self, text: Optional[str]) -> str:
        if not text:
            return ""
        cleaned = str(text)
        for phrase in getattr(ClientProfile, "_OUTCOME_BLOCKLIST", ()):  # type: ignore[attr-defined]
            cleaned = re.sub(re.escape(phrase), "support focus", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _ensure_minimum_items(
        self,
        items: Optional[Sequence[str]],
        *,
        fallback: Sequence[str],
        min_items: int,
        max_items: int,
    ) -> List[str]:
        unique: List[str] = []
        seen: set[str] = set()
        for entry in items or []:
            cleaned = self._strip_outcome_language(entry)
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered not in seen:
                seen.add(lowered)
                unique.append(cleaned)
        for entry in fallback:
            if len(unique) >= min_items:
                break
            cleaned = self._strip_outcome_language(entry)
            lowered = cleaned.lower()
            if lowered not in seen and cleaned:
                seen.add(lowered)
                unique.append(cleaned)
        if not unique:
            unique = [self._strip_outcome_language(entry) for entry in fallback if entry][:min_items]
        return unique[:max_items]

    def _ensure_local_geography(
        self,
        items: Optional[Sequence[str]],
        *,
        document_localities: Sequence[Tuple[str, int]],
        atlas_neighbourhoods: Sequence[str],
        min_items: int,
        max_items: int,
    ) -> List[str]:
        unique: List[str] = []
        seen: Set[str] = set()
        for entry in items or []:
            record = _match_locality(entry)
            if not record:
                continue
            canonical = _canonicalize_label(record.name)
            if canonical in seen:
                continue
            seen.add(canonical)
            unique.append(record.name)

        fallback_candidates: List[str] = []
        for name, _count in document_localities:
            record = _match_locality(name)
            if record:
                fallback_candidates.append(record.name)
        for name in atlas_neighbourhoods:
            record = _match_locality(name)
            if record:
                fallback_candidates.append(record.name)
        for record in _locality_index().values():
            fallback_candidates.append(record.name)

        for candidate in fallback_candidates:
            if len(unique) >= min_items:
                break
            canonical = _canonicalize_label(candidate)
            if canonical in seen:
                continue
            seen.add(canonical)
            unique.append(candidate)

        if len(unique) < min_items:
            for entry in items or []:
                canonical = _canonicalize_label(entry)
                if canonical in seen:
                    continue
                seen.add(canonical)
                unique.append(str(entry))
                if len(unique) >= min_items:
                    break

        if len(unique) < min_items:
            defaults = ["Central West End", "Tower Grove South", "Delmar Loop"]
            for candidate in defaults:
                if len(unique) >= min_items:
                    break
                canonical = _canonicalize_label(candidate)
                if canonical in seen:
                    continue
                seen.add(canonical)
                unique.append(candidate)

        return unique[:max_items]

    def _ensure_community_connections(
        self,
        items: Optional[Sequence[str]],
        *,
        local_geography: Sequence[str],
        min_items: int,
        max_items: int,
    ) -> List[str]:
        locality_records: List[LocalityRecord] = []
        locality_keys: Set[str] = set()
        for locality in local_geography:
            record = _match_locality(locality)
            if record:
                locality_records.append(record)
                locality_keys.add(_canonicalize_label(record.name))

        allowed_assets = _assets_for_records(locality_records)
        global_assets = _global_asset_index()

        unique: List[str] = []
        seen: Set[str] = set()
        for entry in items or []:
            canonical = _canonicalize_label(entry)
            label = None
            if canonical in allowed_assets:
                label = allowed_assets[canonical]
            else:
                asset_entry = global_assets.get(canonical)
                if asset_entry and asset_entry["localities"] & locality_keys:
                    label = str(asset_entry["label"])
            if not label or canonical in seen:
                continue
            seen.add(canonical)
            unique.append(label)

        fallback_pool: List[str] = []
        for record in locality_records:
            fallback_pool.extend(
                record.community_centers
                + record.health_resources
                + record.landmark_schools
                + record.signature_eateries
                + record.youth_sports_leagues
            )

        for asset in fallback_pool:
            if len(unique) >= min_items:
                break
            canonical = _canonicalize_label(asset)
            if canonical in seen:
                continue
            seen.add(canonical)
            unique.append(asset)

        if len(unique) < min_items:
            for asset_entry in global_assets.values():
                if len(unique) >= min_items:
                    break
                canonical = _canonicalize_label(asset_entry["label"])
                if canonical in seen:
                    continue
                seen.add(canonical)
                unique.append(str(asset_entry["label"]))

        return unique[:max_items]

    def _ensure_prediction_errors(
        self,
        entries: Sequence[PredictionErrorEntry],
        *,
        default_source: str,
        locality_hint: Optional[str] = None,
    ) -> List[PredictionErrorEntry]:
        normalized: List[PredictionErrorEntry] = []
        seen: set[tuple[str, str]] = set()
        for entry in entries or []:
            try:
                validated = entry if isinstance(entry, PredictionErrorEntry) else PredictionErrorEntry.model_validate(entry)
            except Exception:
                continue
            key = (
                validated.trigger_context.strip().lower(),
                validated.surprising_pivot.strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            normalized.append(validated)

        fallback_templates = self._default_prediction_errors(default_source, locality_hint)
        for template in fallback_templates:
            if len(normalized) >= 2:
                break
            try:
                validated = PredictionErrorEntry.model_validate(template)
            except Exception:
                continue
            key = (
                validated.trigger_context.strip().lower(),
                validated.surprising_pivot.strip().lower(),
            )
            if key not in seen:
                seen.add(key)
                normalized.append(validated)
        return normalized[:5]

    def _merge_external_sources(
        self,
        existing: Sequence[ExternalResearchSource],
        extras: Sequence[ExternalResearchSource],
    ) -> List[ExternalResearchSource]:
        merged: List[ExternalResearchSource] = []
        seen: set[str] = set()
        for source in existing or []:
            label = str(getattr(source, "label", "")).strip()
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(source)
        for source in extras or []:
            label = str(getattr(source, "label", "")).strip()
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(source)
        max_items = ClientProfileSet.model_fields["external_sources"].json_schema_extra.get("maxItems")
        if max_items and len(merged) > max_items:
            merged = merged[:max_items]
        return merged

    def _extract_first_source_tag(self, *fields: Optional[str]) -> str:
        pattern = re.compile(r"\[Source[^\]]+\]")
        for field in fields:
            if not field:
                continue
            match = pattern.search(str(field))
            if match:
                return match.group(0)
        return "[Source F1]"
