"""Client Profile Agent - Links intake language to retrieved research citations."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import ValidationError

from .base_agent import BaseAgent
from src.models.enlitens_schemas import ClientProfile, ClientProfileSet
from src.synthesis.ollama_client import OllamaClient
from src.utils.enlitens_constitution import EnlitensConstitution

logger = logging.getLogger(__name__)


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

    async def initialize(self) -> bool:
        try:
            self.ollama_client = OllamaClient(default_model=self.model)
            if not await self.ollama_client.check_connection():
                raise RuntimeError(
                    "Language model backend unavailable for ClientProfileAgent."
                )
            self.is_initialized = True
            logger.info("✅ %s agent initialized", self.name)
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to initialize %s: %s", self.name, exc)
            return False

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if self.ollama_client is None:
                raise RuntimeError("ClientProfileAgent not initialized")

            retrieved_passages = context.get("retrieved_passages") or []
            raw_client_context = context.get("raw_client_context")
            client_insights = context.get("client_insights") or {}
            st_louis_context = context.get("st_louis_context") or {}

            intake_quotes = self._collect_intake_quotes(client_insights, raw_client_context)
            intake_block = self._render_intake_block(intake_quotes, raw_client_context)
            retrieved_block = self._render_retrieved_passages_block(
                retrieved_passages,
                raw_client_context=raw_client_context,
                raw_founder_context=context.get("raw_founder_context"),
                max_passages=6,
            )

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
                "      \"intake_reference\": str,\n"
                "      \"research_reference\": str,\n"
                "      \"benefit_explanation\": str,\n"
                "      \"st_louis_alignment\": str | null\n"
                "    } * 3\n"
                "  ],\n"
                "  \"shared_thread\": str | null\n"
                "}"
            )

            prompt = f"""
You are the Enlitens Client Profile Agent. Your mandate is to anchor neuroscience research to the exact words clients share during intake, proving why the study matters for them in St. Louis.

{constitution_block}

INTAKE LANGUAGE TO HONOUR (reuse phrases verbatim inside quotes):
{intake_block}

ST. LOUIS CONTEXT SNAPSHOT:
{st_louis_block}

RETRIEVED PASSAGES WITH SOURCE TAGS:
{retrieved_block}

OUTPUT REQUIREMENTS:
1. Produce exactly THREE distinct profiles in JSON.
2. Each "intake_reference" must reuse the exact client phrasing inside quotes. If no quote exists, say "No direct intake quote provided" and flag the gap.
3. "research_reference" AND "benefit_explanation" must each cite at least one [Source #] tag from the passages above.
4. Use "st_louis_alignment" to connect the research to St. Louis realities (neighbourhood stressors, transit, economics). If data is missing, state that explicitly while still citing [Source #].
5. Highlight why each profile benefits from the paper – make it concrete and rebellious, not generic sympathy.
6. Return JSON exactly in this shape (no commentary):
{schema_hint}
"""

            cache_kwargs = self._cache_kwargs(context, suffix="profiles")
            response = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=ClientProfileSet,
                temperature=0.15,
                max_retries=3,
                enforce_grammar=True,
                **cache_kwargs,
            )

            if response is None:
                logger.warning("⚠️ ClientProfileAgent falling back to deterministic profiles")
                response = self._fallback_profiles(intake_quotes, retrieved_passages, st_louis_context)

            payload = response.model_dump()
            source_tags = self._collect_source_tags(response)
            return {
                "client_profiles": payload,
                "client_profile_summary": {
                    "intake_quotes_used": [p.intake_reference for p in response.profiles],
                    "source_tags": sorted(source_tags),
                    "shared_thread": response.shared_thread,
                },
            }
        except Exception as exc:
            logger.error("ClientProfileAgent failed: %s", exc)
            fallback = self._fallback_profiles([], context.get("retrieved_passages") or [], context.get("st_louis_context") or {})
            return {
                "client_profiles": fallback.model_dump(),
                "client_profile_summary": {
                    "intake_quotes_used": [p.intake_reference for p in fallback.profiles],
                    "source_tags": sorted(self._collect_source_tags(fallback)),
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

        for profile in profiles.profiles:
            if "[Source" not in profile.research_reference:
                raise ValueError("Each research_reference must cite a [Source #]")
            if "[Source" not in profile.benefit_explanation:
                raise ValueError("Each benefit_explanation must cite a [Source #]")
            if not any(marker in profile.intake_reference for marker in ('"', "'", "“", "”")):
                raise ValueError("Intake references must include quoted client language")
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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
        intake_quotes: List[str],
        retrieved_passages: Sequence[Dict[str, Any]],
        st_louis_context: Dict[str, Any],
    ) -> ClientProfileSet:
        quote = intake_quotes[0] if intake_quotes else "No direct intake quote provided"
        if not any(marker in quote for marker in ('"', "“", "”", "'")):
            quote = f'"{quote}"'
        source_tag = "[Source 1]" if retrieved_passages else "[Source F1]"
        stl_phrase = (
            "St. Louis community data" if st_louis_context else "St. Louis context not documented"
        )
        templates = [
            "Transit sensory overload",
            "Workplace autonomic crash",
            "Neighborhood hypervigilance",
        ]
        profiles: List[ClientProfile] = []
        for name in templates:
            profiles.append(
                ClientProfile(
                    profile_name=name,
                    intake_reference=quote,
                    research_reference=f"{source_tag} evidence describes the stressors this client flagged.",
                    benefit_explanation=f"{source_tag} shows targeted supports that ease their exact trigger.",
                    st_louis_alignment=f"{source_tag} plus {stl_phrase} call for localized action.",
                )
            )
        return ClientProfileSet(profiles=profiles, shared_thread="Research-driven validation for intake pain points")

    def _collect_source_tags(self, profiles: ClientProfileSet) -> List[str]:
        tags: List[str] = []
        pattern = re.compile(r"\[Source[^\]]+\]")
        for profile in profiles.profiles:
            for field in (profile.research_reference, profile.benefit_explanation, profile.st_louis_alignment or ""):
                tags.extend(pattern.findall(field))
        return tags
