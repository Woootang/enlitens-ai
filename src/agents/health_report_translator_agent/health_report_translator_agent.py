"""
Translate the long-form St. Louis health report into a Liz-voiced structured digest.

This agent provides the deterministic "data sheet" the rest of the pipeline can
rely on.  It is intentionally separate from the health report synthesizer:
the synthesizer creates document-specific briefs, whereas the translator
produces a reusable, uniform summary of the source report itself.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.utils.llm_response_cleaner import extract_json_object, strip_reasoning_artifacts

from src.models.health_report_digest import (
    CulturalFlashpoint,
    HealthReportDigest,
    NeighborhoodSnapshot,
    ResourceHighlight,
)

logger = logging.getLogger(__name__)

# Rough heuristics for shortening the report before sending to the LLM.
_SECTION_SPLIT_RE = re.compile(r"\n_{3,}\n")


class HealthReportTranslatorAgent:
    """
    Rewrite the St. Louis health report into a structured digest with Liz's voice.

    The agent:
    * Builds an instruction prompt grounded in the raw report plus optional
      language/topic guardrails.
    * Calls the shared local LLM (via the provided client) to generate
      a JSON structure.
    * Validates and normalises the JSON into the HealthReportDigest schema,
      falling back to a deterministic summary if parsing fails.
    """

    def __init__(self, *, max_prompt_chars: int = 12000, version: str = "1.0") -> None:
        self.max_prompt_chars = max_prompt_chars
        self.version = version

    async def translate(
        self,
        *,
        report_text: str,
        llm_client: Any,
        language_profile: Optional[Dict[str, Any]] = None,
        topic_alignment: Optional[Dict[str, Any]] = None,
        data_profile: Optional[Dict[str, Any]] = None,
    ) -> HealthReportDigest:
        """
        Produce a structured Liz-voiced digest for the St. Louis health report.

        Args:
            report_text: Raw text of the health report (already extracted).
            llm_client: Shared local LLM client (e.g., Qwen 2.5 via vLLM).
            language_profile: Optional guardrails from AudienceLanguageProfiler.
            topic_alignment: Optional alignment profile to emphasise bridges.
            data_profile: Optional deterministic stats from DataProfiler.

        Returns:
            HealthReportDigest object (never None; falls back to deterministic summary).
        """
        digest_id = f"stl-health-digest-{datetime.utcnow():%Y%m%d%H%M%S}"
        source_metadata = self._build_source_metadata(report_text, data_profile)

        if not report_text or len(report_text.strip()) < 200:
            logger.warning("HealthReportTranslator: report text missing or too short; using fallback digest.")
            return self._fallback_digest(
                digest_id=digest_id,
                source_metadata=source_metadata,
                language_profile=language_profile,
            )

        prompt = self._build_prompt(
            report_text=report_text,
            language_profile=language_profile,
            topic_alignment=topic_alignment,
            data_profile=data_profile,
        )

        try:
            logger.info(
                "🧠 HealthReportTranslator: requesting digest from LLM (prompt %d chars)",
                len(prompt),
            )
            generation = await llm_client.generate_response(
                prompt=prompt,
                temperature=0.15,
                num_predict=1600,
                response_format="json_object",
            )
            response = generation.get("response", "")
        except Exception as exc:  # pragma: no cover - external dependency
            logger.error("HealthReportTranslator: LLM call failed: %s", exc)
            return self._fallback_digest(
                digest_id=digest_id,
                source_metadata=source_metadata,
                language_profile=language_profile,
            )

        parsed_payload = self._parse_response(response)

        if parsed_payload is None:
            logger.warning("HealthReportTranslator: falling back due to parse failure.")
            return self._fallback_digest(
                digest_id=digest_id,
                source_metadata=source_metadata,
                language_profile=language_profile,
            )

        digest = self._normalise_payload(
            payload=parsed_payload,
            digest_id=digest_id,
            source_metadata=source_metadata,
        )
        digest.version = self.version
        digest.prompt_block = self._compose_prompt_block(digest, language_profile)

        logger.info(
            "🧠 HealthReportTranslator: digest ready with %d flashpoints, %d neighborhoods.",
            len(digest.cultural_flashpoints),
            len(digest.neighborhood_snapshots),
        )
        return digest

    # --------------------------------------------------------------------- #
    # Prompt construction
    # --------------------------------------------------------------------- #

    def _build_prompt(
        self,
        *,
        report_text: str,
        language_profile: Optional[Dict[str, Any]],
        topic_alignment: Optional[Dict[str, Any]],
        data_profile: Optional[Dict[str, Any]],
    ) -> str:
        """
        Construct the instruction prompt for the LLM.
        """
        trimmed_report = self._trim_report(report_text)

        language_block = ""
        if language_profile:
            snippet = language_profile.get("prompt_block")
            banned_terms = language_profile.get("banned_terms") or []
            if snippet:
                language_block = f"""
CLIENT LANGUAGE GUARDRAILS:
{snippet}
"""
            if banned_terms:
                language_block += (
                    "\nABSOLUTELY AVOID THESE WORDS UNLESS THEY APPEAR IN THE SOURCE: "
                    + ", ".join(sorted(set(banned_terms)))
                    + "\n"
                )

        alignment_block = ""
        if topic_alignment:
            note = topic_alignment.get("alignment_note") or topic_alignment.get("explanation")
            confidence = topic_alignment.get("alignment_confidence", "adjacent")
            themes = topic_alignment.get("related_persona_themes") or topic_alignment.get("themes") or []
            if note:
                alignment_block = textwrap.dedent(
                    f"""
TOPIC ALIGNMENT NOTE (confidence={confidence}):
{note}
Remember: pass along how the paper themes map into neurodivergent lived experience.
Related persona themes to reinforce: {', '.join(themes[:6]) if themes else 'see personas'}.
"""
                )

        deterministic_stats = ""
        if data_profile:
            word_count = data_profile.get("word_count")
            top_terms = data_profile.get("top_terms") or []
            lead_sentences = data_profile.get("lead_sentences")
            deterministic_stats = "SOURCE STATS:\n"
            if word_count:
                deterministic_stats += f"- Word count: {word_count}\n"
            if top_terms:
                deterministic_stats += f"- Top terms: {', '.join(top_terms[:10])}\n"
            if lead_sentences:
                deterministic_stats += f"- Lead sentences: {lead_sentences[:350]}\n"

        instructions = textwrap.dedent(
            f"""
You are Liz Wooten, founder of Enlitens.  Your job is to rewrite the St. Louis
health report into a plain-spoken reference sheet our multi-agent system can rely on.

GOALS:
1. Tell the truth about St. Louis cultural pressure without therapy jargon.
2. Spell out how those pressures land on neurodivergent folks in our client profiles.
3. Give marketing, clinical, and validation agents clean JSON they can reuse.

OUTPUT FORMAT: VALID JSON ONLY.  Use this schema exactly:
{{
  "headline": <string>,
  "summary_bullets": [<string>, ...],
  "liz_voice_pillars": [<string>, ...],
  "cultural_flashpoints": [
    {{
      "label": <string>,
      "description": <string>,
      "mental_health_impact": [<string>, ...],
      "counseling_moves": [<string>, ...]
    }}
  ],
  "neighborhood_snapshots": [
    {{
      "area": <string>,
      "vibe": <string>,
      "pressure_points": [<string>, ...],
      "strengths": [<string>, ...],
      "cues_to_listen_for": [<string>, ...]
    }}
  ],
  "resource_highlights": [
    {{
      "name": <string>,
      "description": <string>,
      "link": <string | null>
    }}
  ],
  "language_watchouts": {{
    "words_to_use": [<string>, ...],
    "words_to_avoid": [<string>, ...]
  }},
  "recommended_actions": [<string>, ...]
}}

RULES:
- Speak like Liz: direct, skeptical of fluff, anchored in neuroscience and lived reality.
- Tie every point back to St. Louis dynamics (high school question, county splits, etc.).
- No clinical jargon, no mindfulness clichés, no "journey/pathway" metaphors.
- Prefer first-person or second-person phrasing that mirrors intake transcripts.
- Bury every summary bullet, flashpoint description, and neighborhood snapshot with at least one concrete number (rates, counts, dollars) pulled from the report; if the source omits a number, say \"No stat reported\" so downstream agents know the gap.
- Point out any banned words (mindfulness, journey, manifest, pathway, etc.) that appear in the source so copywriters can scrub them.
- In cultural_flashpoints[].counseling_moves include language Liz would actually use with clients (short, directive sentences).

{language_block}
{alignment_block}
{deterministic_stats}
SOURCE REPORT (trimmed to fit context window):
{trimmed_report}
"""
        )

        return instructions

    # --------------------------------------------------------------------- #
    # Response handling
    # --------------------------------------------------------------------- #

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from the LLM response.
        """
        if not response:
            return None

        try:
            json_blob = extract_json_object(response)
            if json_blob is None:
                raise ValueError("JSON delimiters not found")
            parsed = json.loads(json_blob)
            if not isinstance(parsed, dict):
                raise ValueError("Parsed payload is not an object")
            return parsed
        except Exception as exc:
            logger.warning(
                "HealthReportTranslator: failed to parse JSON (clean attempt): %s", exc
            )
            cleaned = strip_reasoning_artifacts(response)
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    logger.info("HealthReportTranslator: parsed JSON after stripping reasoning artefacts.")
                    return parsed
            except Exception as inner_exc:
                logger.warning(
                    "HealthReportTranslator: JSON parse retry failed: %s | sample=%s",
                    inner_exc,
                    cleaned[:400],
                )
            return None

    def _normalise_payload(
        self,
        *,
        payload: Dict[str, Any],
        digest_id: str,
        source_metadata: Dict[str, Any],
    ) -> HealthReportDigest:
        """
        Convert raw dict into HealthReportDigest with defaults.
        """
        headline = self._coerce_str(payload.get("headline")) or "St. Louis runs on social codes we have to name out loud."
        summary_bullets = self._coerce_list(payload.get("summary_bullets"))
        liz_voice_pillars = self._coerce_list(payload.get("liz_voice_pillars"))

        flashpoints = [
            CulturalFlashpoint(
                label=self._coerce_str(item.get("label")) or "Name the High School filter.",
                description=self._coerce_str(item.get("description")) or "The high school question is a caste system shortcut. Call it out.",
                mental_health_impact=self._coerce_list(item.get("mental_health_impact")),
                counseling_moves=self._coerce_list(item.get("counseling_moves")),
            )
            for item in self._coerce_list(payload.get("cultural_flashpoints"), dict_expected=True)
        ]

        neighborhoods = [
            NeighborhoodSnapshot(
                area=self._coerce_str(item.get("area")) or "St. Louis City",
                vibe=self._coerce_str(item.get("vibe")) or "Gritty, loud, but honest about what's broken.",
                pressure_points=self._coerce_list(item.get("pressure_points")),
                strengths=self._coerce_list(item.get("strengths")),
                cues_to_listen_for=self._coerce_list(item.get("cues_to_listen_for")),
            )
            for item in self._coerce_list(payload.get("neighborhood_snapshots"), dict_expected=True)
        ]

        resources = [
            ResourceHighlight(
                name=self._coerce_str(item.get("name")) or "Saint Louis Crisis Nursery",
                description=self._coerce_str(item.get("description")) or "Emergency support that keeps families afloat when systems fail.",
                link=self._coerce_optional_str(item.get("link")),
            )
            for item in self._coerce_list(payload.get("resource_highlights"), dict_expected=True)
        ]

        language_watchouts = payload.get("language_watchouts")
        if not isinstance(language_watchouts, dict):
            language_watchouts = {}
        normalised_watchouts = {
            "words_to_use": self._coerce_list(language_watchouts.get("words_to_use")),
            "words_to_avoid": self._coerce_list(language_watchouts.get("words_to_avoid")),
        }

        recommended_actions = self._coerce_list(payload.get("recommended_actions"))

        key_statistics = self._coerce_statistics(payload.get("key_statistics"))
        if not key_statistics:
            key_statistics = source_metadata.get("quick_stats") or []

        digest = HealthReportDigest(
            digest_id=digest_id,
            generated_at=datetime.utcnow(),
            version=self.version,
            headline=headline,
            summary_bullets=summary_bullets,
            liz_voice_pillars=liz_voice_pillars,
            cultural_flashpoints=flashpoints[:6],
            neighborhood_snapshots=neighborhoods[:8],
            resource_highlights=resources[:10],
            language_watchouts=normalised_watchouts,
            recommended_actions=recommended_actions[:10],
            key_statistics=key_statistics[:10],
            prompt_block="",
            source_metadata=source_metadata,
        )
        return digest

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _trim_report(self, report_text: str) -> str:
        """
        Trim the report for prompt inclusion while keeping representative sections.
        """
        if len(report_text) <= self.max_prompt_chars:
            return report_text

        sections = _SECTION_SPLIT_RE.split(report_text)
        if len(sections) < 2:
            return report_text[: self.max_prompt_chars]

        excerpts: List[str] = []
        total_chars = 0
        for section in sections:
            snippet = section.strip()
            if not snippet:
                continue
            excerpt = snippet[:1200]
            if total_chars + len(excerpt) > self.max_prompt_chars:
                break
            excerpts.append(excerpt)
            total_chars += len(excerpt)
        return "\n\n---\n\n".join(excerpts)[: self.max_prompt_chars]

    def _build_source_metadata(
        self,
        report_text: str,
        data_profile: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Assemble lightweight deterministic metadata for observability.
        """
        metadata: Dict[str, Any] = {
            "char_count": len(report_text or ""),
            "generated_at": datetime.utcnow().isoformat(),
        }
        if data_profile:
            metadata.update(
                {
                    "word_count": data_profile.get("word_count"),
                    "top_terms": data_profile.get("top_terms"),
                    "lead_sentences": data_profile.get("lead_sentences"),
                }
            )
        metadata["quick_stats"] = self._extract_numeric_snippets(report_text)
        return metadata

    def _fallback_digest(
        self,
        *,
        digest_id: str,
        source_metadata: Dict[str, Any],
        language_profile: Optional[Dict[str, Any]],
    ) -> HealthReportDigest:
        """
        Produce a deterministic fallback when the LLM path fails.
        """
        summary_bullets = [
            "St. Louis sorts people with the high school question—name it as a caste cue.",
            "County vs. city divides show up as survival math for our neurodivergent clients.",
            "Locals sniff out outsiders fast; speak in plain, lived-in language.",
        ]
        liz_voice_pillars = [
            "No fluff. Call out the system, not the person.",
            "Translate academic language into how clients actually talk.",
            "Tie every insight back to nervous system load and safety.",
        ]
        watchouts = {
            "words_to_use": ["real talk", "call it what it is", "you are not broken"],
            "words_to_avoid": ["mindfulness", "journey", "pathway", "manifest"],
        }
        if language_profile:
            banned_terms = language_profile.get("banned_terms")
            if isinstance(banned_terms, list) and banned_terms:
                watchouts["words_to_avoid"] = sorted(set(watchouts["words_to_avoid"]) | {term.lower() for term in banned_terms})

        digest = HealthReportDigest(
            digest_id=digest_id,
            generated_at=datetime.utcnow(),
            version=self.version,
            headline="St. Louis survives on insider codes—our clients pay the price until we say it out loud.",
            summary_bullets=summary_bullets,
            liz_voice_pillars=liz_voice_pillars,
            cultural_flashpoints=[
                CulturalFlashpoint(
                    label="High School = caste system in disguise",
                    description="Seven words decide if you are 'in' or 'other.' We name it so clients stop blaming themselves.",
                    mental_health_impact=[
                        "Clients relive high school trauma when asked 'Where'd you go?'",
                        "Masking gets heavier when you never fit the code.",
                    ],
                    counseling_moves=[
                        "Validate the microaggression and reframe it as a broken system.",
                        "Offer scripts that shut down the question without shame.",
                    ],
                )
            ],
            neighborhood_snapshots=[
                NeighborhoodSnapshot(
                    area="City (South + Central corridors)",
                    vibe="Gritty, proud, always hustling to stay afloat.",
                    pressure_points=[
                        "High sensory load from noise and density.",
                        "Safety worries and housing instability.",
                    ],
                    strengths=["Community pride", "Walkable third spaces", "Mutual aid"],
                    cues_to_listen_for=["talking about Tower Grove, The Hill, Cherokee"],
                )
            ],
            resource_highlights=[
                ResourceHighlight(
                    name="Saint Louis Crisis Nursery",
                    description="Emergency childcare/shelter that keeps families intact during crisis.",
                    link=None,
                )
            ],
            language_watchouts=watchouts,
            recommended_actions=[
                "Bake the high school caste explanation into every client-facing asset.",
                "Flag copy that slips into therapy jargon before it hits the CMS.",
            ],
            key_statistics=source_metadata.get("quick_stats")[:6]
            if source_metadata.get("quick_stats")
            else [
                {
                    "stat": "2.8 million",
                    "context": "People in the metro navigating insider/outsider sorting every day.",
                },
                {
                    "stat": "94 municipalities",
                    "context": "Fragmented rules that make services feel inconsistent and exhausting.",
                },
            ],
            prompt_block="",
            source_metadata=source_metadata,
        )
        digest.prompt_block = self._compose_prompt_block(digest, language_profile)
        return digest

    def _compose_prompt_block(
        self,
        digest: HealthReportDigest,
        language_profile: Optional[Dict[str, Any]],
    ) -> str:
        """
        Build a reusable snippet that downstream prompts can drop in verbatim.
        """
        lines: List[str] = [
            f"LIZ HEALTH DIGEST: {digest.headline}",
            "",
            "HARD TRUTHS:",
        ]
        for bullet in digest.summary_bullets[:5]:
            lines.append(f"- {bullet}")

        if digest.cultural_flashpoints:
            lines.append("")
            lines.append("Cultural flashpoints to name:")
            for flash in digest.cultural_flashpoints[:4]:
                lines.append(f"- {flash.label}: {flash.description}")

        if digest.key_statistics:
            lines.append("")
            lines.append("Key numbers we can cite:")
            for stat in digest.key_statistics[:5]:
                entry = ""
                if isinstance(stat, dict):
                    value = stat.get("stat") or stat.get("value") or ""
                    context = stat.get("context") or stat.get("narrative") or ""
                    if value and context:
                        entry = f"{value}: {context}"
                    elif value:
                        entry = value
                    else:
                        entry = context
                elif isinstance(stat, str):
                    entry = stat
                if entry:
                    lines.append(f"- {entry}")

        if digest.neighborhood_snapshots:
            lines.append("")
            lines.append("Neighborhood tells + cues:")
            for snap in digest.neighborhood_snapshots[:4]:
                cue = ", ".join(snap.cues_to_listen_for[:2]) if snap.cues_to_listen_for else "listen for local slang."
                lines.append(f"- {snap.area}: {snap.vibe} | cues: {cue}")

        watchouts = digest.language_watchouts or {}
        words_to_use = watchouts.get("words_to_use") or []
        words_to_avoid = watchouts.get("words_to_avoid") or []
        if words_to_use or words_to_avoid:
            lines.append("")
            if words_to_use:
                lines.append(f"Words that land: {', '.join(words_to_use[:8])}")
            if words_to_avoid:
                lines.append(f"Words to shoot down: {', '.join(words_to_avoid[:8])}")

        if language_profile and language_profile.get("prompt_block"):
            lines.append("")
            lines.append("Client language snapshot:")
            lines.append(language_profile["prompt_block"])

        lines.append("")
        lines.append("Stay blunt. No mindfulness/manifest/\"journey\" fluff. Validate the nervous system before offering strategy.")
        return "\n".join(lines)

    def _extract_numeric_snippets(self, report_text: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Pull deterministic numeric snippets from the source report so we always
        have receipts even if the LLM under-delivers.
        """
        if not report_text:
            return []

        patterns = [
            r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",
            r"\b\d+(?:\.\d+)?%\b",
            r"\b\d+(?:\.\d+)?\s+per\s+\d+(?:,\d{3})*\b",
        ]
        combined = re.compile("|".join(patterns), re.IGNORECASE)
        snippets: List[Dict[str, str]] = []
        seen: set[str] = set()

        for match in combined.finditer(report_text):
            sent_start = report_text.rfind(".", 0, match.start())
            if sent_start == -1:
                sent_start = max(0, match.start() - 140)
            else:
                sent_start += 1
            sent_end = report_text.find(".", match.end())
            if sent_end == -1:
                sent_end = min(len(report_text), match.end() + 140)
            else:
                sent_end += 1
            window = re.sub(r"\s+", " ", report_text[sent_start:sent_end].strip())
            window = re.sub(r"^[\d\.\s]+", "", window)
            if len(window) > 260:
                window = f"{window[:250].rstrip()}…"
            value = match.group(0).strip()
            fingerprint = window[:160]
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            snippets.append({"stat": value, "context": window})
            if len(snippets) >= limit:
                break

        return snippets

    @staticmethod
    def _coerce_statistics(value: Any) -> List[Dict[str, str]]:
        """
        Normalise the key_statistics payload from the LLM.
        """
        stats: List[Dict[str, str]] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    stat_value = str(item.get("stat") or item.get("value") or "").strip()
                    context = str(item.get("context") or item.get("narrative") or "").strip()
                    if stat_value or context:
                        stats.append({"stat": stat_value, "context": context})
                elif isinstance(item, str):
                    cleaned = item.strip()
                    if cleaned:
                        stats.append({"stat": "", "context": cleaned})
        return stats

    @staticmethod
    def _coerce_str(value: Any) -> Optional[str]:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        return None

    @staticmethod
    def _coerce_optional_str(value: Any) -> Optional[str]:
        coerced = HealthReportTranslatorAgent._coerce_str(value)
        return coerced

    @staticmethod
    def _coerce_list(value: Any, *, dict_expected: bool = False) -> List[Any]:
        if isinstance(value, list):
            if dict_expected:
                return [item for item in value if isinstance(item, dict)]
            return [item for item in value if isinstance(item, (str, int, float, dict, list))]
        if isinstance(value, (str, int, float)) and not dict_expected:
            return [value]
        return []


__all__ = ["HealthReportTranslatorAgent"]


