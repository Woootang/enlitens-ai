"""
Automated reviewer that inspects curated context before final verification.

The goal is to catch obvious issues (missing mechanism bridge, absent
statistics, Liz-voice drift) early and feed actionable feedback back into the
curation loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from src.utils.llm_response_cleaner import extract_json_object

logger = logging.getLogger(__name__)


class ContextReviewAgent:
    """Lightweight LLM-based reviewer for curated context bundles."""

    def __init__(self, max_prompt_chars: int = 6000) -> None:
        self.max_prompt_chars = max_prompt_chars

    async def review(
        self,
        *,
        curated_context: Dict[str, Any],
        llm_client: Any,
        language_profile: Optional[Dict[str, Any]] = None,
        alignment_profile: Optional[Dict[str, Any]] = None,
        health_digest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(
            curated_context=curated_context,
            language_profile=language_profile,
            alignment_profile=alignment_profile,
            health_digest=health_digest,
        )
        try:
            logger.debug("🕵️ Context review agent evaluating curated bundle.")
            generation = await llm_client.generate_response(
                prompt=prompt,
                temperature=0.2,
                num_predict=800,
                response_format="json_object",
            )
            parsed = self._parse_response(generation.get("response", ""))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Context review agent failed: %s", exc)
            parsed = {
                "overall_pass": False,
                "issues": [f"Reviewer execution failed: {exc}"],
                "mechanism_alignment_ok": False,
                "statistical_support_ok": False,
                "liz_voice_ok": False,
            }
        return parsed

    def _build_prompt(
        self,
        *,
        curated_context: Dict[str, Any],
        language_profile: Optional[Dict[str, Any]],
        alignment_profile: Optional[Dict[str, Any]],
        health_digest: Optional[Dict[str, Any]],
    ) -> str:
        personas_preview = (curated_context.get("personas_text") or "")[: self.max_prompt_chars]
        health_preview = (curated_context.get("health_brief") or "")[: self.max_prompt_chars]
        voice_preview = (curated_context.get("voice_guide") or "")[: self.max_prompt_chars]

        alignment_note = ""
        if alignment_profile:
            note = alignment_profile.get("alignment_note") or ""
            confidence = alignment_profile.get("alignment_confidence", "adjacent")
            themes = alignment_profile.get("related_persona_themes") or []
            themes_line = f"Themes to hit: {', '.join(themes[:6])}." if themes else ""
            alignment_note = f"Alignment (confidence={confidence}): {note}\n{themes_line}\n"

        language_note = ""
        if language_profile:
            snippet = language_profile.get("prompt_block")
            if snippet:
                language_note = f"Voice guardrails:\n{snippet}\n"

        digest_note = ""
        if health_digest:
            headline = health_digest.get("headline")
            bullets = health_digest.get("summary_bullets") or []
            digest_lines = []
            if headline:
                digest_lines.append(headline)
            digest_lines.extend(bullets[:4])
            if digest_lines:
                digest_note = "Digest reference:\n" + "\n".join(digest_lines[:6]) + "\n"

        external_context = curated_context.get("external_context") or {}
        external_note = ""
        if external_context:
            summaries = []
            local_resources = external_context.get("local_resources") or []
            if local_resources:
                summaries.append(
                    "Local resources:\n" + "\n".join(f"- {item.get('name')}: {item.get('description')}"
                                                     for item in local_resources[:4])
                )
            entity_defs = external_context.get("entity_summaries") or {}
            if entity_defs:
                summaries.append(
                    "Entity definitions:\n" + "\n".join(f"- {key}: {val}" for key, val in list(entity_defs.items())[:4])
                )
            if summaries:
                external_note = "\n".join(summaries) + "\n"

        prompt = f"""
You are an internal QA reviewer for Enlitens. Inspect the curated context that
will feed downstream agents and flag issues *before* the main verifier runs.

PERSONAS (excerpt):
{personas_preview}

HEALTH BRIEF (excerpt):
{health_preview}

VOICE GUIDE (excerpt):
{voice_preview}

{alignment_note}
{language_note}
{digest_note}
{external_note}

Respond with a JSON object:
{{
  "overall_pass": true|false,
  "mechanism_alignment_ok": true|false,
  "statistical_support_ok": true|false,
  "liz_voice_ok": true|false,
  "persona_feedback": <string|null>,
  "health_feedback": <string|null>,
  "voice_feedback": <string|null>,
  "issues": [<string>, ...]
}}

Consider it a PASS only if the mechanism linkage is explicit, the brief cites
useful statistics, and the tone matches Liz's guardrails. Provide actionable
feedback in each *_feedback field when something is missing.
""".strip()
        return prompt

    def _parse_response(self, response: str) -> Dict[str, Any]:
        json_blob = extract_json_object(response)
        if json_blob is None:
            raise ValueError("Reviewer did not return JSON.")
        parsed = json.loads(json_blob)
        parsed.setdefault("issues", [])
        parsed.setdefault("overall_pass", False)
        parsed["mechanism_alignment_ok"] = bool(parsed.get("mechanism_alignment_ok"))
        parsed["statistical_support_ok"] = bool(parsed.get("statistical_support_ok"))
        parsed["liz_voice_ok"] = bool(parsed.get("liz_voice_ok"))
        return parsed

