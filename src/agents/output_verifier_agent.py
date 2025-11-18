"""
LLM-based judge for evaluating the final multi-agent output bundle.

The verifier ensures the generated knowledge entry is grounded in the paper,
covers required content areas, and adheres to Liz's voice guidelines before we
persist it or send it downstream.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from src.utils.llm_response_cleaner import extract_json_object

logger = logging.getLogger(__name__)


class OutputVerifierAgent:
    """Assess the final supervisor output prior to conversion into KB entries."""

    def __init__(self, max_prompt_chars: int = 5000) -> None:
        self.max_prompt_chars = max_prompt_chars

    async def evaluate(
        self,
        *,
        summary_context: Dict[str, Any],
        supervisor_result: Dict[str, Any],
        llm_client: Any,
    ) -> Dict[str, Any]:
        """
        Run an LLM evaluation returning a structured verdict.

        Returns JSON-compatible dict with keys:
            status: "pass" | "revise" | "error"
            warnings: list[str]
            recommendations: list[str]
        """
        try:
            prompt = self._build_prompt(summary_context, supervisor_result)
        except Exception as exc:
            logger.warning("Output verifier prompt build failed: %s", exc)
            return {"status": "error", "warnings": [str(exc)], "recommendations": []}

        try:
            logger.info("🛡️ Output verifier evaluating supervisor result")
            generation = await llm_client.generate_response(
                prompt=prompt,
                temperature=0.2,
                num_predict=900,
                response_format="json_object",
            )
            return self._parse_response(generation.get("response", ""))
        except Exception as exc:
            logger.warning("Output verifier execution failed: %s", exc)
            return {"status": "error", "warnings": [f"execution_error: {exc}"], "recommendations": []}

    def _build_prompt(self, summary_context: Dict[str, Any], supervisor_result: Dict[str, Any]) -> str:
        """Construct the evaluation prompt from supplied context."""
        personas_preview = (summary_context.get("personas_text") or "")[: self.max_prompt_chars]
        health_preview = (summary_context.get("health_brief") or "")[: self.max_prompt_chars]
        voice_preview = (summary_context.get("voice_guide") or "")[: self.max_prompt_chars]
        mechanism_preview = (summary_context.get("mechanism_bridge") or "")[: self.max_prompt_chars]
        stats_preview = summary_context.get("local_stats") or []
        language_guidance = ""
        language_profile = summary_context.get("language_profile") if isinstance(summary_context, dict) else None
        if language_profile:
            snippet = language_profile.get("prompt_block")
            if snippet:
                language_guidance = f"\nAUDIENCE LANGUAGE GUARDRAILS:\n{snippet}\n"
        health_digest_guidance = ""
        if isinstance(summary_context, dict):
            digest_prompt = summary_context.get("health_digest_prompt")
            digest_payload = summary_context.get("health_digest")
            if digest_prompt:
                health_digest_guidance = f"\nST. LOUIS DIGEST SNAPSHOT:\n{digest_prompt[:800]}\n"
            elif isinstance(digest_payload, dict):
                headline = digest_payload.get("headline")
                bullets = digest_payload.get("summary_bullets") or []
                lines = []
                if headline:
                    lines.append(f"Headline: {headline}")
                for bullet in bullets[:4]:
                    lines.append(f"- {bullet}")
                if lines:
                    health_digest_guidance = "\nST. LOUIS DIGEST SNAPSHOT:\n" + "\n".join(lines) + "\n"
        alignment_guidance = ""
        alignment_profile = summary_context.get("alignment_profile") if isinstance(summary_context, dict) else None
        if alignment_profile:
            note = alignment_profile.get("alignment_note") or ""
            confidence = alignment_profile.get("alignment_confidence", "adjacent")
            themes = alignment_profile.get("related_persona_themes") or []
            themes_line = f"Focus on translating mechanisms into: {', '.join(themes[:6])}." if themes else ""
            alignment_guidance = (
                f"\nTOPIC ALIGNMENT NOTE (confidence={confidence}):\n{note}\n{themes_line}\n"
                "Approve outputs that clearly articulate this bridge even when diagnoses are implicit."
            )
        mechanism_guidance = ""
        if mechanism_preview:
            mechanism_guidance = f"\nMECHANISM ↔ PERSONA BRIDGE (curator view):\n{mechanism_preview}\n"
        stats_guidance = ""
        if stats_preview:
            stats_lines = "\n".join(f"- {stat}" for stat in stats_preview[:6])
            stats_guidance = f"\nLOCAL STATS ANCHORS:\n{stats_lines}\n"

        marketing = supervisor_result.get("agent_outputs", {}).get("marketing_content", {})
        clinical = supervisor_result.get("agent_outputs", {}).get("clinical_content", {})
        validation = supervisor_result.get("agent_outputs", {}).get("validation_result") or supervisor_result.get(
            "validation_result"
        )

        prompt = f"""You are a senior editor validating the final output of Enlitens AI.
The goal is to ensure the generated content is:
1. Faithful to the research paper and curated personas.
2. Clinically accurate and locally grounded (St. Louis context).
3. Written in Liz Wooten's warm, neuro-affirming voice.

CURATED CONTEXT (personas/health/voice excerpts):
--- PERSONAS ---
{personas_preview}
--- HEALTH BRIEF ---
{health_preview}
--- VOICE GUIDE ---
{voice_preview}
{language_guidance}
{health_digest_guidance}
{alignment_guidance}
{mechanism_guidance}
{stats_guidance}

FINAL SUPERVISOR OUTPUT (key sections):
- Marketing content keys: {list(marketing.keys())}
- Clinical highlights sample: {str(clinical)[:600]}
- Validation summary: {str(validation)[:600]}

Respond with a JSON object:
{{
  "status": "pass" | "revise",
  "warnings": [<string>, ...],
  "recommendations": [<string>, ...]
}}

Use "revise" if any critical section is missing, factual grounding is doubtful,
or the tone deviates from Liz's guidance. Provide practical recommendations.
"""
        return prompt

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data."""
        try:
            json_blob = extract_json_object(response)
            if json_blob is None:
                raise ValueError("No JSON payload detected")
            parsed = json.loads(json_blob)
            parsed.setdefault("warnings", [])
            parsed.setdefault("recommendations", [])
            if parsed.get("status") not in {"pass", "revise"}:
                parsed["status"] = "revise"
                parsed["warnings"].append("Missing or invalid status; defaulted to 'revise'.")
            return parsed
        except Exception as exc:
            logger.warning("Failed to parse output verifier response: %s", exc)
            return {
                "status": "error",
                "warnings": [f"parse_error: {exc}"],
                "recommendations": [],
            }


__all__ = ["OutputVerifierAgent"]

