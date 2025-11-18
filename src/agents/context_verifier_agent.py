"""
LLM-based verifier for the curated context bundle (personas, health brief, voice guide).

The verifier mirrors the DS-STAR plan/execute/verify loop by analysing the
generated context and providing structured feedback that downstream logic can
act upon.  It only relies on the locally hosted LLM (via the shared VLLM
client) and returns machine-parsable JSON so the caller can decide whether to
retry, adjust prompts, or accept the result.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from src.utils.llm_response_cleaner import extract_json_object

logger = logging.getLogger(__name__)


class ContextVerificationAgent:
    """Run reflective evaluations on the curated context block."""

    def __init__(self, max_prompt_chars: int = 4000) -> None:
        self.max_prompt_chars = max_prompt_chars

    async def evaluate(
        self,
        *,
        paper_profile: Optional[Dict[str, Any]],
        curated_context: Dict[str, Any],
        data_profiles: Optional[Dict[str, Any]],
        llm_client: Any,
        alignment_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the curated context and return structured guidance.

        Returns a dict with:
            status: "pass", "revise", or "error"
            persona_feedback, health_feedback, voice_feedback: optional strings
            issues: list of textual observations
        """
        try:
            prompt = self._build_prompt(
                paper_profile=paper_profile,
                curated_context=curated_context,
                data_profiles=data_profiles,
                alignment_profile=alignment_profile,
            )
        except Exception as exc:
            logger.warning("Failed to build verification prompt: %s", exc)
            return {
                "status": "error",
                "issues": ["Failed to construct verification prompt", str(exc)],
            }

        try:
            logger.info("🧪 Verifier evaluating curated context (prompt %d chars)", len(prompt))
            generation = await llm_client.generate_response(
                prompt=prompt,
                temperature=0.1,
                num_predict=800,
                response_format="json_object",
            )
            parsed = self._parse_response(generation.get("response", ""))
        except Exception as exc:
            logger.warning("Verifier execution error: %s", exc)
            return {"status": "error", "issues": [f"Verifier execution failed: {exc}"]}

        logger.info(
            "🧪 Verifier result: status=%s, issues=%d",
            parsed.get("status"),
            len(parsed.get("issues", [])),
        )
        return parsed

    def _build_prompt(
        self,
        *,
        paper_profile: Optional[Dict[str, Any]],
        curated_context: Dict[str, Any],
        data_profiles: Optional[Dict[str, Any]],
        alignment_profile: Optional[Dict[str, Any]],
    ) -> str:
        """Create the verification prompt supplied to the local LLM."""
        persona_preview = (curated_context.get("personas_text") or "")[: self.max_prompt_chars]
        health_preview = (curated_context.get("health_brief") or "")[: self.max_prompt_chars]
        voice_preview = (curated_context.get("voice_guide") or "")[: self.max_prompt_chars]
        language_guidance = ""
        language_profile = curated_context.get("language_profile") if isinstance(curated_context, dict) else None
        if language_profile:
            snippet = language_profile.get("prompt_block")
            if snippet:
                language_guidance = f"\nAUDIENCE LANGUAGE GUARDRAILS:\n{snippet}\n"

        health_digest_guidance = ""
        if isinstance(curated_context, dict):
            digest_prompt = curated_context.get("health_digest_prompt")
            digest_payload = curated_context.get("health_digest")
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

        external_guidance = ""
        if isinstance(curated_context, dict):
            external_context = curated_context.get("external_context") or {}
            external_lines: List[str] = []
            local_resources = external_context.get("local_resources") or []
            if local_resources:
                external_lines.append("Local resources (Google Maps):")
                for item in local_resources[:4]:
                    name = item.get("name", "Resource")
                    description = item.get("description", "")
                    location_query = item.get("location_query", "")
                    external_lines.append(f"- {name} near {location_query}: {description}")
            entity_summaries = external_context.get("entity_summaries") or {}
            if entity_summaries:
                external_lines.append("Entity definitions (Wikimedia):")
                for key, summary in list(entity_summaries.items())[:4]:
                    external_lines.append(f"- {key}: {summary}")
            if external_lines:
                external_guidance = "\nEXTERNAL CONTEXT:\n" + "\n".join(external_lines) + "\n"

        alignment_guidance = ""
        if alignment_profile:
            note = alignment_profile.get("alignment_note") or ""
            confidence = alignment_profile.get("alignment_confidence", "adjacent")
            themes = alignment_profile.get("related_persona_themes") or []
            themes_line = f"Linked persona themes supplied: {', '.join(themes[:6])}." if themes else ""
            alignment_guidance = (
                f"\nTOPIC ALIGNMENT NOTE (confidence={confidence}):\n{note}\n{themes_line}\n"
                "Allow a PASS if the personas and health brief explicitly explain this bridge, even when diagnoses are indirect."
            )

        paper_summary = ""
        if paper_profile:
            paper_summary = (
                f"- Word count: {paper_profile.get('word_count')}\n"
                f"- Lead sentences: {paper_profile.get('lead_sentences', '')[:400]}\n"
                f"- Top terms: {', '.join(paper_profile.get('top_terms', [])[:8])}"
            )

        entity_summary = ""
        if data_profiles and "entities" in data_profiles:
            buckets = data_profiles["entities"].get("buckets", {})
            parts = []
            for bucket, info in buckets.items():
                parts.append(
                    f"{bucket}: {info.get('count', 0)} "
                    f"(sample: {', '.join(info.get('sample', [])[:3])})"
                )
            entity_summary = "\n".join(parts)

        health_profile = ""
        if data_profiles and "health_report" in data_profiles:
            hp = data_profiles["health_report"]
            health_profile = (
                f"- Word count: {hp.get('word_count')}\n"
                f"- Top terms: {', '.join(hp.get('top_terms', [])[:8])}\n"
                f"- Lead sentences: {hp.get('lead_sentences', '')[:400]}"
            )

        prompt = f"""You are an autonomous quality assurance analyst for Enlitens AI.
Your task is to inspect the curated context that will feed downstream content
generation.  Evaluate whether it fully represents the source paper, aligns
with the requested personas, and follows Liz's voice guidelines.

SOURCE PAPER SNAPSHOT:
{paper_summary or 'No paper summary available.'}

EXTRACTED ENTITY SUMMARY:
{entity_summary or 'Entities not provided.'}

HEALTH REPORT SNAPSHOT:
{health_profile or 'No health report profile available.'}

CURATED PERSONAS (excerpt):
{persona_preview}

HEALTH BRIEF (excerpt):
{health_preview}

VOICE GUIDE (excerpt):
{voice_preview}
{language_guidance}
{health_digest_guidance}
{external_guidance}
{alignment_guidance}

Please respond with a valid JSON object using the following schema:
{{
  "status": "pass" | "revise",
  "persona_feedback": <string or null>,
  "health_feedback": <string or null>,
  "voice_feedback": <string or null>,
  "issues": [<string>, ...],
  "mechanism_alignment_ok": <bool>,
  "statistical_support_ok": <bool>,
  "liz_voice_ok": <bool>
}}

Use "pass" if the context is comprehensive and either directly aligned OR it
clearly applies the alignment note to connect research mechanisms to the
personas. Use "revise" only when the bridge is missing, statistics are absent,
or Liz's voice guide is ignored. Provide actionable feedback in the respective
*_feedback fields.

Set "mechanism_alignment_ok" to true only when the personas and health brief
explicitly explain how the paper's findings relate to the lived experiences or
themes in the alignment note (adjacent links are acceptable if spelled out).
Set "statistical_support_ok" to true only when the health brief (or digest)
 surfaces concrete numbers, rates, or prevalence tied to either the paper or
the St. Louis digest sections. Set "liz_voice_ok" to true only when the
language demonstrably mirrors Liz's voice guidance (plain-spoken, no banned
terminology, first-person bridges, etc.).
"""
        return prompt

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Attempt to parse the verifier response as JSON.  If parsing fails,
        downgrade to an error result so the caller can decide what to do.
        """
        try:
            json_blob = extract_json_object(response)
            if json_blob is None:
                raise ValueError("No JSON object found in verifier output")
            parsed = json.loads(json_blob)
            parsed.setdefault("issues", [])
            if parsed.get("status") not in {"pass", "revise"}:
                parsed["status"] = "revise"
                parsed["issues"].append("Verifier response missing valid status; defaulted to 'revise'.")
            parsed["mechanism_alignment_ok"] = bool(parsed.get("mechanism_alignment_ok"))
            parsed["statistical_support_ok"] = bool(parsed.get("statistical_support_ok"))
            parsed["liz_voice_ok"] = bool(parsed.get("liz_voice_ok"))
            return parsed
        except Exception as exc:
            logger.warning("Failed to parse verifier response: %s", exc)
            return {
                "status": "error",
                "issues": [f"Could not parse verifier JSON: {exc}"],
                "mechanism_alignment_ok": False,
                "statistical_support_ok": False,
                "liz_voice_ok": False,
            }


__all__ = ["ContextVerificationAgent"]

