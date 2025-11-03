"""Utilities for condensing retrieval and document context into brief prompts."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _sample_document_sections(
    text: str,
    *,
    segments: int = 3,
    segment_chars: int = 360,
) -> str:
    """Create a spaced sampling of the document that preserves late sections."""

    normalized = _normalize_whitespace(text)
    if not normalized:
        return ""

    length = len(normalized)
    if length <= segment_chars:
        return normalized

    if segments <= 1:
        segments = 1

    max_start = max(length - segment_chars, 0)
    positions = {
        max(0, min(max_start, int(round(fraction * max_start))))
        for fraction in (
            i / max(segments - 1, 1)
            for i in range(segments)
        )
    }

    snippets = []
    for index, start in enumerate(sorted(positions), start=1):
        snippet = normalized[start : start + segment_chars].strip()
        if not snippet:
            continue
        ellipsis = "…" if start + segment_chars < length else ""
        snippets.append(f"- Section {index}: {snippet}{ellipsis}")

    return "\n".join(snippets) if snippets else normalized


async def _summarize_with_client(
    ollama_client: Any,
    prompt: str,
    *,
    num_predict: int = 512,
    temperature: float = 0.1,
) -> str:
    if ollama_client is None or not hasattr(ollama_client, "generate_response"):
        return ""

    try:
        payload = await ollama_client.generate_response(
            prompt,
            temperature=temperature,
            num_predict=num_predict,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Context brief generation failed: %s", exc)
        return ""

    text = (payload or {}).get("response")
    if not text:
        return ""
    return text.strip()


async def compose_document_brief(
    *,
    document_text: Optional[str],
    retrieved_passages: Optional[Sequence[Mapping[str, Any]]],
    ollama_client: Optional[Any],
    passage_display_limit: int = 5,
    overflow_threshold: int = 8,
    passage_snippet_chars: int = 240,
    fallback_segments: int = 3,
    fallback_segment_chars: int = 360,
) -> str:
    """Return a concise brief for prompts without truncating the source text."""

    passages = [p for p in (retrieved_passages or []) if p and p.get("text")]

    if passages:
        lines = []
        for idx, passage in enumerate(passages[:passage_display_limit], start=1):
            snippet = _truncate(
                _normalize_whitespace(str(passage.get("text", ""))),
                passage_snippet_chars,
            )
            if snippet:
                lines.append(f"- [Source {idx}] {snippet}")

        overflow = passages[passage_display_limit:]
        if overflow and len(passages) > overflow_threshold:
            prompt_lines = [
                "You are compressing retrieved research overflow into up to three bullets.",
                "Use the provided [Extra #] tags in your summary so citations stay traceable.",
                "Highlight only novel insights not already covered by Sources 1-" + str(passage_display_limit) + ".",
                "",
            ]
            for idx, passage in enumerate(overflow, start=1):
                snippet = _truncate(
                    _normalize_whitespace(str(passage.get("text", ""))),
                    400,
                )
                if snippet:
                    prompt_lines.append(f"[Extra {idx}] {snippet}")

            overflow_prompt = "\n".join(prompt_lines)
            overflow_summary = await _summarize_with_client(ollama_client, overflow_prompt)
            if overflow_summary:
                lines.append("")
                lines.append("Additional retrieved insights:")
                lines.append(overflow_summary)

        return "\n".join(lines) if lines else "No retrieved passages available for briefing."

    if document_text:
        normalized = _normalize_whitespace(document_text)
        llm_prompt = (
            "Summarize the following document into three tight bullets.\n"
            "Ensure the final bullet captures late-stage findings or conclusions so they are not lost.\n\n"
            f"DOCUMENT:\n{normalized}"
        )
        summary = await _summarize_with_client(ollama_client, llm_prompt)
        if summary:
            return summary

        sampled = _sample_document_sections(
            document_text,
            segments=fallback_segments,
            segment_chars=fallback_segment_chars,
        )
        return sampled or "Document text provided but no summary was produced."

    return "No document context was supplied."
