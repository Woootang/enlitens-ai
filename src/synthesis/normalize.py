"""Normalization helpers for synthesis models.

These utilities perform light-touch sanitation on payloads returned by LLMs
before they are validated against strict Pydantic schemas. Keeping the logic in
one place allows agents and clients to share resilient fallbacks when the model
occasionally returns nested list structures instead of simple strings.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _flatten_to_string(value: Any) -> str:
    """Flatten nested iterables into a whitespace-joined string."""

    if isinstance(value, (str, bytes)):
        return value.decode() if isinstance(value, bytes) else value

    if isinstance(value, dict):
        # Preserve key ordering deterministically for dictionaries by joining
        # "key: value" pairs. This case should be rare but keeps us from losing
        # information if the model returns an unexpected object.
        parts: List[str] = []
        for key in sorted(value.keys()):
            parts.append(f"{key}: {_flatten_to_string(value[key])}")
        return ", ".join(parts)

    if isinstance(value, Iterable):
        flattened: List[str] = []
        for item in value:
            flattened.append(_flatten_to_string(item))
        return " ".join(part for part in flattened if part)

    return str(value)


def normalize_research_content_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure ResearchContent list fields are composed of flat strings.

    Some models occasionally emit nested list structures for fields like
    ``findings``. The ResearchContent schema expects ``List[str]`` entries, so we
    coerce each list element into a single string before validation.
    """

    if not isinstance(payload, dict):
        return payload

    list_fields = {
        "findings",
        "statistics",
        "methodologies",
        "limitations",
        "future_directions",
        "implications",
        "citations",
        "references",
    }

    normalized: Dict[str, Any] = dict(payload)

    for field in list_fields:
        values = normalized.get(field)
        if values is None:
            continue
        if not isinstance(values, list):
            # Allow single string values through – the schema validator will
            # coerce them into lists if necessary.
            continue

        coerced: List[str] = []
        for item in values:
            coerced.append(_flatten_to_string(item).strip())

        normalized[field] = [value for value in coerced if value]

    return normalized


__all__ = ["normalize_research_content_payload"]

