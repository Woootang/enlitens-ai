"""Utilities to normalise Qwen-style responses by stripping reasoning preambles."""

from __future__ import annotations

import re
from typing import Any, List, Optional

import json

try:  # PyYAML is optional but available in the env; fall back gracefully if missing.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

_THOUGHT_BLOCK_RE = re.compile(r"<(think|reflection)>.*?</\1>", re.IGNORECASE | re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?|```", re.IGNORECASE)
_FINAL_TAG_RE = re.compile(r"</?final>", re.IGNORECASE)
_CONFIDENCE_LINE_RE = re.compile(r"^\s*(confidence(?:\s*score)?)\s*[:=]\s*[0-9.]+\s*$", re.IGNORECASE | re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",(?=\s*[}\]])")
_PERSONA_FILENAME_RE = re.compile(r"(?:persona|profile)_[A-Za-z0-9_\-]+\.json")


def strip_reasoning_artifacts(text: str) -> str:
    """Remove `<think>` style blocks and standalone confidence lines."""
    if not text:
        return text

    cleaned = _THOUGHT_BLOCK_RE.sub("", text)
    cleaned = _CODE_FENCE_RE.sub("", cleaned)
    cleaned = _FINAL_TAG_RE.sub("", cleaned)
    cleaned = _CONFIDENCE_LINE_RE.sub("", cleaned)
    return cleaned.strip()


def _ensure_json_serialisable(payload: Any) -> Optional[str]:
    if isinstance(payload, (dict, list)):
        try:
            return json.dumps(payload)
        except Exception:
            return None
    return None


def _attempt_json_load(candidate: str) -> Optional[str]:
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None


def extract_json_object(text: str) -> Optional[str]:
    """Return the first JSON object found after cleaning reasoning artefacts."""
    cleaned = strip_reasoning_artifacts(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = cleaned[start : end + 1]

    # Fast path: already valid JSON.
    valid = _attempt_json_load(candidate)
    if valid is not None:
        return valid

    # Remove trailing commas that the Python JSON parser rejects.
    no_trailing = _TRAILING_COMMA_RE.sub("", candidate)
    if no_trailing != candidate:
        valid = _attempt_json_load(no_trailing)
        if valid is not None:
            return valid

    # YAML is more permissive (handles trailing commas, single quotes, etc.).
    if yaml is not None:
        try:
            payload = yaml.safe_load(candidate)
        except Exception:
            payload = None
        serialised = _ensure_json_serialisable(payload)
        if serialised is not None:
            return serialised

    return None


def extract_persona_filenames(text: str) -> List[str]:
    """Extract persona/profile filenames from messy LLM output while preserving order."""
    if not text:
        return []

    candidates: List[str] = []
    for source in (strip_reasoning_artifacts(text), text):
        if not source:
            continue
        matches = _PERSONA_FILENAME_RE.findall(source)
        for match in matches:
            if match not in candidates:
                candidates.append(match)
        if candidates:
            break
    return candidates


__all__ = ["strip_reasoning_artifacts", "extract_json_object", "extract_persona_filenames"]

