from __future__ import annotations

import re
from typing import Dict, Iterable

# Default terminology policy aimed at non-diagnostic, non-DSM language
DEFAULT_REPLACEMENTS: Dict[str, str] = {
    r"\bdisorder(s)?\b": "condition",
    r"\bmental\s+disorder(s)?\b": "mental health condition",
    r"\bdiagnos(e|ed|is|tic|tics)\b": "identify",
    r"\bpatient(s)?\b": "person",
}

DEFAULT_BANNED: Iterable[str] = [
    r"\bDSM-?5\b",
]


def sanitize_text(text: str, replacements: Dict[str, str] | None = None) -> str:
    if not text:
        return text
    rules = replacements or DEFAULT_REPLACEMENTS
    sanitized = text
    for pattern, repl in rules.items():
        sanitized = re.sub(pattern, repl, sanitized, flags=re.IGNORECASE)
    return sanitized


def contains_banned_terms(text: str, banned: Iterable[str] | None = None) -> bool:
    if not text:
        return False
    checks = banned or DEFAULT_BANNED
    return any(re.search(pat, text, flags=re.IGNORECASE) for pat in checks)


def sanitize_structure(obj):
    """Recursively sanitize strings in a JSON-serializable structure."""
    if obj is None:
        return obj
    if isinstance(obj, str):
        return sanitize_text(obj)
    if isinstance(obj, list):
        return [sanitize_structure(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize_structure(v) for k, v in obj.items()}
    return obj


