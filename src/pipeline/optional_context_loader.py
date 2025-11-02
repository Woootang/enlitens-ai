"""Utilities for loading optional contextual resources."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def read_optional_text_file(path: Path, *, description: str) -> Optional[str]:
    """Read text from an optional file, logging guidance when it is absent."""

    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
        logger.debug(
            "Optional %s file not found at %s. Provide this file to enrich %s.",
            description,
            path,
            description,
        )
        return None
    except Exception as exc:
        logger.warning(
            "Unable to read optional %s file at %s: %s. The pipeline will proceed without it; "
            "verify the file exists and is readable to include %s.",
            description,
            path,
            exc,
            description,
        )
        return None


def analyze_optional_context(
    path: Path,
    *,
    description: str,
    analyzer: Optional[Callable[[List[str]], Dict[str, Any]]],
    fallback_slice: int = 1000,
) -> Dict[str, Any]:
    """Load optional context content and pass it through an analyzer if available."""

    content = read_optional_text_file(path, description=description)
    if content is None:
        return {}

    if analyzer is None:
        return {"raw_content": content[:fallback_slice]}

    try:
        return analyzer([content])
    except Exception as exc:
        logger.warning(
            "Analyzer for %s failed: %s. Returning raw content fallback.",
            description,
            exc,
        )
        return {"raw_content": content[:fallback_slice]}
