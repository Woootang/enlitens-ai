"""Lightweight sitecustomize to preload modules before pytest stubs replace them."""
from __future__ import annotations

import importlib
import logging

LOGGER = logging.getLogger(__name__)


def _preload(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - best effort preload
        LOGGER.debug("sitecustomize: failed to preload %s (%s)", module_name, exc)


for _module in (
    "src.extraction.enhanced_pdf_extractor",
    "src.agents.extraction_team",
):
    _preload(_module)
