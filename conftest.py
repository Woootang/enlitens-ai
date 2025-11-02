"""Pytest configuration to preload heavy modules before test stubs."""
from __future__ import annotations

import importlib
import sys
from typing import Iterable


def _preload_modules(modules: Iterable[str]) -> None:
    for module in modules:
        if module in sys.modules:
            continue
        try:
            importlib.import_module(module)
        except Exception:
            # Tests that rely on stubs will override these modules as needed.
            continue


def pytest_configure(config):  # type: ignore[override]
    _preload_modules(
        [
            "src.extraction.enhanced_pdf_extractor",
            "src.agents.extraction_team",
        ]
    )
