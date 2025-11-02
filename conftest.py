"""Pytest configuration to preload heavy modules before test stubs."""
from __future__ import annotations

from src.utils.module_preload import PRELOAD_MODULES, preload_modules


def pytest_configure(config):  # type: ignore[override]
    preload_modules(PRELOAD_MODULES)
