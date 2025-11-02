"""Lightweight sitecustomize to preload modules before pytest stubs replace them."""
from __future__ import annotations

import logging

from src.utils.module_preload import PRELOAD_MODULES, preload_modules

LOGGER = logging.getLogger(__name__)


preload_modules(
    PRELOAD_MODULES,
    on_error=lambda module, exc: LOGGER.debug(
        "sitecustomize: failed to preload %s (%s)", module, exc
    ),
)
