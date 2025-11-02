"""Central configuration for modules that should be eagerly imported.

The test suite as well as :mod:`sitecustomize` attempt to import a couple of
heavyweight modules up-front so that their optional dependency stubs can be
registered consistently.  Prior to this helper the module lists were duplicated
across ``conftest.py`` and ``sitecustomize.py`` which made it easy for them to
drift out of sync.  This module keeps the list and the preload helper in a
single place so both callers share the exact same behaviour.
"""
from __future__ import annotations

import importlib
import sys
from typing import Callable, Iterable, Optional


PRELOAD_MODULES: tuple[str, ...] = (
    "src.extraction.enhanced_pdf_extractor",
    "src.agents.extraction_team",
)


def preload_modules(
    modules: Iterable[str] = PRELOAD_MODULES,
    *,
    on_error: Optional[Callable[[str, Exception], None]] = None,
) -> None:
    """Import each module in *modules* while ignoring import errors.

    Parameters
    ----------
    modules:
        Sequence of module names that should be imported.  Defaults to
        :data:`PRELOAD_MODULES`.
    on_error:
        Optional callback that will be invoked with ``(module_name, exc)`` when
        an import fails.  This allows callers such as :mod:`sitecustomize` to
        emit debug logging without duplicating the import logic.

    The imports are wrapped in a ``try``/``except`` block so that environments
    missing heavy optional dependencies can still run the test suite.  Modules
    that are already present in :data:`sys.modules` are skipped.
    """

    for module_name in modules:
        if module_name in sys.modules:
            continue
        try:
            importlib.import_module(module_name)
        except Exception:
            # Tests rely on light-weight stubs when the real dependency graph is
            # unavailable.  We deliberately swallow the exception here because
            # callers only need best-effort preloading.
            if on_error is not None:
                on_error(module_name, sys.exc_info()[1])
            continue

