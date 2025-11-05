"""Data loading utilities for Enlitens regional context."""

from .locality_loader import (
    LocalityRecord,
    all_localities,
    get_locality,
    load_locality_reference,
    search_by_demographics,
)

__all__ = [
    "LocalityRecord",
    "all_localities",
    "get_locality",
    "load_locality_reference",
    "search_by_demographics",
]
