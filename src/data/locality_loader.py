"""Utilities for loading and querying St. Louis regional locality reference data."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class LocalityRecord:
    """Structured representation of a locality reference row."""

    name: str
    jurisdiction: str
    median_income_band: str
    demographic_descriptors: str
    landmark_schools: List[str] = field(default_factory=list)
    youth_sports_leagues: List[str] = field(default_factory=list)
    community_centers: List[str] = field(default_factory=list)
    health_resources: List[str] = field(default_factory=list)
    signature_eateries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation of the record."""

        data = asdict(self)
        data["locality_name"] = data.pop("name")
        return data


def _default_csv_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "locality_reference.csv"


def _parse_list_field(raw: str) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(";") if item.strip()]


@lru_cache(maxsize=1)
def load_locality_reference(csv_path: Optional[str] = None) -> Dict[str, LocalityRecord]:
    """Load locality records from the reference CSV.

    Parameters
    ----------
    csv_path:
        Optional override for the CSV path. When omitted, the packaged
        ``data/locality_reference.csv`` file is used.
    """

    path = Path(csv_path) if csv_path else _default_csv_path()
    if not path.exists():
        raise FileNotFoundError(f"Locality reference CSV not found at {path}")

    records: Dict[str, LocalityRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = (row.get("locality_name") or row.get("name") or "").strip()
            if not name:
                continue
            record = LocalityRecord(
                name=name,
                jurisdiction=(row.get("jurisdiction") or "").strip(),
                median_income_band=(row.get("median_income_band") or "").strip(),
                demographic_descriptors=(row.get("demographic_descriptors") or "").strip(),
                landmark_schools=_parse_list_field(row.get("landmark_schools", "")),
                youth_sports_leagues=_parse_list_field(row.get("youth_sports_leagues", "")),
                community_centers=_parse_list_field(row.get("community_centers", "")),
                health_resources=_parse_list_field(row.get("health_resources", "")),
                signature_eateries=_parse_list_field(row.get("signature_eateries", "")),
            )
            records[name.lower()] = record

    return records


def get_locality(name: str) -> Optional[LocalityRecord]:
    """Fetch a single locality record by name (case-insensitive)."""

    if not name:
        return None
    return load_locality_reference().get(name.strip().lower())


def search_by_demographics(
    *,
    median_income_band: Optional[str] = None,
    composition_keywords: Optional[Iterable[str]] = None,
    jurisdiction: Optional[str] = None,
) -> List[LocalityRecord]:
    """Return localities matching demographic filters.

    Parameters
    ----------
    median_income_band:
        Case-insensitive substring match against the ``median_income_band``
        field.
    composition_keywords:
        Iterable of substrings that should appear within the
        ``demographic_descriptors`` field. A string is treated as a single
        keyword. All provided keywords must match.
    jurisdiction:
        Case-insensitive substring match against the jurisdiction label.
    """

    records = load_locality_reference()
    keywords: List[str] = []
    if composition_keywords:
        if isinstance(composition_keywords, str):
            keywords = [composition_keywords.lower()]
        else:
            keywords = [str(keyword).lower() for keyword in composition_keywords if str(keyword).strip()]

    def matches(record: LocalityRecord) -> bool:
        if median_income_band:
            if median_income_band.lower() not in record.median_income_band.lower():
                return False
        if jurisdiction:
            if jurisdiction.lower() not in record.jurisdiction.lower():
                return False
        if keywords:
            descriptor = record.demographic_descriptors.lower()
            if not all(keyword in descriptor for keyword in keywords):
                return False
        return True

    return sorted((record for record in records.values() if matches(record)), key=lambda rec: rec.name)


def all_localities() -> List[LocalityRecord]:
    """Return all locality records preserving alphabetical order."""

    return sorted(load_locality_reference().values(), key=lambda record: record.name)


__all__ = [
    "LocalityRecord",
    "load_locality_reference",
    "get_locality",
    "search_by_demographics",
    "all_localities",
]
