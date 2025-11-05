"""Helpers to surface prediction error pivots from the knowledge base."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence
import json
import logging
import re

from pydantic import ValidationError

from src.models.enlitens_schemas import EnlitensKnowledgeBase
from src.models.prediction_error import PredictionErrorEntry

logger = logging.getLogger(__name__)

DEFAULT_KNOWLEDGE_BASE_PATH = Path("enlitens_knowledge_base_latest.json")
_SOURCE_PATTERN = re.compile(r"\[(?:Source|Ext)[^\]]+\]", re.IGNORECASE)


@dataclass(frozen=True)
class PredictionErrorRecord:
    """Lightweight view of a prediction error tied to a profile."""

    document_id: str
    profile_name: str
    entry: PredictionErrorEntry
    shared_thread: Optional[str] = None
    source_tags: Sequence[str] = ()

    def with_sources(self) -> "PredictionErrorRecord":
        """Return a copy that includes parsed source tags."""
        sources = set(self.source_tags)
        sources.update(_SOURCE_PATTERN.findall(self.entry.trigger_context))
        sources.update(_SOURCE_PATTERN.findall(self.entry.surprising_pivot))
        sources.update(_SOURCE_PATTERN.findall(self.entry.intended_cognitive_effect))
        return PredictionErrorRecord(
            document_id=self.document_id,
            profile_name=self.profile_name,
            entry=self.entry,
            shared_thread=self.shared_thread,
            source_tags=tuple(sorted({tag.strip() for tag in sources if tag.strip()})),
        )


def _load_knowledge_base(path: Path) -> Optional[EnlitensKnowledgeBase]:
    if not path.exists():
        logger.debug("Knowledge base file not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to read knowledge base %s: %s", path, exc)
        return None
    try:
        return EnlitensKnowledgeBase.model_validate(payload)
    except ValidationError as exc:  # pragma: no cover - defensive
        logger.error("Knowledge base validation failed for %s: %s", path, exc)
        return None


def iter_prediction_error_records(
    *,
    knowledge_base: Optional[EnlitensKnowledgeBase] = None,
    path: Path = DEFAULT_KNOWLEDGE_BASE_PATH,
) -> Iterator[PredictionErrorRecord]:
    """Yield prediction error entries across all documents."""

    kb = knowledge_base or _load_knowledge_base(path)
    if kb is None:
        return

    for document in kb.documents:
        document_id = document.metadata.document_id
        shared_thread = None
        if document.client_profiles:
            shared_thread = document.client_profiles.shared_thread
            for profile in document.client_profiles.profiles:
                for entry in profile.prediction_errors:
                    yield PredictionErrorRecord(
                        document_id=document_id,
                        profile_name=profile.profile_name,
                        entry=entry,
                        shared_thread=shared_thread,
                    )


def load_prediction_error_records(
    path: Path = DEFAULT_KNOWLEDGE_BASE_PATH,
) -> List[PredictionErrorRecord]:
    """Load all prediction error records, including derived source tags."""

    return [record.with_sources() for record in iter_prediction_error_records(path=path)]


def get_prediction_errors_for_profile(
    *,
    document_id: str,
    profile_name: str,
    path: Path = DEFAULT_KNOWLEDGE_BASE_PATH,
) -> List[PredictionErrorEntry]:
    """Retrieve all prediction errors for a specific profile."""

    matches: List[PredictionErrorEntry] = []
    for record in iter_prediction_error_records(path=path):
        if record.document_id == document_id and record.profile_name == profile_name:
            matches.append(record.entry)
    return matches


def filter_prediction_errors_by_locality(
    locality: str,
    *,
    path: Path = DEFAULT_KNOWLEDGE_BASE_PATH,
    knowledge_base: Optional[EnlitensKnowledgeBase] = None,
) -> List[PredictionErrorRecord]:
    """Return prediction errors referencing the provided locality keyword."""

    locality_lower = locality.lower()
    results: List[PredictionErrorRecord] = []
    for record in iter_prediction_error_records(path=path, knowledge_base=knowledge_base):
        haystack = " ".join(
            (
                record.entry.trigger_context,
                record.entry.surprising_pivot,
                record.entry.intended_cognitive_effect,
            )
        ).lower()
        if locality_lower in haystack:
            results.append(record.with_sources())
    return results


def collect_prediction_error_index(
    *,
    path: Path = DEFAULT_KNOWLEDGE_BASE_PATH,
    knowledge_base: Optional[EnlitensKnowledgeBase] = None,
) -> Dict[str, List[PredictionErrorRecord]]:
    """Group prediction errors by document for quick lookup."""

    index: Dict[str, List[PredictionErrorRecord]] = {}
    for record in iter_prediction_error_records(path=path, knowledge_base=knowledge_base):
        index.setdefault(record.document_id, []).append(record.with_sources())
    return index
