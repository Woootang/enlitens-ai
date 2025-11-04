"""Utilities for tracking knowledge base processing status.

This module centralizes logic for reading and writing the lightweight status
file that signals whether the latest knowledge-base generation completed
successfully. Tools that need to inspect `enlitens_knowledge_base_latest.json`
can use these helpers to gracefully surface failures instead of assuming the
JSON payload is always available.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import json


STATUS_FILE_NAME = "enlitens_knowledge_base_status.json"


@dataclass(frozen=True)
class KnowledgeBaseStatus:
    """Represents the serialized status of the knowledge base writer."""

    status: str
    reason: str
    timestamp: str
    affected_documents: list[str]
    details: Optional[dict[str, Any]] = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "KnowledgeBaseStatus":
        return cls(
            status=str(payload.get("status", "unknown")),
            reason=str(payload.get("reason", "")),
            timestamp=str(payload.get("timestamp", "")),
            affected_documents=list(payload.get("affected_documents", []) or []),
            details=dict(payload.get("details", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "affected_documents": self.affected_documents,
        }
        if self.details:
            payload["details"] = self.details
        return payload


class KnowledgeBaseUnavailableError(RuntimeError):
    """Raised when consumers attempt to load a knowledge base after a crash."""

    def __init__(self, status: KnowledgeBaseStatus):
        message = format_status_message(status)
        super().__init__(message)
        self.status = status


def format_status_message(status: KnowledgeBaseStatus) -> str:
    """Build a concise human-readable message for tooling surfaces."""

    affected = ", ".join(status.affected_documents) if status.affected_documents else "n/a"
    return (
        f"Knowledge base unavailable (status={status.status}): {status.reason}"
        f" | affected_documents: {affected}"
    )


def read_processing_status(path: Path) -> Optional[KnowledgeBaseStatus]:
    """Load the processing status from disk if present."""

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None

    try:
        return KnowledgeBaseStatus.from_mapping(payload)
    except Exception:
        return None


def write_processing_status(
    path: Path,
    *,
    reason: str,
    affected_documents: Iterable[str],
    status: str = "error",
    details: Optional[Mapping[str, Any]] = None,
) -> KnowledgeBaseStatus:
    """Persist an updated processing status to disk."""

    status_payload = KnowledgeBaseStatus(
        status=status,
        reason=reason,
        timestamp=datetime.utcnow().isoformat(),
        affected_documents=list(affected_documents),
        details=dict(details or {}),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(status_payload.to_dict(), handle, indent=2, ensure_ascii=False)
    tmp_path.replace(path)
    return status_payload


def clear_processing_status(path: Path) -> None:
    """Remove any persisted status indicator."""

    if path.exists():
        path.unlink()
