#!/usr/bin/env python3
"""
Vector mirroring for processed knowledge records.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.retrieval.vector_store import BaseVectorStore, ChromaVectorStore

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "enlitens_sections")
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/vector_store/chroma")


def _section_chunks(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    extraction = record.get("extraction") or {}
    document_id = record.get("document_id", "unknown")
    metadata = record.get("metadata") or {}
    processed_at = metadata.get("processed_at")

    chunks: List[Dict[str, Any]] = []
    for section, value in extraction.items():
        if section == "citations":
            continue
        if isinstance(value, str) and value.strip():
            chunk_id = f"{document_id}::{section}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": value.strip(),
                    "metadata": {
                        "document_id": document_id,
                        "section": section,
                        "model_key": record.get("model_key"),
                        "processed_at": processed_at,
                    },
                }
            )
    return chunks


@dataclass
class VectorMirror:
    """Persist section chunks into a local Chroma vector store."""

    store: Optional[BaseVectorStore] = None

    def __post_init__(self) -> None:
        if self.store is None:
            try:
                persist_dir = Path(DEFAULT_PERSIST_DIR)
                persist_dir.mkdir(parents=True, exist_ok=True)
                self.store = ChromaVectorStore(
                    collection_name=DEFAULT_COLLECTION,
                    persist_directory=str(persist_dir),
                )
                logger.info("🧠 Vector mirror ready at %s", persist_dir)
            except Exception as exc:  # pragma: no cover - optional dependency path
                logger.error("Vector mirror unavailable: %s", exc)
                self.store = None

    def mirror(self, record: Dict[str, Any]) -> None:
        if not self.store:
            return
        chunks = _section_chunks(record)
        if not chunks:
            logger.debug("No section chunks generated for %s; skipping vector upsert.", record.get("document_id"))
            return
        try:
            self.store.upsert(chunks)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to mirror vectors for %s: %s", record.get("document_id"), exc)

