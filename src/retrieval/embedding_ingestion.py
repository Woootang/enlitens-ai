"""Embedding ingestion pipeline for Enlitens knowledge entries."""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.models.enlitens_schemas import EnlitensKnowledgeEntry

from .chunker import DocumentChunker
from .vector_store import BaseVectorStore, QdrantVectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Summary of an ingestion event for a single document."""

    document_id: str
    chunks_ingested: int
    full_text_chunks: int
    agent_chunks: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrityItem:
    """Integrity check result for a single document."""

    document_id: str
    expected_chunks: int
    indexed_chunks: int
    status: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrityReport:
    """Aggregate integrity report for the entire index."""

    total_expected: int
    total_indexed: int
    documents: List[IntegrityItem]
    generated_at: datetime = field(default_factory=datetime.utcnow)


class EmbeddingIngestionPipeline:
    """Pipeline that converts processed documents into vector embeddings."""

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        chunk_size_tokens: int = 900,
        chunk_overlap_ratio: float = 0.15,
        agent_chunk_token_threshold: int = 180,
    ) -> None:
        self.vector_store = vector_store or QdrantVectorStore()
        self.chunker = DocumentChunker(
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_ratio=chunk_overlap_ratio,
        )
        self.agent_chunk_token_threshold = agent_chunk_token_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest_entry(self, entry: EnlitensKnowledgeEntry) -> IngestionStats:
        """Ingest a processed knowledge entry into the vector store."""

        document_id = entry.metadata.document_id
        full_text = entry.full_document_text or ""
        agent_outputs = self._entry_to_agent_outputs(entry)
        metadata = self._sanitize_metadata(entry)

        return self.ingest_document(
            document_id=document_id,
            full_text=full_text,
            agent_outputs=agent_outputs,
            metadata=metadata,
        )

    def ingest_document(
        self,
        document_id: str,
        full_text: str,
        agent_outputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rebuild: bool = False,
    ) -> IngestionStats:
        """Ingest a document and optional agent outputs into the vector store."""

        if rebuild:
            logger.info("Refreshing vector index for %s", document_id)
            try:
                self.vector_store.delete_by_document(document_id)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to delete existing vectors for %s: %s", document_id, exc)

        chunks = self._generate_chunks(
            document_id=document_id,
            full_text=full_text,
            agent_outputs=agent_outputs or {},
            metadata=metadata or {},
        )

        if not chunks:
            logger.warning("No chunks generated for document %s", document_id)
            return IngestionStats(
                document_id=document_id,
                chunks_ingested=0,
                full_text_chunks=0,
                agent_chunks=0,
                metadata=metadata or {},
            )

        try:
            self.vector_store.upsert(chunks)
        except Exception as exc:
            logger.error("Embedding ingestion failed for %s: %s", document_id, exc)
            raise

        full_text_chunks = sum(1 for chunk in chunks if chunk["metadata"].get("source_type") == "full_document_text")
        agent_chunks = len(chunks) - full_text_chunks
        logger.info(
            "Ingested %d chunks for %s (%d full text, %d agent outputs)",
            len(chunks),
            document_id,
            full_text_chunks,
            agent_chunks,
        )

        return IngestionStats(
            document_id=document_id,
            chunks_ingested=len(chunks),
            full_text_chunks=full_text_chunks,
            agent_chunks=agent_chunks,
            metadata=metadata or {},
        )

    def ingest_entries(self, entries: Iterable[EnlitensKnowledgeEntry], rebuild: bool = False) -> List[IngestionStats]:
        stats: List[IngestionStats] = []
        for entry in entries:
            try:
                stats.append(self.ingest_entry(entry) if not rebuild else self.ingest_entry_with_rebuild(entry))
            except Exception as exc:
                logger.error("Failed to ingest entry %s: %s", entry.metadata.document_id, exc)
        return stats

    def ingest_entry_with_rebuild(self, entry: EnlitensKnowledgeEntry) -> IngestionStats:
        return self.ingest_document(
            document_id=entry.metadata.document_id,
            full_text=entry.full_document_text or "",
            agent_outputs=self._entry_to_agent_outputs(entry),
            metadata=self._sanitize_metadata(entry),
            rebuild=True,
        )

    def run_integrity_check(
        self,
        entries: Sequence[EnlitensKnowledgeEntry],
    ) -> IntegrityReport:
        """Compare expected chunk counts with indexed chunk counts."""

        report_items: List[IntegrityItem] = []
        total_expected = 0
        total_indexed = 0

        for entry in entries:
            expected_chunks = self._estimate_chunk_count(
                document_id=entry.metadata.document_id,
                full_text=entry.full_document_text or "",
                agent_outputs=self._entry_to_agent_outputs(entry),
                metadata=self._sanitize_metadata(entry),
            )
            indexed_chunks = self.vector_store.count_by_document(entry.metadata.document_id)

            status = "ok"
            if indexed_chunks < expected_chunks:
                status = "missing"
            elif indexed_chunks > expected_chunks:
                status = "stale"

            report_items.append(
                IntegrityItem(
                    document_id=entry.metadata.document_id,
                    expected_chunks=expected_chunks,
                    indexed_chunks=indexed_chunks,
                    status=status,
                    details={"filename": entry.metadata.filename},
                )
            )

            total_expected += expected_chunks
            total_indexed += indexed_chunks

        return IntegrityReport(
            total_expected=total_expected,
            total_indexed=total_indexed,
            documents=report_items,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _entry_to_agent_outputs(self, entry: EnlitensKnowledgeEntry) -> Dict[str, Any]:
        return {
            "rebellion_framework": entry.rebellion_framework.model_dump(),
            "marketing_content": entry.marketing_content.model_dump(),
            "seo_content": entry.seo_content.model_dump(),
            "website_copy": entry.website_copy.model_dump(),
            "blog_content": entry.blog_content.model_dump(),
            "social_media_content": entry.social_media_content.model_dump(),
            "educational_content": entry.educational_content.model_dump(),
            "clinical_content": entry.clinical_content.model_dump(),
            "research_content": entry.research_content.model_dump(),
            "content_creation_ideas": entry.content_creation_ideas.model_dump(),
            "extracted_entities": entry.extracted_entities.model_dump(),
        }

    def _sanitize_metadata(self, entry: EnlitensKnowledgeEntry) -> Dict[str, Any]:
        metadata = entry.metadata.model_dump()
        metadata.update({
            "document_id": entry.metadata.document_id,
            "filename": entry.metadata.filename,
        })
        if entry.metadata.processing_timestamp:
            metadata["processing_timestamp"] = entry.metadata.processing_timestamp.isoformat()
        metadata["full_text_available"] = bool(entry.full_document_text)
        return metadata

    def _generate_chunks(
        self,
        document_id: str,
        full_text: str,
        agent_outputs: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        if full_text and full_text.strip():
            chunks.extend(self._chunk_full_text(document_id, full_text, metadata))

        agent_chunks = self._chunk_agent_outputs(document_id, agent_outputs, metadata)
        chunks.extend(agent_chunks)
        return chunks

    def _chunk_full_text(
        self,
        document_id: str,
        full_text: str,
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        chunk_metadata = {
            "document_id": document_id,
            "source_type": "full_document_text",
        }
        chunk_metadata.update(self._select_metadata_fields(metadata))

        doc_chunks = self.chunker.chunk(full_text, metadata)
        normalized: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(doc_chunks):
            normalized.append(
                {
                    "chunk_id": f"{document_id}::full::{idx}",
                    "text": chunk.get("text", ""),
                    "metadata": {
                        **chunk_metadata,
                        "chunk_index": idx,
                        "token_count": chunk.get("token_count"),
                        "pages": chunk.get("pages", []),
                        "sections": chunk.get("sections", []),
                    },
                }
            )
        return normalized

    def _chunk_agent_outputs(
        self,
        document_id: str,
        agent_outputs: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        base_metadata = self._select_metadata_fields(metadata)

        for agent_name, payload in (agent_outputs or {}).items():
            flattened_segments = self._flatten_agent_payload(agent_name, payload)
            for segment_idx, segment in enumerate(flattened_segments):
                text = segment["text"].strip()
                if not text:
                    continue

                field_path = segment.get("field_path", "value")
                chunk_id_prefix = f"{document_id}::agent::{agent_name}::{field_path}".replace(" ", "_")

                # Use the chunker for longer segments to ensure consistent token limits
                if self._estimate_tokens(text) > self.agent_chunk_token_threshold:
                    sub_chunks = self.chunker.chunk(text, {"sections": [{"title": field_path}]})
                    for sub_idx, sub_chunk in enumerate(sub_chunks):
                        chunks.append(
                            {
                                "chunk_id": f"{chunk_id_prefix}::{sub_idx}",
                                "text": sub_chunk.get("text", ""),
                                "metadata": {
                                    **base_metadata,
                                    "document_id": document_id,
                                    "source_type": "agent_output",
                                    "agent": agent_name,
                                    "field_path": field_path,
                                    "chunk_index": sub_idx,
                                },
                            }
                        )
                else:
                    chunks.append(
                        {
                            "chunk_id": f"{chunk_id_prefix}::{segment_idx}",
                            "text": text,
                            "metadata": {
                                **base_metadata,
                                "document_id": document_id,
                                "source_type": "agent_output",
                                "agent": agent_name,
                                "field_path": field_path,
                                "chunk_index": segment_idx,
                            },
                        }
                    )
        return chunks

    def _flatten_agent_payload(self, agent_name: str, value: Any, path: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        path = path or []
        segments: List[Dict[str, Any]] = []

        if value is None:
            return segments

        if isinstance(value, str):
            segments.append({"agent": agent_name, "field_path": ".".join(path) or agent_name, "text": value})
            return segments

        if isinstance(value, (int, float)):
            segments.append({"agent": agent_name, "field_path": ".".join(path) or agent_name, "text": str(value)})
            return segments

        if isinstance(value, list):
            for idx, item in enumerate(value):
                segments.extend(self._flatten_agent_payload(agent_name, item, path + [str(idx)]))
            return segments

        if isinstance(value, dict):
            for key, item in value.items():
                segments.extend(self._flatten_agent_payload(agent_name, item, path + [str(key)]))
            return segments

        # Fallback for other data types
        segments.append({"agent": agent_name, "field_path": ".".join(path) or agent_name, "text": json.dumps(value)})
        return segments

    def _select_metadata_fields(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        allowed_keys = {
            "document_id",
            "filename",
            "doc_type",
            "processing_timestamp",
            "quality_score",
            "confidence_score",
        }
        return {key: metadata[key] for key in allowed_keys if key in metadata}

    def _estimate_tokens(self, text: str) -> int:
        words = [token for token in text.split() if token]
        return max(len(words), math.ceil(len(text) / 4))

    def _estimate_chunk_count(
        self,
        document_id: str,
        full_text: str,
        agent_outputs: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> int:
        chunks = self._generate_chunks(document_id, full_text, agent_outputs, metadata)
        return len(chunks)


def load_knowledge_entries_from_path(path: str) -> List[EnlitensKnowledgeEntry]:
    """Utility for loading knowledge entries from a JSON file."""
    from src.models.enlitens_schemas import EnlitensKnowledgeBase

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if isinstance(raw, dict) and "documents" in raw:
        kb = EnlitensKnowledgeBase.model_validate(raw)
        return kb.documents

    # Fallback: assume list of entries
    return [EnlitensKnowledgeEntry.model_validate(item) for item in raw]
