#!/usr/bin/env python3
"""
PostgreSQL persistence for processed knowledge entries.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

try:  # Optional dependency
    import psycopg2
    from psycopg2 import OperationalError
    from psycopg2.extras import Json
    from psycopg2.extensions import AsIs, ISOLATION_LEVEL_AUTOCOMMIT, register_adapter
except ImportError:  # pragma: no cover - psycopg2 may not be installed in lightweight envs
    psycopg2 = None  # type: ignore
    OperationalError = Exception  # type: ignore
    Json = None  # type: ignore
    AsIs = None  # type: ignore
    ISOLATION_LEVEL_AUTOCOMMIT = None  # type: ignore
    register_adapter = None  # type: ignore

from .embedding_provider import EmbeddingProvider, get_embedding_provider

logger = logging.getLogger(__name__)


class VectorParam(list):
    """Wrapper to adapt Python lists to pgvector literals."""


def _adapt_vector(vector: "VectorParam") -> AsIs:
    literal = "[" + ",".join(f"{float(x):.6f}" for x in vector) + "]"
    return AsIs(f"'{literal}'")


if register_adapter and AsIs:
    register_adapter(VectorParam, _adapt_vector)

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PsycopgConnection  # pragma: no cover
else:
    PsycopgConnection = Any


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        logger.debug("Unable to parse timestamp %s; storing NULL", value)
        return None


def _collect_sections(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    extraction = record.get("extraction") or {}
    sections: List[Tuple[str, str]] = []
    for key, value in extraction.items():
        if key == "citations":
            continue
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                sections.append((key, cleaned))
    return sections


def _build_document_text(record: Dict[str, Any]) -> Optional[str]:
    extraction = record.get("extraction") or {}
    major_fields = []
    for key in ("background", "methods", "findings"):
        value = extraction.get(key)
        if isinstance(value, str) and value.strip():
            major_fields.append(value.strip())
    if major_fields:
        return "\n\n".join(major_fields)
    return None


@dataclass
class DocumentPersistencePayload:
    """Decoded data prepared for database insertion."""

    document_id: str
    title: Optional[str]
    primary_topic: Optional[str]
    model_key: str
    processed_at: Optional[datetime]
    source_pdf: Optional[str]
    checksum: Optional[str]
    tags: Sequence[str]
    metadata: Dict[str, Any]
    docling: Dict[str, Any]
    extraction: Dict[str, Any]
    enrichment: Dict[str, Any]
    knowledge_entry: Optional[Dict[str, Any]]
    gemini_validated: bool
    section_payloads: List[Tuple[str, str]]
    document_text: Optional[str]


class PostgresStore:
    """Thin wrapper around psycopg2 to store processed documents."""

    def __init__(
        self,
        *,
        dsn: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        self.dsn = dsn or os.getenv("DATABASE_URL", "postgresql://localhost/enlitens")
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self._conn: Optional[Any] = None
        self.available = True

        enabled_flag = os.getenv("ENLITENS_ENABLE_POSTGRES", "0").lower() in {"1", "true", "yes", "on"}
        if not enabled_flag:
            logger.info("Postgres persistence disabled (set ENLITENS_ENABLE_POSTGRES=1 to enable).")
            self.available = False
            return

        if psycopg2 is None:
            logger.warning("psycopg2 not installed; skipping Postgres persistence.")
            self.available = False
            return

        try:
            self._connect()
            self._ensure_schema()
        except OperationalError as exc:
            logger.error("Postgres unavailable (%s); persistence disabled.", exc)
            self.available = False
        except Exception as exc:  # pragma: no cover - unexpected setup failure
            logger.error("Failed to set up Postgres schema: %s", exc)
            self.available = False

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def _connect(self) -> PsycopgConnection:
        if psycopg2 is None:
            raise RuntimeError("psycopg2 is required for database connectivity")

        if self._conn and self._conn.closed == 0:
            return self._conn

        logger.debug("Connecting to Postgres at %s", self.dsn)
        conn = psycopg2.connect(self.dsn)
        if ISOLATION_LEVEL_AUTOCOMMIT is not None:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        self._conn = conn
        return conn

    @contextmanager
    def cursor(self):
        if not self.available:
            raise RuntimeError("PostgresStore is not available.")
        conn = self._connect()
        cur = conn.cursor()
        try:
            yield cur
        except Exception:
            cur.close()
            raise
        else:
            cur.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _ensure_schema(self) -> None:
        if not self.available:
            return
        embedding_dim = self.embedding_provider.dimension
        with self.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT,
                    primary_topic TEXT,
                    model_key TEXT,
                    processed_at TIMESTAMPTZ,
                    source_pdf TEXT,
                    checksum_sha256 TEXT,
                    tags JSONB,
                    metadata JSONB,
                    docling JSONB,
                    extraction JSONB,
                    enrichment JSONB,
                    knowledge_entry JSONB,
                    gemini_validated BOOLEAN DEFAULT FALSE,
                    embedding VECTOR({embedding_dim})
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS sections (
                    id BIGSERIAL PRIMARY KEY,
                    document_id TEXT REFERENCES documents(document_id) ON DELETE CASCADE,
                    section_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR({embedding_dim})
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_sections_document ON sections(document_id, section_name)"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def upsert_record(self, record: Dict[str, Any]) -> None:
        if not self.available:
            return

        payload = self._build_payload(record)
        section_texts = [text for _, text in payload.section_payloads]
        section_embeddings: List[List[float]] = []
        if section_texts:
            section_embeddings = self.embedding_provider.embed(section_texts)

        document_embedding: Optional[List[float]] = None
        if payload.document_text:
            embedded = self.embedding_provider.embed_one(payload.document_text)
            if embedded:
                document_embedding = embedded

        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (
                    document_id,
                    title,
                    primary_topic,
                    model_key,
                    processed_at,
                    source_pdf,
                    checksum_sha256,
                    tags,
                    metadata,
                    docling,
                    extraction,
                    enrichment,
                    knowledge_entry,
                    gemini_validated,
                    embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    primary_topic = EXCLUDED.primary_topic,
                    model_key = EXCLUDED.model_key,
                    processed_at = EXCLUDED.processed_at,
                    source_pdf = EXCLUDED.source_pdf,
                    checksum_sha256 = EXCLUDED.checksum_sha256,
                    tags = EXCLUDED.tags,
                    metadata = EXCLUDED.metadata,
                    docling = EXCLUDED.docling,
                    extraction = EXCLUDED.extraction,
                    enrichment = EXCLUDED.enrichment,
                    knowledge_entry = EXCLUDED.knowledge_entry,
                    gemini_validated = EXCLUDED.gemini_validated,
                    embedding = EXCLUDED.embedding
                """,
                (
                    payload.document_id,
                    payload.title,
                    payload.primary_topic,
                    payload.model_key,
                    payload.processed_at,
                    payload.source_pdf,
                    payload.checksum,
                    Json(list(payload.tags)) if Json else None,
                    Json(payload.metadata) if Json else json.dumps(payload.metadata),
                    Json(payload.docling) if Json else json.dumps(payload.docling),
                    Json(payload.extraction) if Json else json.dumps(payload.extraction),
                    Json(payload.enrichment) if Json else json.dumps(payload.enrichment),
                    Json(payload.knowledge_entry) if (Json and payload.knowledge_entry is not None) else (
                        json.dumps(payload.knowledge_entry) if payload.knowledge_entry is not None else None
                    ),
                    payload.gemini_validated,
                    VectorParam(document_embedding) if (document_embedding and register_adapter) else None,
                ),
            )

            cur.execute("DELETE FROM sections WHERE document_id = %s", (payload.document_id,))
            if payload.section_payloads:
                section_rows = []
                for (section_name, content), embedding in zip(
                    payload.section_payloads,
                    section_embeddings or [None] * len(payload.section_payloads),
                ):
                    section_rows.append(
                        (
                            payload.document_id,
                            section_name,
                            content,
                            VectorParam(embedding) if (embedding and register_adapter) else None,
                        )
                    )
                cur.executemany(
                    """
                    INSERT INTO sections (document_id, section_name, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    section_rows,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_payload(self, record: Dict[str, Any]) -> DocumentPersistencePayload:
        metadata = record.get("metadata") or {}
        docling = record.get("docling") or {}
        extraction = record.get("extraction") or {}
        enrichment = record.get("enrichment") or {}

        return DocumentPersistencePayload(
            document_id=record.get("document_id"),
            title=metadata.get("title"),
            primary_topic=metadata.get("primary_topic"),
            model_key=record.get("model_key"),
            processed_at=_parse_timestamp(metadata.get("processed_at")),
            source_pdf=(record.get("source") or {}).get("pdf_path"),
            checksum=(record.get("source") or {}).get("checksum_sha256"),
            tags=metadata.get("tags") or [],
            metadata=metadata,
            docling=docling,
            extraction=extraction,
            enrichment=enrichment,
            knowledge_entry=record.get("knowledge_entry"),
            gemini_validated=bool(record.get("gemini_validated")),
            section_payloads=_collect_sections(record),
            document_text=_build_document_text(record),
        )

