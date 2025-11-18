#!/usr/bin/env python3
"""
Optional Neo4j publisher for the Enlitens knowledge graph.

This module is deliberately defensive: if the `neo4j` package is not
installed or the connection parameters are not supplied, all methods turn
into no-ops so the ingestion pipeline can run without a graph backend.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

try:  # Optional dependency
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - module may be absent in light envs
    GraphDatabase = None  # type: ignore


def _safe_iter(values: Optional[Iterable[str]]) -> Iterable[str]:
    if not values:
        return []
    return [value for value in values if value]


@dataclass
class Neo4jPublisher:
    """
    Push document metadata into Neo4j so downstream agents can reason over
    document/tag/citation relationships.
    """

    uri: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None

    def __post_init__(self) -> None:
        env_enabled = os.getenv("ENLITENS_ENABLE_NEO4J", "0").lower() in {"1", "true", "yes", "on"}
        if not env_enabled:
            logger.info("Neo4j publisher disabled (set ENLITENS_ENABLE_NEO4J=1 to enable).")
            self.driver = None
            return

        if GraphDatabase is None:
            logger.warning("neo4j Python driver not installed; skipping graph publishing.")
            self.driver = None
            return

        uri = self.uri or os.getenv("ENLITENS_NEO4J_URI")
        user = self.user or os.getenv("ENLITENS_NEO4J_USER")
        password = self.password or os.getenv("ENLITENS_NEO4J_PASSWORD")

        if not uri or not user or not password:
            logger.warning("Neo4j credentials missing; set ENLITENS_NEO4J_URI/USER/PASSWORD.")
            self.driver = None
            return

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("🕸️ Connected to Neo4j at %s", uri)
        except Exception as exc:  # pragma: no cover - connection failure
            logger.error("Unable to connect to Neo4j (%s); graph publishing disabled.", exc)
            self.driver = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def publish_document(self, record: Dict[str, Any]) -> None:
        if not self.driver:
            return

        metadata = record.get("metadata") or {}
        doc_id = record.get("document_id")
        title = metadata.get("title")
        processed_at = metadata.get("processed_at")
        tags = _safe_iter(metadata.get("tags") or [])
        citations = _safe_iter((record.get("extraction") or {}).get("citations"))

        session = self.driver.session()
        try:
            session.execute_write(
                _upsert_document,
                document_id=doc_id,
                title=title,
                processed_at=processed_at,
                model_key=record.get("model_key"),
                checksum=(record.get("source") or {}).get("checksum_sha256"),
            )

            for tag in tags:
                session.execute_write(
                    _link_tag,
                    document_id=doc_id,
                    tag=tag,
                )

            for doi in citations:
                session.execute_write(
                    _link_citation,
                    document_id=doc_id,
                    doi=doi,
                )
        except Exception as exc:  # pragma: no cover - graph error
            logger.error("Neo4j publish failed for %s: %s", doc_id, exc)
        finally:
            session.close()

    def close(self) -> None:
        if self.driver:
            self.driver.close()


# ----------------------------------------------------------------------
# Cypher helpers
# ----------------------------------------------------------------------
def _upsert_document(tx, *, document_id: str, title: Optional[str], processed_at: Optional[str], model_key: Optional[str], checksum: Optional[str]):
    tx.run(
        """
        MERGE (d:Document {document_id: $document_id})
        SET d.title = $title,
            d.processed_at = $processed_at,
            d.model_key = $model_key,
            d.checksum_sha256 = $checksum
        """,
        document_id=document_id,
        title=title,
        processed_at=processed_at,
        model_key=model_key,
        checksum=checksum,
    )


def _link_tag(tx, *, document_id: str, tag: str):
    tx.run(
        """
        MERGE (t:Tag {name: $tag})
        MERGE (d:Document {document_id: $document_id})
        MERGE (d)-[:HAS_TAG]->(t)
        """,
        document_id=document_id,
        tag=tag,
    )


def _link_citation(tx, *, document_id: str, doi: str):
    tx.run(
        """
        MERGE (c:Citation {doi: $doi})
        MERGE (d:Document {document_id: $document_id})
        MERGE (d)-[:CITES]->(c)
        """,
        document_id=document_id,
        doi=doi,
    )


