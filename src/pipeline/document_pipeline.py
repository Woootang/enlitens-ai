#!/usr/bin/env python3
"""
High-level PDF processing helpers used by the ingest orchestrator.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from process_pdfs.cache_utils import get_or_create_extraction
from process_pdfs.enrichment import build_enrichment_payload
from process_pdfs.extraction import extract_scientific_content
from src.integrations.gemini_cli_json_assembler import GeminiJSONAssembler
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

DOC_CACHE_ROOT = Path("cache/docling_outputs")


def slugify(value: str, *, max_length: int = 64) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:max_length] or "document"


def compute_checksum(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _default_document_id(pdf_path: Path, checksum: str) -> str:
    stem = slugify(pdf_path.stem)
    short_hash = checksum[:8]
    return f"{stem}-{short_hash}"


def _build_metadata(
    *,
    pdf_path: Path,
    checksum: str,
    docling_payload: Dict[str, Any],
    model_key: str,
) -> Dict[str, Any]:
    processed_at = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    paper_id = docling_payload.get("paper_id") or _default_document_id(pdf_path, checksum)
    metadata = {
        "document_id": paper_id,
        "source_pdf": str(pdf_path),
        "checksum_sha256": checksum,
        "processed_at": processed_at,
        "model_key": model_key,
        "docling_version": docling_payload.get("docling_version"),
    }
    return metadata


def _extract_tags(docling_payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = docling_payload.get("metadata") or {}
    tags = set()
    for key in ("keywords", "subjects", "tags"):
        values = metadata.get(key)
        if isinstance(values, str):
            tags.update(part.strip() for part in values.split(",") if part.strip())
        elif isinstance(values, (list, tuple)):
            tags.update(str(item).strip() for item in values if item)
    return {
        "tags": sorted({slugify(tag) for tag in tags if tag}),
        "title": metadata.get("title"),
        "primary_topic": metadata.get("subject_headings") or metadata.get("primary_topic"),
    }


def _finalize_with_gemini(base_payload: Dict[str, Any]) -> Dict[str, Any]:
    assembler = GeminiJSONAssembler()
    if not assembler.available:
        logger.info("Gemini CLI unavailable; skipping validation for %s", base_payload["document_id"])
        return {**base_payload, "gemini_validated": False}

    # Send FULL document text and all extractions to Gemini CLI
    full_text = base_payload.get("docling", {}).get("verbatim_text", "")
    request_payload = {
        "document_id": base_payload["document_id"],
        "metadata": base_payload.get("docling", {}).get("metadata"),
        "document_text": full_text,  # FULL document text
        "agent_outputs": {
            "extraction": base_payload.get("extraction"),
            "enrichment": base_payload.get("enrichment"),
        },
        "quality": {
            "model": base_payload.get("model_key"),
            "source_pdf": base_payload.get("source", {}).get("pdf_path"),
        },
    }
    validated = assembler.assemble_entry(request_payload)
    if not validated:
        logger.warning("Gemini returned no payload for %s; keeping draft", base_payload["document_id"])
        return {**base_payload, "gemini_validated": False}

    return {
        **base_payload,
        "gemini_validated": True,
        "knowledge_entry": validated,
    }


def process_pdf_document(
    pdf_path: Path,
    *,
    llm_client: LLMClient,
    model_key: str,
    force_extraction: bool = False,
    run_gemini: bool = True,
) -> Dict[str, Any]:
    """
    Run the full ingestion pipeline for a single PDF and return the combined record.
    """
    start_time = time.time()
    logger.info("📥 Starting ingestion for %s", pdf_path.name)

    docling_payload = get_or_create_extraction(
        pdf_path,
        cache_root=DOC_CACHE_ROOT,
        force=force_extraction,
    )

    checksum = compute_checksum(pdf_path)
    metadata = _build_metadata(
        pdf_path=pdf_path,
        checksum=checksum,
        docling_payload=docling_payload,
        model_key=model_key,
    )
    tags_info = _extract_tags(docling_payload)
    metadata.update({k: v for k, v in tags_info.items() if v})

    extraction_payload = extract_scientific_content(
        document_text=docling_payload["verbatim_text"],
        llm_client=llm_client,
        metadata=metadata,
    )
    enrichment_payload = build_enrichment_payload(docling_payload.get("metadata", {}), extraction_payload)

    base_record = {
        "document_id": metadata["document_id"],
        "model_key": model_key,
        "source": {
            "pdf_path": str(pdf_path),
            "checksum_sha256": checksum,
            "original_filename": pdf_path.name,
        },
        "processing": {
            "seconds": time.time() - start_time,
            "processed_at": metadata["processed_at"],
        },
        "metadata": metadata,
        "docling": docling_payload,
        "extraction": extraction_payload,
        "enrichment": enrichment_payload,
    }

    if run_gemini:
        return _finalize_with_gemini(base_record)
    return base_record

