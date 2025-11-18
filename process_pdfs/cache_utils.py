#!/usr/bin/env python3
"""
Cache utilities for Docling extraction results.

The local comparison pipeline needs to reuse heavy PDF extraction outputs
across multiple model runs (MedGemma vs Llama).  This module provides a
simple JSON cache keyed by the paper identifier / SHA so the ingestion
stage only runs once per document.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from process_pdfs.ingestion import process_pdf

logger = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = Path("cache/docling_outputs")


def _cache_path(cache_root: Path, paper_id: str) -> Path:
    safe_id = paper_id.replace("/", "_")
    return cache_root / f"{safe_id}.json"


def load_cached_extraction(
    pdf_path: Path,
    paper_id: Optional[str] = None,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> Optional[Dict]:
    """
    Load a cached Docling extraction if available.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    target_id = paper_id or pdf_path.stem
    cache_file = _cache_path(cache_root, target_id)
    if not cache_file.exists():
        return None

    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        logger.info("📁 Loaded cached Docling extraction for %s", target_id)
        return payload
    except json.JSONDecodeError:
        logger.warning("⚠️ Cache file %s was corrupted. Regenerating.", cache_file)
        cache_file.unlink(missing_ok=True)
        return None


def cache_extraction_result(
    result: Dict,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> None:
    """
    Persist Docling extraction payload to disk for reuse.
    """
    paper_id = result.get("paper_id") or "unknown"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(cache_root, paper_id)
    cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("💾 Cached Docling extraction for %s -> %s", paper_id, cache_file)


def get_or_create_extraction(
    pdf_path: Path,
    paper_id: Optional[str] = None,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    force: bool = False,
) -> Dict:
    """
    Return a Docling extraction result, optionally reusing a cached version.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    target_id = paper_id or Path(pdf_path).stem

    if not force:
        cached = load_cached_extraction(Path(pdf_path), target_id, cache_root)
        if cached:
            return cached

    logger.info("🛠️ Running Docling extraction for %s", pdf_path)
    extraction = process_pdf(Path(pdf_path), paper_id=target_id)
    cache_extraction_result(extraction, cache_root)
    return extraction


