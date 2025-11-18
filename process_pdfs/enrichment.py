#!/usr/bin/env python3
"""
Lightweight enrichment helpers for local PDF pipeline.

The comparison workflow augments the raw LLM extraction with external
context (Wikipedia backgrounds, citation lookups, etc.).  Everything is
cached on disk so repeated model runs (MedGemma ↔ Llama) do not hammer
public APIs.
"""
from __future__ import annotations

import json
import logging
import hashlib
import re
import time
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("cache/enrichment")
WIKIPEDIA_CACHE = CACHE_ROOT / "wikipedia"
CROSSREF_CACHE = CACHE_ROOT / "crossref"
SEM_SCHOLAR_CACHE = CACHE_ROOT / "semantic_scholar"


def _cache_path(root: Path, key: str) -> Path:
    key = key.strip()
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")
    if len(slug) > 120:
        slug = slug[:120].rstrip("-")
    filename = f"{slug}-{digest}.json" if slug else f"{digest}.json"
    return root / filename


def _read_cache(root: Path, key: str) -> Optional[Dict]:
    root.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(root, key)
    if not cache_file.exists():
        return None
    try:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("⚠️ Corrupted enrichment cache %s. Regenerating.", cache_file)
        cache_file.unlink(missing_ok=True)
        return None


def _write_cache(root: Path, key: str, payload: Dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(root, key)
    cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sanitize_wikipedia_term(term: str) -> str:
    """
    Sanitize and validate a term for Wikipedia API lookup.
    Returns empty string if term is unsuitable (too long, contains JSON, etc.)
    """
    # Reject JSON-like structures
    if any(char in term for char in ["{", "}", "[", "]", '":']):
        return ""
    
    cleaned = re.sub(r"\s+", " ", term.strip())
    cleaned = re.sub(r"[\"'`]", "", cleaned)
    cleaned = cleaned.strip(".,;:()[]{}").strip()
    
    # Reject overly long terms (likely full sentences)
    if len(cleaned) > 80:
        return ""
    
    # Reject terms with too many words (likely sentences)
    word_count = len(cleaned.split())
    if word_count > 8:
        return ""
    
    if not cleaned:
        return ""
    
    return quote(cleaned.replace(" ", "_"), safe="_")


def fetch_wikipedia_summary(term: str) -> Optional[Dict]:
    """
    Fetch a concise Wikipedia summary for a scientific term.
    """
    if not term:
        return None

    cached = _read_cache(WIKIPEDIA_CACHE, term)
    if cached:
        return cached

    slug = _sanitize_wikipedia_term(term)
    if not slug:
        return None

    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            if response.status_code == 404:
                logger.info("No Wikipedia article for %s", term)
                return None
            response.raise_for_status()
            payload = response.json()
            summary = {
                "term": term,
                "title": payload.get("title"),
                "description": payload.get("description"),
                "extract": payload.get("extract"),
                "url": payload.get("content_urls", {}).get("desktop", {}).get("page"),
            }
            _write_cache(WIKIPEDIA_CACHE, term, summary)
            return summary
    except Exception as exc:
        logger.info("Wikipedia lookup failed for %s: %s", term, exc)
        return None


def fetch_crossref_metadata(doi: str) -> Optional[Dict]:
    """
    Retrieve citation metadata from Crossref.
    """
    if not doi:
        return None

    cached = _read_cache(CROSSREF_CACHE, doi)
    if cached:
        return cached

    url = f"https://api.crossref.org/works/{doi}"
    headers = {
        "User-Agent": "EnlitensAI/1.0 (mailto:tech@enlitens.ai)",
    }
    try:
        with httpx.Client(timeout=10.0, headers=headers) as client:
            backoff = 0.5
            for attempt in range(3):
                response = client.get(url)
                if response.status_code == 404:
                    logger.info("Crossref did not find DOI %s", doi)
                    return None
                if response.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                response.raise_for_status()
                break
            payload = response.json().get("message", {})
            metadata = {
                "doi": doi,
                "title": payload.get("title", [""])[0] if payload.get("title") else "",
                "journal": payload.get("container-title", [""])[0] if payload.get("container-title") else "",
                "publisher": payload.get("publisher"),
                "published": payload.get("published-print") or payload.get("published-online"),
                "author": payload.get("author"),
            }
            _write_cache(CROSSREF_CACHE, doi, metadata)
            return metadata
    except Exception as exc:
        logger.info("Crossref lookup failed for %s: %s", doi, exc)
        return None


def fetch_semantic_scholar_metadata(doi: str) -> Optional[Dict]:
    """
    Fetch enriched paper metadata from Semantic Scholar (if available).
    """
    if not doi:
        return None

    cached = _read_cache(SEM_SCHOLAR_CACHE, doi)
    if cached:
        return cached

    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    params = {"fields": "title,abstract,year,authors,citationCount,url"}
    try:
        with httpx.Client(timeout=10.0) as client:
            backoff = 1.0
            for attempt in range(3):
                response = client.get(url, params=params)
                if response.status_code == 404:
                    return None
                if response.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                response.raise_for_status()
                break
            payload = response.json()
            metadata = {
                "doi": doi,
                "title": payload.get("title"),
                "abstract": payload.get("abstract"),
                "year": payload.get("year"),
                "authors": payload.get("authors"),
                "citation_count": payload.get("citationCount"),
                "url": payload.get("url"),
            }
            _write_cache(SEM_SCHOLAR_CACHE, doi, metadata)
            return metadata
    except Exception as exc:
        logger.info("Semantic Scholar lookup failed for %s: %s", doi, exc)
        return None


def build_enrichment_payload(metadata: Dict, extraction: Dict) -> Dict:
    """
    Build an enrichment bundle covering key terms and citations.
    """
    enrichment: Dict[str, Dict] = {"wikipedia": {}, "citations": {}}

    # Wikipedia summaries for key phrases
    title = metadata.get("title") or extraction.get("background", "")[:120]
    if title:
        summary = fetch_wikipedia_summary(title)
        if summary:
            enrichment["wikipedia"]["title"] = summary

    # Additional terms (methods, findings first sentences)
    for section_key in ("methods", "findings", "limitations"):
        text = extraction.get(section_key) or ""
        if not text:
            continue
        candidate = text.split(".")[0].strip()
        if candidate and len(candidate.split()) > 3:
            summary = fetch_wikipedia_summary(candidate)
            if summary:
                enrichment["wikipedia"][section_key] = summary

    # Citation metadata
    for doi in extraction.get("citations", []):
        doi = str(doi).strip()
        if not doi:
            continue
        crossref = fetch_crossref_metadata(doi)
        sem_scholar = fetch_semantic_scholar_metadata(doi)
        enrichment["citations"][doi] = {
            "crossref": crossref,
            "semantic_scholar": sem_scholar,
        }

    return enrichment


