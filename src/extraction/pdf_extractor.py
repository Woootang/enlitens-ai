"""Hybrid PDF extraction utilities for Enlitens."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF

try:
    from pymupdf4llm import to_markdown as pymupdf_to_markdown
except Exception:  # pragma: no cover - optional dependency failures
    pymupdf_to_markdown = None

logger = logging.getLogger(__name__)

DOI_REGEX = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")


class HybridExtractor:
    """Extract PDF content to markdown with rich metadata."""

    def __init__(self, cache_dir: str | os.PathLike[str]):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, pdf_path: str) -> Dict[str, Any]:
        pdf_path = Path(pdf_path).resolve()
        try:
            stat_result = pdf_path.stat()
            cache_fingerprint = f"{pdf_path}:{stat_result.st_mtime_ns}:{stat_result.st_size}"
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        cache_key = hashlib.md5(cache_fingerprint.encode("utf-8")).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            logger.debug("Loading cached extraction for %s", pdf_path)
            with cache_file.open("r", encoding="utf-8") as fh:
                cached = json.load(fh)
            # Ensure the current path is reflected even when cached under a
            # different working directory.
            metadata = cached.setdefault("metadata", {})
            metadata["source_path"] = str(pdf_path)
            cached.setdefault("title", metadata.get("title", ""))
            cached.setdefault("doi", metadata.get("doi", ""))
            cached.setdefault("page_map", metadata.get("page_map", []))
            cached.setdefault("sections", metadata.get("sections", []))
            return cached

        logger.info("Extracting PDF via PyMuPDF4LLM: %s", pdf_path)
        markdown, metadata = self._extract_with_pymupdf(pdf_path)

        if not markdown.strip():
            logger.warning("PyMuPDF4LLM produced empty output, falling back to Marker")
            markdown, metadata = self._extract_with_marker(pdf_path)

        metadata.setdefault("title", Path(pdf_path).stem)
        metadata["source_path"] = str(pdf_path)

        extraction = {
            "full_text": markdown,
            "markdown_path": str(self._write_markdown(cache_key, markdown)),
            "metadata": metadata,
            "title": metadata.get("title", ""),
            "doi": metadata.get("doi", ""),
            "page_map": metadata.get("page_map", []),
            "sections": metadata.get("sections", []),
        }

        with cache_file.open("w", encoding="utf-8") as fh:
            json.dump(extraction, fh, ensure_ascii=False, indent=2)

        return extraction

    def _write_markdown(self, cache_key: str, markdown: str) -> Path:
        md_path = self.cache_dir / f"{cache_key}.md"
        with md_path.open("w", encoding="utf-8") as fh:
            fh.write(markdown)
        return md_path

    def _extract_with_pymupdf(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        markdown_chunks: List[str] = []
        page_map: List[Dict[str, Any]] = []
        sections: List[Dict[str, Any]] = []

        backend = "pymupdf4llm" if pymupdf_to_markdown is not None else "pymupdf"

        pdf_path_str = str(pdf_path)

        with fitz.open(pdf_path_str) as doc:
            try:
                title = (doc.metadata or {}).get("title") or Path(pdf_path_str).stem
            except Exception:
                title = Path(pdf_path_str).stem

            cursor = 0
            for index, page in enumerate(doc):
                page_text = self._page_to_markdown(doc, page, pdf_path_str)
                if not page_text.strip():
                    continue

                start = cursor
                cursor += len(page_text) + 2  # account for join separators
                page_map.append(
                    {
                        "page_number": index + 1,
                        "start": start,
                        "end": cursor,
                        "approx_char_length": len(page_text),
                    }
                )
                markdown_chunks.append(page_text)
                sections.extend(
                    self._extract_sections(page_text, index + 1)
                )

        markdown = "\n\n".join(markdown_chunks)
        doi_match = DOI_REGEX.search(markdown)

        metadata = {
            "title": title,
            "doi": doi_match.group(0) if doi_match else "",
            "page_map": page_map,
            "sections": sections,
            "source_path": pdf_path_str,
            "extraction_backend": backend,
        }

        return markdown, metadata

    def _extract_with_marker(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        temp_dir = self.cache_dir / "marker"
        temp_dir.mkdir(exist_ok=True)
        output_md = temp_dir / (Path(pdf_path).stem + ".md")

        try:
            subprocess.run(
                [
                    "marker",
                    "convert",
                    str(pdf_path),
                    "--output",
                    str(output_md),
                    "--markdown",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:  # pragma: no cover
            logger.error("Marker conversion failed: %s", exc)
            return self._extract_with_pymupdf_fallback(pdf_path)

        try:
            with output_md.open("r", encoding="utf-8") as fh:
                markdown = fh.read()
        except FileNotFoundError:  # pragma: no cover
            logger.error("Marker output not found, falling back to PyMuPDF fallback")
            return self._extract_with_pymupdf_fallback(pdf_path)

        return markdown, {
            "title": Path(pdf_path).stem,
            "doi": self._find_doi(markdown),
            "page_map": [],
            "sections": [],
            "source_path": str(pdf_path),
            "extraction_backend": "marker",
        }

    def _extract_with_pymupdf_fallback(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        with fitz.open(str(pdf_path)) as doc:
            markdown_chunks = [page.get_text("markdown") for page in doc]
        markdown = "\n\n".join(markdown_chunks)
        return markdown, {
            "title": Path(pdf_path).stem,
            "doi": self._find_doi(markdown),
            "page_map": [],
            "sections": [],
            "source_path": str(pdf_path),
            "extraction_backend": "pymupdf",
        }

    def _page_to_markdown(self, doc: fitz.Document, page: fitz.Page, pdf_path: str) -> str:
        if pymupdf_to_markdown is not None:
            try:
                return pymupdf_to_markdown(doc, page_numbers=[page.number])
            except TypeError:
                try:
                    return pymupdf_to_markdown(pdf_path, page_numbers=[page.number])
                except Exception:
                    pass
            except Exception as exc:  # pragma: no cover
                logger.debug("pymupdf4llm.to_markdown failed on page: %s", exc)

        return page.get_text("markdown")

    def _extract_sections(self, page_text: str, page_number: int) -> List[Dict[str, Any]]:
        sections: List[Dict[str, Any]] = []
        for line in page_text.splitlines():
            stripped = line.strip().strip("# ")
            if not stripped:
                continue
            if self._looks_like_heading(stripped):
                sections.append(
                    {
                        "title": stripped,
                        "page_number": page_number,
                    }
                )
        return sections

    def _looks_like_heading(self, text: str) -> bool:
        if len(text.split()) > 15:
            return False
        if text.isupper() and len(text) > 8:
            return True
        if re.match(r"^\d+(?:\.\d+)*\s+", text):
            return True
        return text.endswith(":")

    def _find_doi(self, text: str) -> str:
        match = DOI_REGEX.search(text)
        return match.group(0) if match else ""
