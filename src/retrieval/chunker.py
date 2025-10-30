"""Structural and semantic chunking utilities."""
from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a chunked span of a document."""

    chunk_id: str
    text: str
    token_count: int
    start_char: int
    end_char: int
    pages: List[int]
    sections: List[str]
    metadata: Dict[str, Any]


class DocumentChunker:
    """Split markdown documents into overlapping semantic chunks."""

    def __init__(
        self,
        chunk_size_tokens: int = 900,
        chunk_overlap_ratio: float = 0.15,
    ) -> None:
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = max(1, int(chunk_size_tokens * chunk_overlap_ratio))

    def chunk(self, markdown_text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not markdown_text.strip():
            return []

        segments = self._build_segments(markdown_text)
        if not segments:
            return []

        page_map = metadata.get("page_map", [])
        sections = metadata.get("sections", [])
        doi = metadata.get("doi", "")
        source_path = metadata.get("source_path")

        chunks: List[Dict[str, Any]] = []
        idx = 0
        while idx < len(segments):
            token_total = 0
            start_idx = idx
            current_segments: List[Dict[str, Any]] = []

            while idx < len(segments) and (
                token_total + segments[idx]["tokens"] <= self.chunk_size_tokens
                or not current_segments
            ):
                current_segments.append(segments[idx])
                token_total += segments[idx]["tokens"]
                idx += 1

            chunk_start = current_segments[0]["start"]
            chunk_end = current_segments[-1]["end"]
            chunk_text = "\n\n".join(seg["text"] for seg in current_segments)
            chunk_pages = self._resolve_pages(page_map, chunk_start, chunk_end)
            chunk_sections = self._resolve_sections(sections, chunk_pages)

            chunk = Chunk(
                chunk_id=uuid.uuid4().hex,
                text=chunk_text,
                token_count=token_total,
                start_char=chunk_start,
                end_char=chunk_end,
                pages=chunk_pages,
                sections=chunk_sections,
                metadata={
                    "doi": doi,
                    "source_path": source_path,
                },
            )
            chunks.append(chunk.__dict__)

            if idx >= len(segments):
                break

            idx = self._apply_overlap(segments, start_idx, idx)

            if idx <= start_idx:
                idx = start_idx + 1

        logger.debug("Generated %d chunks", len(chunks))
        return chunks

    def _build_segments(self, markdown_text: str) -> List[Dict[str, Any]]:
        raw_segments = markdown_text.split("\n\n")
        cursor = 0
        segments: List[Dict[str, Any]] = []

        for part in raw_segments:
            text = part.strip()
            if not text:
                cursor += len(part) + 2
                continue

            try:
                start_idx = markdown_text.index(part, cursor)
            except ValueError:
                start_idx = cursor
            end_idx = start_idx + len(part)
            cursor = end_idx + 2

            tokens = self._approximate_token_count(text)
            segments.append(
                {
                    "text": text,
                    "tokens": max(tokens, 1),
                    "start": start_idx,
                    "end": end_idx,
                }
            )

        return segments

    def _approximate_token_count(self, text: str) -> int:
        words = [word for word in text.split() if word]
        return max(len(words), math.ceil(len(text) / 4))

    def _resolve_pages(
        self,
        page_map: List[Dict[str, Any]],
        start: int,
        end: int,
    ) -> List[int]:
        pages: List[int] = []
        for page in page_map:
            if page.get("end", 0) >= start and page.get("start", 0) <= end:
                pages.append(page.get("page_number"))
        return sorted(set(pages))

    def _resolve_sections(
        self,
        sections: List[Dict[str, Any]],
        pages: List[int],
    ) -> List[str]:
        if not pages:
            return []
        section_titles = [
            sec.get("title")
            for sec in sections
            if sec.get("page_number") in pages and sec.get("title")
        ]
        return section_titles

    def _apply_overlap(
        self,
        segments: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
    ) -> int:
        tokens_to_keep = self.chunk_overlap_tokens
        if tokens_to_keep <= 0:
            return end_idx

        token_accum = 0
        new_start = end_idx
        for reverse_idx in range(end_idx - 1, start_idx - 1, -1):
            token_accum += segments[reverse_idx]["tokens"]
            if token_accum >= tokens_to_keep:
                new_start = reverse_idx
                break
        return new_start
