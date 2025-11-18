"""
Offline data profiling utilities for Enlitens AI.

These helpers provide lightweight summaries of heterogeneous inputs so that
planning and verification agents have structured context about what data is
available before they begin reasoning.  All computation is deterministic and
keeps PHI on the local machine.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class DataProfiler:
    """Generate lightweight deterministic profiles for common Enlitens inputs."""

    SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")
    TOKEN_REGEX = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")

    def profile_text(
        self,
        label: str,
        text: str,
        *,
        max_preview_chars: int = 600,
        top_term_count: int = 8,
    ) -> Dict[str, Any]:
        """
        Produce a compact summary for a text document.

        The summary includes word and character counts, approximate reading time
        and the most frequent keywords (filtered for stopword-like tokens).  A
        short preview snippet is also provided for quick inspection in logs or
        dashboards.
        """
        safe_text = text or ""
        char_count = len(safe_text)
        tokens = self.TOKEN_REGEX.findall(safe_text.lower())
        word_count = len(tokens)

        # Reading time approximation (200 words / minute default)
        reading_minutes = word_count / 200 if word_count else 0

        token_counts = Counter(tokens)
        top_terms = [
            term
            for term, _ in token_counts.most_common(top_term_count * 2)
            if len(term) > 3
        ]
        # Deduplicate while preserving order and limit to requested count
        seen: set[str] = set()
        keywords: List[str] = []
        for term in top_terms:
            if term not in seen:
                seen.add(term)
                keywords.append(term)
            if len(keywords) == top_term_count:
                break

        preview = safe_text[:max_preview_chars].strip()
        if len(safe_text) > max_preview_chars:
            preview += "…"

        sentences = self.SENTENCE_SPLIT_REGEX.split(safe_text.strip())
        lead_sentences = " ".join(sentences[:3]).strip()

        return {
            "label": label,
            "type": "text",
            "character_count": char_count,
            "word_count": word_count,
            "approx_minutes_to_read": round(reading_minutes, 2),
            "top_terms": keywords,
            "preview": preview,
            "lead_sentences": lead_sentences,
        }

    def profile_entities(self, entities: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Summarise extracted entity buckets (diseases, genes, etc.).

        Each bucket is represented by its size and a small sample of values for
        quick visual inspection.
        """
        summary: Dict[str, Any] = {"type": "entities", "buckets": {}}

        for bucket, items in (entities or {}).items():
            if not isinstance(items, Iterable):
                continue
            normalized: List[str] = []
            for item in items:
                if isinstance(item, str):
                    normalized.append(item)
                elif isinstance(item, dict):
                    text_value = item.get("text") or item.get("name")
                    if text_value:
                        normalized.append(str(text_value))
                else:
                    normalized.append(str(item))

            summary["buckets"][bucket] = {
                "count": len(normalized),
                "sample": normalized[:5],
            }

        summary["total_entities"] = sum(
            bucket_data["count"] for bucket_data in summary["buckets"].values()
        )
        return summary

    def profile_persona_catalog(
        self, personas: Optional[Iterable[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Provide aggregate counts for the persona repository so planning agents
        understand coverage before selection.
        """
        personas_list = list(personas or [])
        demographics_counter: Counter[str] = Counter()
        locality_counter: Counter[str] = Counter()
        diagnosis_counter: Counter[str] = Counter()

        for persona in personas_list:
            identity = persona.get("identity_demographics", {})
            demographics_counter[identity.get("age_range", "Unknown")] += 1
            locality_counter[identity.get("locality", "Unknown")] += 1

            neuro = persona.get("neurodivergence_mental_health", {})
            for field in ("formal_diagnoses", "self_identified_traits"):
                values = neuro.get(field, [])
                if isinstance(values, list):
                    for val in values:
                        diagnosis_counter[str(val)] += 1

        def top_counts(counter: Counter[str], limit: int = 5) -> List[Dict[str, Any]]:
            return [
                {"value": value, "count": count}
                for value, count in counter.most_common(limit)
            ]

        return {
            "type": "persona_catalog",
            "total_personas": len(personas_list),
            "top_age_ranges": top_counts(demographics_counter),
            "top_localities": top_counts(locality_counter),
            "top_diagnoses": top_counts(diagnosis_counter),
        }

    def profile_file(self, path: Path) -> Dict[str, Any]:
        """
        Convenience helper to profile a text-based file on disk.
        Falls back to empty profile if file cannot be read safely.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            text = ""
        return self.profile_text(path.name, text)


__all__ = ["DataProfiler"]

