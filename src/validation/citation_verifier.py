"""Semantic and structural citation verification."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class CitationVerifier:
    """Verify citations using semantic similarity and rule checks."""

    def __init__(self, similarity_threshold: float = 0.55) -> None:
        self.similarity_threshold = similarity_threshold
        self.embedder = SentenceTransformer("BAAI/bge-m3", device="cpu")

    def verify(
        self,
        synthesis_payload: Dict[str, Any],
        quotes: List[Dict[str, Any]],
    ) -> Tuple[bool, List[str]]:
        quote_lookup = {quote.get("citation_id"): quote for quote in quotes}
        issues: List[str] = []

        declared_citations = synthesis_payload.get("source_citations", [])
        for citation in declared_citations:
            citation_id = citation.get("citation_id")
            if citation_id not in quote_lookup:
                issues.append(f"Citation {citation_id} missing from stage-one quotes")

        checks = [
            ("key_findings", "finding_text"),
            ("neuroscientific_concepts", "definition_accessible"),
            ("clinical_applications", "mechanism"),
            ("therapeutic_targets", "expected_outcomes"),
            ("client_presentations", "neural_basis"),
            ("intervention_suggestions", "how_to_implement"),
        ]

        for field, text_key in checks:
            entries = synthesis_payload.get(field, []) or []
            for idx, entry in enumerate(entries):
                cite_ids = entry.get("citations", [])
                if not cite_ids:
                    issues.append(f"{field}[{idx}] missing citations")
                    continue
                summary_text = self._extract_text(entry, text_key)
                for citation_id in cite_ids:
                    quote = quote_lookup.get(citation_id)
                    if not quote:
                        issues.append(f"{field}[{idx}] references unknown citation {citation_id}")
                        continue
                    similarity = self._cosine_similarity(summary_text, quote.get("quote", ""))
                    if similarity < self.similarity_threshold:
                        issues.append(
                            f"{field}[{idx}] citation {citation_id} similarity {similarity:.2f} below threshold"
                        )

        is_valid = not issues
        return is_valid, issues

    def _extract_text(self, entry: Dict[str, Any], key: str) -> str:
        primary = entry.get(key, "")
        if not isinstance(primary, str):
            primary = str(primary)
        additional_keys = [
            "finding_text",
            "relevance_to_enlitens",
            "concept_name",
            "intervention",
            "neural_basis",
            "symptom_description",
        ]
        extras = [entry.get(k, "") for k in additional_keys if isinstance(entry.get(k), str)]
        return " \n".join([primary] + extras)

    def _cosine_similarity(self, text_a: str, text_b: str) -> float:
        if not text_a.strip() or not text_b.strip():
            return 0.0
        embeddings = self.embedder.encode([text_a, text_b], normalize_embeddings=True)
        return float(np.dot(embeddings[0], embeddings[1]))
