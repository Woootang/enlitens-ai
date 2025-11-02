"""Context RAG Agent - Constitution-aware retrieval orchestration."""
from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .base_agent import BaseAgent
from src.retrieval.vector_store import BaseVectorStore, QdrantVectorStore, SearchResult
from src.utils.enlitens_constitution import EnlitensConstitution
from src.utils.vector_math import ensure_float_list, mean as vector_mean

logger = logging.getLogger(__name__)


class ContextRAGAgent(BaseAgent):
    """Agent specialized in constitution-aware retrieval-augmented grounding."""

    YEAR_PATTERN = re.compile(r"(19|20)\d{2}")

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        top_k: int = 6,
        max_document_chars: int = 5000,
        *,
        alignment_weight: float = 0.4,
        max_iterations: int = 2,
        followup_queries: Optional[Sequence[str]] = None,
        min_combined_score: float = -0.15,
    ) -> None:
        super().__init__(
            name="ContextRAG",
            role="Contextual Enhancement with RAG",
        )
        self.vector_store = vector_store or QdrantVectorStore()
        self.top_k = top_k
        self.max_document_chars = max_document_chars
        self.constitution = EnlitensConstitution()
        self.alignment_weight = alignment_weight
        self.max_iterations = max(1, max_iterations)
        self.followup_queries = list(followup_queries or [])
        self.min_combined_score = min_combined_score

    async def initialize(self) -> bool:
        """Initialize the context RAG agent."""
        try:
            self.is_initialized = True
            logger.info("✅ %s agent initialized (top_k=%d)", self.name, self.top_k)
            return True
        except Exception as exc:
            logger.error("Failed to initialize %s: %s", self.name, exc)
            return False

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance content with contextual retrieval results."""
        try:
            document_text = context.get("document_text", "")
            st_louis_context = context.get("st_louis_context", {})
            client_insights = context.get("client_insights") or {}
            founder_insights = context.get("founder_insights") or {}
            intermediate = context.get("intermediate_results") or {}

            query_components = self._collect_query_components(
                document_text=document_text,
                client_insights=client_insights,
                founder_insights=founder_insights,
                st_louis_context=st_louis_context,
                intermediate=intermediate,
            )

            retrieval_results = self._run_retrieval(query_components)
            related_documents = sorted(
                {
                    result.get("document_id")
                    for result in retrieval_results
                    if result.get("document_id")
                }
            )

            alignment_summary = self._summarize_alignment(retrieval_results)

            return {
                "context_enhanced": bool(retrieval_results),
                "st_louis_relevance": "high" if st_louis_context else "unknown",
                "regional_context": st_louis_context,
                "rag_retrieval": {
                    "query_components": query_components,
                    "top_passages": retrieval_results,
                    "related_documents": related_documents,
                    "alignment_summary": alignment_summary,
                },
            }

        except Exception as exc:
            logger.error("Context RAG failed: %s", exc)
            return {"context_enhanced": False, "error": str(exc)}

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate the context enhancement.

        Be permissive so the pipeline continues even when retrieval is empty.
        """
        return True

    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up %s agent", self.name)

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def _collect_query_components(
        self,
        document_text: str,
        client_insights: Dict[str, Any],
        founder_insights: Dict[str, Any],
        st_louis_context: Dict[str, Any],
        intermediate: Dict[str, Any],
    ) -> Dict[str, str]:
        components: Dict[str, str] = {}

        if document_text:
            components["document_summary"] = document_text[: self.max_document_chars]

        if client_insights:
            components["client_themes"] = json.dumps(client_insights, ensure_ascii=False)

        if founder_insights:
            components["founder_voice"] = json.dumps(founder_insights, ensure_ascii=False)

        if st_louis_context:
            components["regional_context"] = json.dumps(st_louis_context, ensure_ascii=False)

        highlight_keys = [
            "science_extraction",
            "clinical_content",
            "rebellion_framework",
        ]
        highlights: List[str] = []
        for key in highlight_keys:
            payload = intermediate.get(key)
            if payload:
                highlights.append(json.dumps(payload, ensure_ascii=False))
        if highlights:
            components["intermediate_highlights"] = "\n".join(highlights)

        return components

    def _run_retrieval(self, query_components: Dict[str, str]) -> List[Dict[str, Any]]:
        if not query_components:
            return []

        aggregated_results: Dict[str, Dict[str, Any]] = {}

        def register_results(results: List[SearchResult], source: str) -> None:
            for result in results:
                record = aggregated_results.get(result.chunk_id)
                payload = result.payload or {}
                metadata = self.constitution.sanitize_mapping({k: v for k, v in payload.items() if k != "text"})
                sources = set([source]) if record is None else set(record["sources"])
                sources.add(source)
                entry = {
                    "chunk_id": result.chunk_id,
                    "score": float(result.score),
                    "text": result.text,
                    "document_id": metadata.get("document_id"),
                    "source_type": metadata.get("source_type"),
                    "agent": metadata.get("agent"),
                    "field_path": metadata.get("field_path"),
                    "metadata": metadata,
                    "sources": sources,
                }
                if record is None or entry["score"] > record["score"]:
                    aggregated_results[result.chunk_id] = entry
                else:
                    record["sources"].add(source)

        query_vectors: List[List[float]] = []
        for component_name, component_text in query_components.items():
            if not component_text:
                continue
            vector_raw = self.vector_store.embedding_model.encode(
                component_text,
                normalize_embeddings=True,
            )
            query_vectors.append(ensure_float_list(vector_raw))
            partial_results = self.vector_store.search(component_text, limit=self.top_k)
            register_results(partial_results, component_name or "component")

        combined_vector: Optional[List[float]] = None
        if query_vectors:
            averaged_vector = vector_mean(query_vectors)
            norm = math.sqrt(sum(value * value for value in averaged_vector))
            if norm > 0:
                combined_vector = [value / norm for value in averaged_vector]

        if combined_vector is not None:
            vector_results = self.vector_store.search(combined_vector, limit=self.top_k)
            register_results(vector_results, "combined")

        iteration = 1
        while iteration < self.max_iterations:
            scored_results = self._score_results(aggregated_results.values())
            if not self._needs_additional_iteration(scored_results):
                break
            follow_up_queries = self._generate_follow_up_queries(query_components, scored_results, iteration)
            if not follow_up_queries:
                break
            for query in follow_up_queries:
                follow_results = self.vector_store.search(query, limit=self.top_k)
                register_results(follow_results, f"followup_{iteration}")
            iteration += 1

        scored_results = self._score_results(aggregated_results.values())
        filtered_results = [
            result
            for result in scored_results
            if result["combined_score"] >= self.min_combined_score
        ]
        return filtered_results[: self.top_k]

    # ------------------------------------------------------------------
    # Alignment scoring helpers
    # ------------------------------------------------------------------
    def _score_results(self, entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored: List[Dict[str, Any]] = []
        for entry in entries:
            alignment_score, reasons = self._compute_alignment(entry)
            combined = (1.0 - self.alignment_weight) * float(entry.get("score", 0.0)) + self.alignment_weight * alignment_score
            enriched = {
                **entry,
                "alignment": {
                    "score": alignment_score,
                    "reasons": reasons,
                },
                "combined_score": combined,
            }
            scored.append(enriched)

        scored.sort(key=lambda item: item["combined_score"], reverse=True)
        for item in scored:
            item["sources"] = sorted(item.get("sources", []))
        return scored

    def _compute_alignment(self, entry: Dict[str, Any]) -> Tuple[float, List[str]]:
        text = entry.get("text", "") or ""
        metadata = entry.get("metadata") or {}
        reasons: List[str] = []
        alignment = 0.0

        year = self._resolve_year(metadata, text)
        if year:
            if year >= 2020:
                alignment += 0.35
                reasons.append(f"recent_year:{year}")
            elif year >= 2015:
                alignment += 0.2
                reasons.append(f"post2015:{year}")
            else:
                alignment -= 0.35
                reasons.append(f"legacy_year:{year}")
        else:
            reasons.append("year_unknown")

        doc_type = str(metadata.get("doc_type", "")).lower()
        if doc_type:
            if any(token in doc_type for token in ["research", "study", "science", "neuro"]):
                alignment += 0.15
                reasons.append(f"doc_type:{doc_type}")
            if any(token in doc_type for token in ["diagnostic", "manual", "pathology"]):
                alignment -= 0.25
                reasons.append(f"doc_type_penalty:{doc_type}")

        quality_score = metadata.get("quality_score")
        if isinstance(quality_score, (int, float)):
            normalized_quality = max(0.0, min(1.0, float(quality_score)))
            alignment += 0.1 * (normalized_quality - 0.5)
            reasons.append(f"quality:{normalized_quality:.2f}")

        confidence_score = metadata.get("confidence_score")
        if isinstance(confidence_score, (int, float)):
            normalized_confidence = max(0.0, min(1.0, float(confidence_score)))
            alignment += 0.05 * (normalized_confidence - 0.5)
            reasons.append(f"confidence:{normalized_confidence:.2f}")

        if self.constitution.contains_pathology(text) or self.constitution.contains_legacy_reference(text):
            alignment -= 0.4
            reasons.append("pathology_flag")
        else:
            alignment += 0.05
            reasons.append("language_aligned")

        if self.constitution.ensure_keyword_presence(text, self.constitution.CONTEXT_KEYWORDS):
            alignment += 0.05
            reasons.append("contextual_focus")
        if self.constitution.ensure_keyword_presence(text, self.constitution.STRENGTH_KEYWORDS):
            alignment += 0.05
            reasons.append("strengths_focus")
        if self.constitution.ensure_keyword_presence(text, self.constitution.SYSTEM_KEYWORDS):
            alignment += 0.05
            reasons.append("system_accountability")
        if self.constitution.ensure_keyword_presence(text, self.constitution.TRAUMA_KEYWORDS):
            alignment += 0.05
            reasons.append("trauma_informed")

        return alignment, reasons

    def _resolve_year(self, metadata: Dict[str, Any], text: str) -> Optional[int]:
        timestamp = metadata.get("processing_timestamp")
        if isinstance(timestamp, datetime):
            return timestamp.year
        if isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp).year
            except ValueError:
                match = self.YEAR_PATTERN.search(timestamp)
                if match:
                    return int(match.group())

        match = self.YEAR_PATTERN.search(text)
        if match:
            return int(match.group())
        return None

    def _needs_additional_iteration(self, results: List[Dict[str, Any]]) -> bool:
        if not results:
            return True
        if len(results) < self.top_k:
            return True
        high_alignment = sum(1 for item in results[: self.top_k] if item["alignment"]["score"] >= 0.1)
        return high_alignment < max(1, self.top_k // 2)

    def _generate_follow_up_queries(
        self,
        query_components: Dict[str, str],
        scored_results: List[Dict[str, Any]],
        iteration: int,
    ) -> List[str]:
        base_segments = [segment for segment in query_components.values() if segment]
        seeds: List[str] = []

        if base_segments:
            joined = " ".join(base_segments)
            seeds.append(joined[: self.max_document_chars])

        if self.followup_queries:
            expansions = list(self.followup_queries)
        else:
            expansions = [
                "context neuroscience 2023 study",  # ENL-001, ENL-003
                "trauma-informed polyvagal nervous system 2022",  # ENL-006
                "neurodiversity strengths adaptive systems 2021",  # ENL-002, ENL-005, ENL-007
            ]
            if any(result["alignment"]["score"] < 0 for result in scored_results[: self.top_k]):
                expansions.append("neurodiversity affirming research 2024 environment")

        queries: List[str] = []
        for seed in seeds:
            for expansion in expansions:
                queries.append(f"{seed} {expansion}".strip())
        if not queries and expansions:
            queries = list(expansions)

        # Limit to avoid explosion
        limit = max(1, min(3, self.top_k))
        return queries[:limit]

    def _summarize_alignment(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"average_alignment": 0.0, "high_alignment_ratio": 0.0, "count": 0}

        alignment_scores = [item["alignment"]["score"] for item in results]
        combined_scores = [item["combined_score"] for item in results]
        high_alignment = sum(1 for score in alignment_scores if score >= 0.1)
        average_alignment = sum(alignment_scores) / float(len(alignment_scores)) if alignment_scores else 0.0
        average_combined = sum(combined_scores) / float(len(combined_scores)) if combined_scores else 0.0
        return {
            "average_alignment": average_alignment,
            "average_combined_score": average_combined,
            "high_alignment_ratio": high_alignment / float(len(results)),
            "count": len(results),
        }
