"""Context RAG Agent - Enhances content with St. Louis context and retrieval."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base_agent import BaseAgent
from src.retrieval.vector_store import BaseVectorStore, QdrantVectorStore, SearchResult

logger = logging.getLogger(__name__)


class ContextRAGAgent(BaseAgent):
    """Agent specialized in contextual enhancement with retrieval-augmented grounding."""

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        top_k: int = 6,
        max_document_chars: int = 5000,
    ) -> None:
        super().__init__(
            name="ContextRAG",
            role="Contextual Enhancement with RAG",
        )
        self.vector_store = vector_store or QdrantVectorStore()
        self.top_k = top_k
        self.max_document_chars = max_document_chars

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

            return {
                "context_enhanced": bool(retrieval_results),
                "st_louis_relevance": "high" if st_louis_context else "unknown",
                "regional_context": st_louis_context,
                "rag_retrieval": {
                    "query_components": query_components,
                    "top_passages": retrieval_results,
                    "related_documents": related_documents,
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

        query_vectors: List[np.ndarray] = []
        for component_text in query_components.values():
            if not component_text:
                continue
            vector = self.vector_store.embedding_model.encode(component_text, normalize_embeddings=True)
            query_vectors.append(vector)
        combined_vector: Optional[np.ndarray] = None
        if query_vectors:
            combined_vector = np.mean(query_vectors, axis=0)
            combined_vector = combined_vector / np.linalg.norm(combined_vector)

        def register_results(results: List[SearchResult], source: str) -> None:
            for result in results:
                record = aggregated_results.get(result.chunk_id)
                payload = result.payload or {}
                sources = set([source]) if record is None else set(record["sources"])
                sources.add(source)
                entry = {
                    "chunk_id": result.chunk_id,
                    "score": float(result.score),
                    "text": result.text,
                    "document_id": payload.get("document_id"),
                    "source_type": payload.get("source_type"),
                    "agent": payload.get("agent"),
                    "field_path": payload.get("field_path"),
                    "metadata": {k: v for k, v in payload.items() if k != "text"},
                    "sources": sources,
                }
                if record is None or entry["score"] > record["score"]:
                    aggregated_results[result.chunk_id] = entry
                else:
                    record["sources"].add(source)

        for name, text in query_components.items():
            partial_results = self.vector_store.search(text, limit=self.top_k)
            register_results(partial_results, name)

        if combined_vector is not None:
            vector_results = self.vector_store.search(combined_vector, limit=self.top_k)
            register_results(vector_results, "combined")

        normalized_results: List[Dict[str, Any]] = []
        for entry in aggregated_results.values():
            entry["sources"] = sorted(entry["sources"])
            normalized_results.append(entry)

        normalized_results.sort(key=lambda item: item["score"], reverse=True)
        return normalized_results[: self.top_k]
