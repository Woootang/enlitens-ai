"""Hybrid retrieval with dense and sparse fusion."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combine dense cosine similarity with BM25 and reranking."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        dense_limit: int = 50,
        rerank_limit: int = 50,
        final_k: int = 5,
    ) -> None:
        self.vector_store = vector_store
        self.dense_limit = dense_limit
        self.rerank_limit = rerank_limit
        self.final_k = final_k

        self.chunk_lookup: Dict[str, Dict[str, Any]] = {}
        self.corpus_tokens: List[List[str]] = []
        self.bm25: BM25Okapi | None = None
        self.reranker: CrossEncoder | None = None

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return

        for chunk in chunks:
            self.chunk_lookup[chunk["chunk_id"]] = chunk

        corpus_texts = [chunk["text"] for chunk in self.chunk_lookup.values()]
        self.corpus_tokens = [self._tokenize(text) for text in corpus_texts]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        top_k = top_k or self.final_k
        dense_results = self.vector_store.search(query, limit=self.dense_limit)
        sparse_results = self._bm25_search(query, limit=self.dense_limit)

        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
        if not fused:
            return []

        reranked = self._rerank(query, fused[: self.rerank_limit])
        return reranked[:top_k]

    def _bm25_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        if self.bm25 is None or not self.corpus_tokens:
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        scored = sorted(
            [
                (float(score), chunk)
                for score, chunk in zip(scores, self.chunk_lookup.values())
                if score > 0
            ],
            key=lambda item: item[0],
            reverse=True,
        )
        top = scored[:limit]
        return [
            {
                "chunk_id": chunk["chunk_id"],
                "score": score,
                "text": chunk["text"],
                "payload": chunk,
            }
            for score, chunk in top
        ]

    def _reciprocal_rank_fusion(
        self,
        dense: List[Dict[str, Any]],
        sparse: List[Dict[str, Any]],
        constant_k: int = 60,
    ) -> List[Dict[str, Any]]:
        scores: Dict[str, float] = {}
        rank_positions: Dict[str, Dict[str, int]] = {}

        for rank, result in enumerate(dense, start=1):
            chunk_id = str(result["chunk_id"])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (constant_k + rank)
            rank_positions.setdefault(chunk_id, {})["dense"] = rank

        for rank, result in enumerate(sparse, start=1):
            chunk_id = str(result["chunk_id"])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (constant_k + rank)
            rank_positions.setdefault(chunk_id, {})["sparse"] = rank

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        fused_results: List[Dict[str, Any]] = []
        for chunk_id, score in ranked:
            chunk = self.chunk_lookup.get(chunk_id)
            if not chunk:
                continue
            fused_results.append(
                {
                    "chunk_id": chunk_id,
                    "score": score,
                    "text": chunk.get("text", ""),
                    "payload": chunk,
                    "ranks": rank_positions.get(chunk_id, {}),
                }
            )
        return fused_results

    def _rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        reranker = self._ensure_reranker()
        if reranker is None:
            return candidates

        pairs = [[query, candidate.get("text", "")] for candidate in candidates]
        scores = reranker.predict(pairs)
        reranked = [
            {**candidate, "score": float(score)}
            for candidate, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked

    def _ensure_reranker(self) -> CrossEncoder | None:
        if self.reranker is None:
            try:
                self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
            except Exception as exc:
                logger.warning("Failed to load reranker: %s", exc)
                self.reranker = None
        return self.reranker

    def _tokenize(self, text: str) -> List[str]:
        return [token.lower() for token in text.split() if token]

    def get_supporting_chunks(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        return [self.chunk_lookup.get(chunk_id) for chunk_id in chunk_ids if chunk_id in self.chunk_lookup]
