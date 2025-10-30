"""Qdrant persistence with local fallbacks."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Persist chunks and metadata while supporting dense retrieval."""

    def __init__(
        self,
        collection_name: str = "enlitens_chunks",
        host: Optional[str] = None,
        port: Optional[int] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
        self.embedding_dim = int(self.embedding_model.get_sentence_embedding_dimension())

        self.client: Optional[QdrantClient] = None
        self.local_store: Dict[str, Dict[str, Any]] = {}

        if url is None and host is None:
            url = os.getenv("QDRANT_URL")
        if api_key is None:
            api_key = os.getenv("QDRANT_API_KEY")
        if host is None:
            host = os.getenv("QDRANT_HOST", "localhost")
        if port is None:
            try:
                port = int(os.getenv("QDRANT_PORT", "6333"))
            except ValueError:
                port = 6333

        try:
            if url:
                self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
            else:
                self.client = QdrantClient(
                    host=host,
                    port=port,
                    api_key=api_key,
                    prefer_grpc=prefer_grpc,
                )
            self._ensure_collection()
            logger.info("Connected to Qdrant collection %s", self.collection_name)
        except Exception as exc:
            self.client = None
            logger.warning("Falling back to in-memory vector store: %s", exc)

    def _ensure_collection(self) -> None:
        if self.client is None:
            return

        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            logger.info("Creating Qdrant collection %s", self.collection_name)
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(size=self.embedding_dim, distance=qmodels.Distance.COSINE),
            )

    def upsert(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return

        embeddings = self.embedding_model.encode(
            [chunk["text"] for chunk in chunks],
            normalize_embeddings=True,
        )

        if self.client is not None:
            points = [
                qmodels.PointStruct(
                    id=chunk["chunk_id"],
                    vector=embedding.tolist(),
                    payload={**chunk, **chunk.get("metadata", {})},
                )
                for chunk, embedding in zip(chunks, embeddings)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)

        for chunk, embedding in zip(chunks, embeddings):
            self.local_store[chunk["chunk_id"]] = {
                "chunk": chunk,
                "embedding": embedding,
            }

    def search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        query_vector = self.embedding_model.encode(query, normalize_embeddings=True)

        if self.client is not None:
            try:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector.tolist(),
                    limit=limit,
                    with_payload=True,
                )
                return [
                    {
                        "chunk_id": point.id,
                        "score": point.score,
                        "payload": point.payload,
                        "text": point.payload.get("text", ""),
                    }
                    for point in results
                ]
            except Exception as exc:
                logger.warning("Qdrant search failed, using local fallback: %s", exc)

        return self._local_search(query_vector, limit)

    def _local_search(self, query_vector: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        if not self.local_store:
            return []

        scores = []
        for entry in self.local_store.values():
            chunk = entry["chunk"]
            embedding = entry["embedding"]
            score = float(np.dot(query_vector, embedding))
            scores.append((score, chunk))

        scores.sort(key=lambda item: item[0], reverse=True)
        top = scores[:limit]
        return [
            {
                "chunk_id": chunk["chunk_id"],
                "score": score,
                "payload": {**chunk, **chunk.get("metadata", {})},
                "text": chunk["text"],
            }
            for score, chunk in top
        ]

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        return [entry["chunk"] for entry in self.local_store.values()]
