#!/usr/bin/env python3
"""
Shared embedding utilities for persistence layers.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Sequence

import numpy as np

from src.retrieval.vector_store import _resolve_embedding_model


class EmbeddingProvider:
    """
    Lightweight wrapper around the project-wide embedding model.
    Lazily loads the model and exposes helpers to embed batches or single texts.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self._model = _resolve_embedding_model(model_name, device)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    @staticmethod
    def _to_list(embeddings: np.ndarray) -> List[List[float]]:
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings.astype(np.float32).tolist()

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        filtered = [text if text is not None else "" for text in texts]
        if not filtered:
            return []
        embeddings = self._model.encode(filtered, normalize_embeddings=True)
        if isinstance(embeddings, np.ndarray):
            return self._to_list(embeddings)
        return [[float(value) for value in vector] for vector in embeddings]

    def embed_one(self, text: str) -> List[float] | None:
        results = self.embed([text])
        return results[0] if results else None


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Provide a singleton embedding provider to avoid reloading models."""
    return EmbeddingProvider()

