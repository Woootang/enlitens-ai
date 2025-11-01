from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("numpy")
import numpy as np

from src.agents.context_rag_agent import ContextRAGAgent
from src.retrieval.vector_store import BaseVectorStore, SearchResult


class _DummyEmbedder:
    def encode(self, *_: Any, **__: Any) -> np.ndarray:
        return np.ones(8, dtype=float)


class _StubVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.embedding_model = _DummyEmbedder()
        self._calls: List[str] = []

    def upsert(self, chunks: List[Dict[str, Any]]) -> None:  # pragma: no cover - unused in test
        raise NotImplementedError

    def search(
        self,
        query: Any,
        limit: int = 50,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        self._calls.append(str(query))
        return [
            SearchResult(
                chunk_id="newer",
                score=0.4,
                text=(
                    "2023 neuroscience study on polyvagal regulation shows context and system redesign matter"
                ),
                payload={
                    "document_id": "doc-new",
                    "doc_type": "research article",
                    "processing_timestamp": "2023-05-01",
                    "quality_score": 0.9,
                },
            ),
            SearchResult(
                chunk_id="legacy",
                score=0.6,
                text=(
                    "2010 DSM manual describes autism disorder symptoms without context or strengths"
                ),
                payload={
                    "document_id": "doc-old",
                    "doc_type": "diagnostic manual",
                    "processing_timestamp": "2010-01-01",
                },
            ),
        ][:limit]

    def count(self) -> int:  # pragma: no cover - unused
        return 0

    def count_by_document(self, document_id: str) -> int:  # pragma: no cover - unused
        return 0

    def delete_by_document(self, document_id: str) -> None:  # pragma: no cover - unused
        raise NotImplementedError


def test_constitution_prioritises_recent_strengths_sources():
    agent = ContextRAGAgent(
        vector_store=_StubVectorStore(),
        top_k=2,
        max_iterations=1,
    )

    result = agent._run_retrieval({"document_summary": "polyvagal nervous system support"})

    assert result, "Expected retrieval results"
    top_ids = [item["document_id"] for item in result]
    assert top_ids[0] == "doc-new"
    assert result[0]["alignment"]["score"] > result[1]["alignment"]["score"]
    # Ensure combined score integrates alignment penalties
    assert result[0]["combined_score"] > result[1]["combined_score"]
    # Legacy item should still be present but penalized for alignment
    assert result[1]["alignment"]["score"] < 0
