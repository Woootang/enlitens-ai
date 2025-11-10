"""Vector store integrations with Qdrant, Chroma, and in-memory fallbacks."""
from __future__ import annotations

import hashlib
import logging
import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # Optional dependency - only required for persistent usage
    import chromadb
    from chromadb.api import ClientAPI as ChromaClient
    from chromadb.errors import InvalidCollectionException
except Exception:  # pragma: no cover - chromadb not always installed in tests
    chromadb = None
    ChromaClient = None  # type: ignore
    InvalidCollectionException = Exception  # type: ignore

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

try:  # sentence_transformers is optional in lightweight environments
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except Exception:  # pragma: no cover - used in testing environments without torch
    _SentenceTransformer = None


class HashingSentenceTransformer:
    """Deterministic hashing-based embedding model used as a lightweight fallback."""

    def __init__(self, dimension: Optional[int] = None) -> None:
        env_dim = os.getenv("ENLITENS_HASH_EMBED_DIM")
        resolved_dim = dimension
        if resolved_dim is None and env_dim:
            try:
                resolved_dim = int(env_dim)
            except ValueError:
                resolved_dim = None
        self.dimension = resolved_dim or 384

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        sentences: Union[str, Sequence[str]],
        normalize_embeddings: bool = True,
        **_: Any,
    ) -> np.ndarray:
        single_input = isinstance(sentences, str)
        if single_input:
            to_encode = [sentences]
        else:
            to_encode = list(sentences)

        embeddings: List[np.ndarray] = []
        for sentence in to_encode:
            text = sentence or ""
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            byte_values = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            repeats = int(math.ceil(self.dimension / float(len(byte_values))))
            tiled = np.tile(byte_values, repeats)[: self.dimension]
            embedding = tiled / 255.0
            embeddings.append(embedding)

        matrix = np.vstack(embeddings).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        if single_input:
            return matrix[0]
        return matrix

logger = logging.getLogger(__name__)


def _resolve_embedding_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Any:
    """Create a shared embedding model with graceful fallbacks for tests."""

    resolved_name = model_name or os.getenv("ENLITENS_EMBED_MODEL", "BAAI/bge-m3")
    resolved_device = device or os.getenv("ENLITENS_EMBED_DEVICE", "cpu")

    if resolved_name in {"hash", "debug-hashing", "hashing"}:
        logger.info("Using hashing embedding model '%s'", resolved_name)
        return HashingSentenceTransformer()

    if _SentenceTransformer is None:
        logger.warning(
            "sentence-transformers is unavailable; falling back to hashing embeddings",
        )
        return HashingSentenceTransformer()

    try:
        logger.debug("Loading sentence transformer '%s' on device '%s'", resolved_name, resolved_device)
        return _SentenceTransformer(resolved_name, device=resolved_device)
    except Exception as exc:  # pragma: no cover - guard against missing dependencies
        logger.warning(
            "Failed to load sentence transformer '%s' (%s); using hashing fallback",
            resolved_name,
            exc,
        )
        return HashingSentenceTransformer()


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _build_filter_condition(metadata_filter: Optional[Dict[str, Any]]) -> Optional[qmodels.Filter]:
    if not metadata_filter:
        return None

    conditions: List[qmodels.FieldCondition] = []
    for key, value in metadata_filter.items():
        if value is None:
            continue
        conditions.append(
            qmodels.FieldCondition(
                key=str(key),
                match=qmodels.MatchValue(value=value),
            )
        )

    if not conditions:
        return None

    return qmodels.Filter(must=conditions)


@dataclass
class SearchResult:
    """Structured retrieval result from a vector search."""

    chunk_id: str
    score: float
    text: str
    payload: Dict[str, Any]


class BaseVectorStore:
    """Common interface for vector store implementations."""

    embedding_model: Any

    def upsert(self, chunks: List[Dict[str, Any]]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def search(
        self,
        query: Union[str, Sequence[float], np.ndarray],
        limit: int = 50,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:  # pragma: no cover - interface
        raise NotImplementedError

    def count(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def count_by_document(self, document_id: str) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def delete_by_document(self, document_id: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class QdrantVectorStore(BaseVectorStore):
    """Persist chunks and metadata while supporting dense retrieval."""

    def __init__(
        self,
        collection_name: str = "enlitens_chunks",
        host: Optional[str] = None,
        port: Optional[int] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        embedding_model_name: Optional[str] = None,
        embedding_device: Optional[str] = None,
    ) -> None:
        self.collection_name = collection_name
        self.embedding_model = _resolve_embedding_model(embedding_model_name, embedding_device)
        self.embedding_dim = int(self.embedding_model.get_sentence_embedding_dimension())

        self.client: Optional[QdrantClient] = None
        self.local_store: Dict[str, Dict[str, Any]] = {}

        # Check environment variables
        env_url = os.getenv("QDRANT_URL")
        env_host = os.getenv("QDRANT_HOST")
        env_port = os.getenv("QDRANT_PORT")
        
        if url is None and host is None and env_url:
            url = env_url
        if api_key is None:
            api_key = os.getenv("QDRANT_API_KEY")

        try:
            # Priority 1: Explicit URL
            if url:
                self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
                logger.info("✅ Connected to Qdrant URL: %s", url)
            # Priority 2: Explicit host/port OR environment variables set
            elif host or env_host:
                final_host = host or env_host or "localhost"
                final_port = port
                if final_port is None:
                    try:
                        final_port = int(env_port) if env_port else 6333
                    except ValueError:
                        final_port = 6333
                self.client = QdrantClient(
                    host=final_host,
                    port=final_port,
                    api_key=api_key,
                    prefer_grpc=prefer_grpc,
                )
                logger.info("✅ Connected to Qdrant server: %s:%s", final_host, final_port)
            # Priority 3: Default to local file storage (no server needed!)
            else:
                local_path = os.path.join(os.getcwd(), "qdrant_storage")
                self._ensure_local_qdrant_path(local_path)
                try:
                    self.client = QdrantClient(path=local_path)
                except Exception as local_exc:
                    message = str(local_exc).lower()
                    if "already accessed" in message or "lock" in message:
                        self._cleanup_qdrant_lock(local_path)
                        self.client = QdrantClient(path=local_path)
                        logger.info("♻️ Cleared stale Qdrant lock and re-opened storage: %s", local_path)
                    else:
                        raise
                logger.info("✅ Using local Qdrant storage: %s", local_path)
             
            self._ensure_collection()
            logger.info("✅ Qdrant collection '%s' ready", self.collection_name)
        except Exception as exc:
            self.client = None
            logger.warning("⚠️ Falling back to in-memory vector store: %s", exc)

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
                "embedding": np.array(embedding),
            }

    def search(
        self,
        query: Union[str, Sequence[float], np.ndarray],
        limit: int = 50,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if isinstance(query, (list, tuple)):
            query_vector = np.array(query, dtype=np.float32)
        elif isinstance(query, np.ndarray):
            query_vector = query
        else:
            if not str(query).strip():
                return []
            query_vector = self.embedding_model.encode(str(query), normalize_embeddings=True)

        query_vector = _normalize_vector(query_vector)

        if self.client is not None:
            try:
                filter_condition = _build_filter_condition(metadata_filter)
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector.tolist(),
                    limit=limit,
                    with_payload=True,
                    query_filter=filter_condition,
                )
                return [
                    SearchResult(
                        chunk_id=str(point.id),
                        score=float(point.score),
                        payload=point.payload or {},
                        text=(point.payload or {}).get("text", ""),
                    )
                    for point in results
                ]
            except Exception as exc:
                logger.warning("Qdrant search failed, using local fallback: %s", exc)

        return self._local_search(query_vector, limit, metadata_filter)

    def _local_search(
        self,
        query_vector: np.ndarray,
        limit: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if not self.local_store:
            return []

        scores: List[Tuple[float, Dict[str, Any]]] = []
        for entry in self.local_store.values():
            chunk = entry["chunk"]
            if metadata_filter and not _metadata_matches(chunk, metadata_filter):
                continue
            embedding = entry["embedding"]
            score = float(np.dot(query_vector, embedding))
            scores.append((score, chunk))

        scores.sort(key=lambda item: item[0], reverse=True)
        top = scores[:limit]
        return [
            SearchResult(
                chunk_id=str(chunk["chunk_id"]),
                score=score,
                payload={**chunk, **chunk.get("metadata", {})},
                text=chunk.get("text", ""),
            )
            for score, chunk in top
        ]

    def count(self) -> int:
        if self.client is not None:
            try:
                response = self.client.count(collection_name=self.collection_name)
                return int(response.count)
            except Exception as exc:
                logger.warning("Failed to get Qdrant count, falling back to local store: %s", exc)
        return len(self.local_store)

    def count_by_document(self, document_id: str) -> int:
        if self.client is not None:
            try:
                response = self.client.count(
                    collection_name=self.collection_name,
                    count_filter=_build_filter_condition({"document_id": document_id}),
                )
                return int(response.count)
            except Exception as exc:
                logger.debug("Failed document-level count via Qdrant: %s", exc)

        return sum(
            1
            for entry in self.local_store.values()
            if entry["chunk"].get("metadata", {}).get("document_id") == document_id
        )

    def delete_by_document(self, document_id: str) -> None:
        if self.client is not None:
            try:
                filter_condition = _build_filter_condition({"document_id": document_id})
                if filter_condition:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=qmodels.FilterSelector(filter=filter_condition),
                    )
            except Exception as exc:
                logger.warning("Failed to delete document %s from Qdrant: %s", document_id, exc)

        keys_to_remove = [
            key
            for key, entry in self.local_store.items()
            if entry["chunk"].get("metadata", {}).get("document_id") == document_id
        ]
        for key in keys_to_remove:
            self.local_store.pop(key, None)

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        return [entry["chunk"] for entry in self.local_store.values()]

    @staticmethod
    def _ensure_local_qdrant_path(local_path: str) -> None:
        Path(local_path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _cleanup_qdrant_lock(local_path: str) -> None:
        lock_path = Path(local_path) / ".lock"
        if lock_path.exists():
            try:
                lock_path.unlink()
                logger.warning("🧹 Removed stale Qdrant lock file at %s", lock_path)
            except OSError as exc:
                logger.warning("⚠️ Unable to remove Qdrant lock file %s: %s", lock_path, exc)


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB persistence backend with shared interface."""

    def __init__(
        self,
        collection_name: str = "enlitens_chunks",
        persist_directory: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        embedding_device: Optional[str] = None,
    ) -> None:
        if chromadb is None:  # pragma: no cover - import guard
            raise ImportError("chromadb is required for ChromaVectorStore")

        self.embedding_model = _resolve_embedding_model(embedding_model_name, embedding_device)
        self.collection_name = collection_name
        persist_path = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

        self.client: ChromaClient = chromadb.PersistentClient(path=persist_path)
        try:
            self.collection = self.client.get_collection(collection_name)
        except InvalidCollectionException:
            logger.info("Creating Chroma collection %s at %s", collection_name, persist_path)
            self.collection = self.client.create_collection(collection_name)

    def upsert(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return

        embeddings = self.embedding_model.encode(
            [chunk["text"] for chunk in chunks],
            normalize_embeddings=True,
        )

        ids = [str(chunk["chunk_id"]) for chunk in chunks]
        metadatas = [{**chunk, **chunk.get("metadata", {})} for chunk in chunks]
        documents = [chunk.get("text", "") for chunk in chunks]

        self.collection.upsert(
            ids=ids,
            embeddings=[embedding.tolist() for embedding in embeddings],
            metadatas=metadatas,
            documents=documents,
        )

    def search(
        self,
        query: Union[str, Sequence[float], np.ndarray],
        limit: int = 50,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if isinstance(query, (list, tuple)):
            query_vector = np.array(query, dtype=np.float32)
        elif isinstance(query, np.ndarray):
            query_vector = query
        else:
            if not str(query).strip():
                return []
            query_vector = self.embedding_model.encode(str(query), normalize_embeddings=True)

        query_vector = _normalize_vector(query_vector)

        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=limit,
            where=metadata_filter,
            include=["metadatas", "distances", "documents", "embeddings"],
        )

        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]

        converted: List[SearchResult] = []
        for chunk_id, distance, metadata, document in zip(ids, distances, metadatas, documents):
            if metadata is None:
                metadata = {}
            score = float(1 - distance) if distance is not None else 0.0
            converted.append(
                SearchResult(
                    chunk_id=str(chunk_id),
                    score=score,
                    payload={**metadata, "text": document},
                    text=document or "",
                )
            )

        return converted

    def count(self) -> int:
        return int(self.collection.count())

    def count_by_document(self, document_id: str) -> int:
        results = self.collection.get(where={"document_id": document_id})
        ids = results.get("ids", []) or []
        return len(ids)

    def delete_by_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})


def _metadata_matches(chunk: Dict[str, Any], metadata_filter: Dict[str, Any]) -> bool:
    metadata = {**chunk.get("metadata", {}), **{k: v for k, v in chunk.items() if k != "metadata"}}
    for key, value in metadata_filter.items():
        if metadata.get(key) != value:
            return False
    return True
