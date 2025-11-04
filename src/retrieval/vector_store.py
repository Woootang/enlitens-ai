"""Vector store integrations with Qdrant, Chroma, and in-memory fallbacks."""
from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:  # Optional dependency - only required for persistent usage
    import chromadb
    from chromadb.api import ClientAPI as ChromaClient
    from chromadb.errors import InvalidCollectionException
except Exception:  # pragma: no cover - chromadb not always installed in tests
    chromadb = None
    ChromaClient = None  # type: ignore
    InvalidCollectionException = Exception  # type: ignore

try:  # Optional dependency - allow tests to run without Qdrant installed
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:  # pragma: no cover - CI environment without qdrant_client
    QdrantClient = None  # type: ignore[assignment]
    qmodels = None  # type: ignore[assignment]

try:  # sentence_transformers is optional in lightweight environments
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except Exception:  # pragma: no cover - used in testing environments without torch
    _SentenceTransformer = None

try:  # pragma: no cover - torch is optional for lightweight test environments
    import torch  # type: ignore
except Exception:  # pragma: no cover - match optional dependency pattern
    torch = None  # type: ignore

try:  # pragma: no cover - packaging may be absent in some environments
    from packaging import version as packaging_version
except ImportError:  # pragma: no cover - fall back to simple comparison
    packaging_version = None  # type: ignore[assignment]

from src.monitoring.error_telemetry import TelemetrySeverity, log_with_telemetry

from src.utils.vector_math import (
    dot as vector_dot,
    ensure_2d_float_list,
    ensure_float_list,
    normalize as vector_normalize,
    numpy_available,
    to_numpy,
    to_numpy_matrix,
)


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
    ) -> Union[List[float], List[List[float]], "np.ndarray"]:
        single_input = isinstance(sentences, str)
        if single_input:
            to_encode = [sentences]
        else:
            to_encode = list(sentences)

        embeddings: List[List[float]] = []
        for sentence in to_encode:
            text = sentence or ""
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            byte_values = [byte / 255.0 for byte in digest]
            repeats = int(math.ceil(self.dimension / float(len(byte_values))))
            tiled = (byte_values * repeats)[: self.dimension]
            embedding = vector_normalize(tiled) if normalize_embeddings else tiled
            embeddings.append(embedding)

        if single_input:
            if numpy_available():
                return to_numpy(embeddings[0])
            return embeddings[0]

        if numpy_available():
            return to_numpy_matrix(embeddings)
        return embeddings

logger = logging.getLogger(__name__)
TELEMETRY_AGENT = "vector_store"
_BGE_M3_MODEL_NAME = "BAAI/bge-m3"
_BGE_M3_TORCH_REQUIREMENT = "2.6"
_BGE_M3_FALLBACK_MODEL = "intfloat/e5-base-v2"


def _parse_version_tuple(version_str: str) -> Optional[Tuple[int, ...]]:
    numeric_portion = version_str.split("+", 1)[0]
    parts: List[int] = []
    for segment in numeric_portion.split("."):
        digits = ""
        for char in segment:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    if not parts:
        return None
    return tuple(parts)


def _version_is_less_than(current: str, minimum_required: str) -> bool:
    if packaging_version is not None:
        try:
            return packaging_version.parse(current) < packaging_version.parse(minimum_required)
        except Exception:  # pragma: no cover - fall back to manual parsing
            pass

    current_tuple = _parse_version_tuple(current)
    required_tuple = _parse_version_tuple(minimum_required)
    if current_tuple is None or required_tuple is None:
        return False

    # Pad tuples to the same length for lexicographic comparison
    max_length = max(len(current_tuple), len(required_tuple))
    current_padded = current_tuple + (0,) * (max_length - len(current_tuple))
    required_padded = required_tuple + (0,) * (max_length - len(required_tuple))
    return current_padded < required_padded


def _maybe_apply_bge_m3_fallback(requested_model: str) -> Tuple[str, Optional[Dict[str, str]]]:
    """Return a safetensors-compatible fallback model when torch is too old."""

    if requested_model != _BGE_M3_MODEL_NAME:
        return requested_model, None

    if torch is None:  # pragma: no cover - occurs when torch is unavailable entirely
        return requested_model, None

    torch_version = getattr(torch, "__version__", None)
    if not torch_version:
        return requested_model, None

    if not _version_is_less_than(torch_version, _BGE_M3_TORCH_REQUIREMENT):
        return requested_model, None

    details = {
        "original_model": requested_model,
        "fallback_model": _BGE_M3_FALLBACK_MODEL,
        "torch_version": torch_version,
        "minimum_required_torch": _BGE_M3_TORCH_REQUIREMENT,
    }
    return _BGE_M3_FALLBACK_MODEL, details


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

    fallback_name, fallback_details = _maybe_apply_bge_m3_fallback(resolved_name)
    if fallback_details:
        log_with_telemetry(
            logger.warning,
            (
                "Torch %s detected; %s requires torch>=%s for safetensors support. "
                "Falling back to %s to avoid incompatibilities."
            ),
            fallback_details["torch_version"],
            fallback_details["original_model"],
            fallback_details["minimum_required_torch"],
            fallback_details["fallback_model"],
            agent=TELEMETRY_AGENT,
            severity=TelemetrySeverity.MINOR,
            impact="retrieval-quality",
            details=fallback_details,
        )
        resolved_name = fallback_name

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


def _normalize_vector(vector: Union[Sequence[float], "np.ndarray"]) -> List[float]:
    return vector_normalize(vector)


def _to_float_vectors(
    embeddings: Union[Sequence[Sequence[float]], "np.ndarray"],
) -> List[List[float]]:
    return ensure_2d_float_list(embeddings)


def _to_float_vector(vector: Union[Sequence[float], "np.ndarray"]) -> List[float]:
    return ensure_float_list(vector)


def _build_filter_condition(metadata_filter: Optional[Dict[str, Any]]) -> Optional[Any]:
    if qmodels is None or not metadata_filter:
        return None

    conditions: List[Any] = []
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
        query: Union[str, Sequence[float]],
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

    def check_health(self) -> Tuple[bool, Dict[str, Any]]:
        """Return whether the backing store is reachable."""
        return True, {"mode": "persistent"}


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

        embeddings_raw = self.embedding_model.encode(
            [chunk["text"] for chunk in chunks],
            normalize_embeddings=True,
        )
        embeddings = _to_float_vectors(embeddings_raw)

        if self.client is not None:
            points = [
                qmodels.PointStruct(
                    id=chunk["chunk_id"],
                    vector=embedding,
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

    def search(
        self,
        query: Union[str, Sequence[float]],
        limit: int = 50,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if isinstance(query, (list, tuple)):
            query_vector = _normalize_vector(query)
        elif numpy_available() and hasattr(query, "shape"):
            query_vector = _normalize_vector(query)  # type: ignore[arg-type]
        else:
            query_text = str(query)
            if not query_text.strip():
                return []
            encoded = self.embedding_model.encode(query_text, normalize_embeddings=True)
            query_vector = _normalize_vector(_to_float_vector(encoded))

        if self.client is not None:
            try:
                filter_condition = _build_filter_condition(metadata_filter)
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
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
        query_vector: Sequence[float],
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
            score = float(vector_dot(query_vector, embedding))
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

    def check_health(self) -> Tuple[bool, Dict[str, Any]]:
        if self.client is None:
            return False, {
                "mode": "in_memory",
                "error": "Qdrant client unavailable; using local fallback",
            }

        try:
            self.client.get_collection(self.collection_name)
        except Exception as exc:
            self.client = None
            return False, {
                "mode": "in_memory",
                "error": str(exc),
                "exception": exc.__class__.__name__,
            }

        return True, {"mode": "persistent"}


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

        embeddings_raw = self.embedding_model.encode(
            [chunk["text"] for chunk in chunks],
            normalize_embeddings=True,
        )
        embeddings = _to_float_vectors(embeddings_raw)

        ids = [str(chunk["chunk_id"]) for chunk in chunks]
        metadatas = [{**chunk, **chunk.get("metadata", {})} for chunk in chunks]
        documents = [chunk.get("text", "") for chunk in chunks]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    def search(
        self,
        query: Union[str, Sequence[float]],
        limit: int = 50,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if isinstance(query, (list, tuple)):
            query_vector = _normalize_vector(query)
        elif numpy_available() and hasattr(query, "shape"):
            query_vector = _normalize_vector(query)  # type: ignore[arg-type]
        else:
            query_text = str(query)
            if not query_text.strip():
                return []
            encoded = self.embedding_model.encode(query_text, normalize_embeddings=True)
            query_vector = _normalize_vector(_to_float_vector(encoded))

        results = self.collection.query(
            query_embeddings=[query_vector],
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
