"""Utility module providing deterministic prompt caching."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PromptCache:
    """Thread-safe in-memory cache for LLM prompt responses."""

    _store: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _build_key(self, prefix: str, chunk_id: str, prompt: str) -> str:
        """Create a deterministic cache key using prefix, chunk, and prompt hash."""
        normalized_prefix = prefix or "default"
        normalized_chunk = chunk_id or "global"
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return f"{normalized_prefix}:{normalized_chunk}:{prompt_hash}"

    def get(self, prefix: str, chunk_id: str, prompt: str) -> Any | None:
        """Retrieve a cached value if present."""
        key = self._build_key(prefix, chunk_id, prompt)
        with self._lock:
            return self._store.get(key)

    def set(self, prefix: str, chunk_id: str, prompt: str, value: Any) -> str:
        """Store a cached value and return the cache key."""
        key = self._build_key(prefix, chunk_id, prompt)
        with self._lock:
            self._store[key] = value
        return key

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._store.clear()
