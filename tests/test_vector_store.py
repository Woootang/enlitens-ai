"""Unit tests for vector store embedding resolution helpers."""

import logging
import types
from typing import Any, Dict, List

import pytest

from src.retrieval import vector_store


class _DummySentenceTransformer:
    """Simple stub for SentenceTransformer used in tests."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

    def get_sentence_embedding_dimension(self) -> int:
        return 3


@pytest.fixture(autouse=True)
def _restore_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure _SentenceTransformer is always restored between tests."""

    original = vector_store._SentenceTransformer
    monkeypatch.setattr(vector_store, "_SentenceTransformer", original, raising=False)


def test_resolve_embedding_model_applies_bge_m3_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {}

    def fake_sentence_transformer(name: str, device: str = "cpu") -> _DummySentenceTransformer:
        calls["model_name"] = name
        calls["device"] = device
        return _DummySentenceTransformer(name, device)

    telemetry_events: List[Dict[str, Any]] = []

    def fake_log_with_telemetry(
        _logger_method: Any,
        message: str,
        *args: Any,
        agent: str,
        severity: str,
        impact: str,
        doc_id: Any = None,
        details: Any = None,
        **kwargs: Any,
    ) -> None:
        telemetry_events.append(
            {
                "message": message % args if args else message,
                "agent": agent,
                "severity": severity,
                "impact": impact,
                "details": details,
            }
        )

    monkeypatch.setattr(vector_store, "_SentenceTransformer", fake_sentence_transformer)
    monkeypatch.setattr(vector_store, "log_with_telemetry", fake_log_with_telemetry)
    monkeypatch.setattr(vector_store, "logger", logging.getLogger("vector_store_test"))
    monkeypatch.setattr(vector_store, "torch", types.SimpleNamespace(__version__="2.1.0"))

    model = vector_store._resolve_embedding_model("BAAI/bge-m3", "cpu")

    assert isinstance(model, _DummySentenceTransformer)
    assert calls["model_name"] == "intfloat/e5-base-v2"
    assert telemetry_events, "Expected telemetry to record fallback usage"

    telemetry = telemetry_events[0]
    assert telemetry["agent"] == vector_store.TELEMETRY_AGENT
    assert telemetry["severity"] == vector_store.TelemetrySeverity.MINOR
    assert telemetry["impact"] == "retrieval-quality"
    assert telemetry["details"]["fallback_model"] == "intfloat/e5-base-v2"
    assert telemetry["details"]["original_model"] == "BAAI/bge-m3"


def test_resolve_embedding_model_uses_requested_model_when_torch_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: Dict[str, Any] = {}

    def fake_sentence_transformer(name: str, device: str = "cpu") -> _DummySentenceTransformer:
        calls["model_name"] = name
        calls["device"] = device
        return _DummySentenceTransformer(name, device)

    def fail_log_with_telemetry(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - guard rail
        raise AssertionError("Telemetry should not be invoked when no fallback is required")

    monkeypatch.setattr(vector_store, "_SentenceTransformer", fake_sentence_transformer)
    monkeypatch.setattr(vector_store, "log_with_telemetry", fail_log_with_telemetry)
    monkeypatch.setattr(vector_store, "logger", logging.getLogger("vector_store_test"))
    monkeypatch.setattr(vector_store, "torch", types.SimpleNamespace(__version__="2.6.0"))

    model = vector_store._resolve_embedding_model("BAAI/bge-m3", "cpu")

    assert isinstance(model, _DummySentenceTransformer)
    assert calls["model_name"] == "BAAI/bge-m3"
