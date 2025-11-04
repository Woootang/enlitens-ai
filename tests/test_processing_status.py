import asyncio
from pathlib import Path
import sys
import types

import pytest

# Provide lightweight stubs for optional langgraph dependency used during import
langgraph_module = types.ModuleType("langgraph")
langgraph_graph_module = types.ModuleType("langgraph.graph")
langgraph_graph_module.StateGraph = object  # type: ignore[attr-defined]
langgraph_graph_module.END = object()
langgraph_module.graph = langgraph_graph_module
sys.modules.setdefault("langgraph", langgraph_module)
sys.modules.setdefault("langgraph.graph", langgraph_graph_module)


class _HttpxClientStub:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):  # pragma: no cover - unused in tests
        return self

    async def __aexit__(self, *exc_info):  # pragma: no cover - unused in tests
        return False

    def __enter__(self):  # pragma: no cover - unused in tests
        return self

    def __exit__(self, *exc_info):  # pragma: no cover - unused in tests
        return False

    async def post(self, *args, **kwargs):  # pragma: no cover - unused in tests
        raise RuntimeError("httpx stub invoked during tests")

    def post(self, *args, **kwargs):  # pragma: no cover - unused in tests
        raise RuntimeError("httpx stub invoked during tests")


httpx_module = types.ModuleType("httpx")
httpx_module.AsyncClient = _HttpxClientStub
httpx_module.Client = _HttpxClientStub
httpx_module.TimeoutException = RuntimeError
httpx_module.ConnectError = RuntimeError
httpx_module.ReadTimeout = RuntimeError
httpx_module.RequestError = RuntimeError
httpx_module.HTTPStatusError = RuntimeError
httpx_module.Response = object  # type: ignore[assignment]
httpx_module.AsyncBaseTransport = object  # type: ignore[assignment]
sys.modules.setdefault("httpx", httpx_module)

json_repair_module = types.ModuleType("json_repair")
json_repair_module.repair_json = lambda payload: payload
sys.modules.setdefault("json_repair", json_repair_module)


class _BaseModelStub:
    def __init__(self, **data) -> None:
        for key, value in data.items():
            setattr(self, key, value)

    def model_dump(self, *args, **kwargs):  # pragma: no cover - unused
        return self.__dict__.copy()

    @classmethod
    def model_validate(cls, data):  # pragma: no cover - unused
        if isinstance(data, dict):
            return cls(**data)
        raise TypeError("Expected mapping for model_validate")


def _field_stub(default=None, default_factory=None, **kwargs):  # pragma: no cover - unused
    if default_factory is not None:
        return default_factory()
    return default


pydantic_module = types.ModuleType("pydantic")
pydantic_module.BaseModel = _BaseModelStub
pydantic_module.Field = _field_stub
pydantic_module.ValidationInfo = type("ValidationInfo", (), {})
pydantic_module.ValidationError = type("ValidationError", (Exception,), {})


def _field_validator_stub(*args, **kwargs):  # pragma: no cover - unused
    def decorator(func):
        return func

    return decorator


pydantic_module.field_validator = _field_validator_stub
sys.modules.setdefault("pydantic", pydantic_module)


class _AsyncRetryingStub:
    def __init__(self, *args, **kwargs) -> None:
        self._attempt_yielded = False

    def __aiter__(self):
        self._attempt_yielded = False
        return self

    async def __anext__(self):
        if self._attempt_yielded:
            raise StopAsyncIteration
        self._attempt_yielded = True

        class _AttemptContext:
            retry_state = types.SimpleNamespace(outcome=None)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc_info):
                return False

        return _AttemptContext()


def _identity(value=None):  # pragma: no cover - unused
    return value


tenacity_module = types.ModuleType("tenacity")
tenacity_module.AsyncRetrying = _AsyncRetryingStub
tenacity_module.retry_if_exception = _identity
tenacity_module.stop_after_attempt = _identity
tenacity_module.wait_exponential = _identity
sys.modules.setdefault("tenacity", tenacity_module)

yaml_module = types.ModuleType("yaml")
yaml_module.safe_load = lambda payload: {}
yaml_module.safe_dump = lambda data, **kwargs: ""
sys.modules.setdefault("yaml", yaml_module)

import process_multi_agent_corpus
from process_multi_agent_corpus import MultiAgentProcessor

from src.knowledge_base.status_file import (
    STATUS_FILE_NAME,
    KnowledgeBaseUnavailableError,
    read_processing_status,
)
from src.retrieval.embedding_ingestion import load_knowledge_entries_from_path
from src.models.enlitens_schemas import EnlitensKnowledgeBase


class _DummySupervisor:
    def __init__(self) -> None:
        self.is_initialized = True

    async def initialize(self) -> bool:  # pragma: no cover - not exercised
        self.is_initialized = True
        return True

    async def get_system_status(self) -> dict[str, object]:
        return {}

    async def cleanup(self) -> None:
        return None


class _DummyExtractor:
    pass


class _DummyExtractionTools:
    pass


class _DummyExtractionTeam:
    pass


class _DummyEmbeddingIngestion:
    def __init__(self, *args, **kwargs) -> None:
        self.vector_store_status = {"is_healthy": False}

    def health_snapshot(self) -> dict[str, object]:  # pragma: no cover - not used
        return self.vector_store_status


def test_status_file_written_and_consumed_on_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_multi_agent_corpus, "SupervisorAgent", lambda: _DummySupervisor())
    monkeypatch.setattr(process_multi_agent_corpus, "EnhancedPDFExtractor", lambda: _DummyExtractor())
    monkeypatch.setattr(process_multi_agent_corpus, "EnhancedExtractionTools", lambda: _DummyExtractionTools())
    monkeypatch.setattr(process_multi_agent_corpus, "ExtractionTeam", lambda: _DummyExtractionTeam())
    monkeypatch.setattr(process_multi_agent_corpus, "EmbeddingIngestion", lambda *args, **kwargs: _DummyEmbeddingIngestion())
    monkeypatch.setattr(process_multi_agent_corpus, "post_monitor_stats", lambda payload: None)

    monkeypatch.setattr(
        MultiAgentProcessor,
        "_bootstrap_context_ingestion",
        lambda self: None,
    )
    monkeypatch.setattr(
        MultiAgentProcessor,
        "_load_st_louis_context",
        lambda self: {
            "demographics": {"mental_health_challenges": []},
            "clinical_priorities": [],
            "founder_voice": {},
        },
    )
    async def _noop(self):
        return None

    async def _load_empty(self):
        return EnlitensKnowledgeBase()

    async def _save_noop(self, kb, processed_count, total_files):
        return None

    async def _cleanup_noop(self):
        return None

    monkeypatch.setattr(MultiAgentProcessor, "_check_system_resources", _noop)
    monkeypatch.setattr(MultiAgentProcessor, "_load_progress", _load_empty)
    monkeypatch.setattr(MultiAgentProcessor, "_save_progress", _save_noop)
    monkeypatch.setattr(MultiAgentProcessor, "_cleanup", _cleanup_noop)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "doc-001.pdf").write_bytes(b"%PDF-1.4\n")

    output_path = tmp_path / "knowledge_base.json"
    processor = MultiAgentProcessor(str(input_dir), str(output_path))

    async def _crash_process_document(self, pdf_path: Path):
        raise RuntimeError("synthetic crash for testing")

    monkeypatch.setattr(MultiAgentProcessor, "process_document", _crash_process_document, raising=True)

    with pytest.raises(RuntimeError, match="synthetic crash"):
        asyncio.run(processor.process_corpus())

    status_path = output_path.parent / STATUS_FILE_NAME
    status = read_processing_status(status_path)
    assert status is not None
    assert status.reason == "synthetic crash for testing"
    assert status.affected_documents == ["doc-001"]

    latest_path = output_path.parent / "enlitens_knowledge_base_latest.json"
    with pytest.raises(KnowledgeBaseUnavailableError) as excinfo:
        load_knowledge_entries_from_path(str(latest_path))

    assert "synthetic crash for testing" in str(excinfo.value)
