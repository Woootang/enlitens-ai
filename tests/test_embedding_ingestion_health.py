import logging
import sys
import types
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import pytest

langgraph_graph_module = types.ModuleType("langgraph.graph")


class _StubStateGraph:
    def __init__(self, *_args, **_kwargs):
        pass

    def add_node(self, *_args, **_kwargs):
        return None

    def set_entry_point(self, *_args, **_kwargs):
        return None

    def add_edge(self, *_args, **_kwargs):
        return None

    def compile(self):
        return self


langgraph_graph_module.StateGraph = _StubStateGraph
langgraph_graph_module.END = object()
langgraph_module = types.ModuleType("langgraph")
langgraph_module.graph = langgraph_graph_module
sys.modules.setdefault("langgraph", langgraph_module)
sys.modules.setdefault("langgraph.graph", langgraph_graph_module)

httpx_module = types.ModuleType("httpx")
httpx_module.Client = object
httpx_module.AsyncClient = object
httpx_module.Request = object
httpx_module.Response = object
httpx_module.MockTransport = object
httpx_module.HTTPStatusError = Exception
sys.modules.setdefault("httpx", httpx_module)

json_repair_module = types.ModuleType("json_repair")
json_repair_module.repair_json = lambda payload: payload
sys.modules.setdefault("json_repair", json_repair_module)


def _register_stub(path: str, **attributes: object) -> None:
    module = types.ModuleType(path)
    for name, value in attributes.items():
        setattr(module, name, value)
    sys.modules.setdefault(path, module)


class _StubAgent:
    def __init__(self, *_args, **_kwargs):
        pass

    async def initialize(self) -> bool:
        return True

    async def process(self, *_args, **_kwargs):
        return {}

    async def validate_output(self, *_args, **_kwargs) -> bool:
        return True

    async def cleanup(self) -> None:
        return None

    def get_status(self) -> Dict[str, Any]:
        return {}


class _StubSettingsLLM:
    provider = "stub"

    def endpoint_for(self, *_args, **_kwargs) -> str:
        return "http://stub"


class _StubSettings:
    def __init__(self) -> None:
        self.llm = _StubSettingsLLM()

    def model_for_agent(self, _agent: str) -> str:
        return "stub-model"


def _get_stub_settings() -> _StubSettings:
    return _StubSettings()


_register_stub("src.utils.settings", get_settings=_get_stub_settings)
_register_stub("src.synthesis.ollama_client", LLMServiceError=Exception)
_register_stub("src.agents.science_extraction_agent", ScienceExtractionAgent=_StubAgent)
_register_stub("src.agents.clinical_synthesis_agent", ClinicalSynthesisAgent=_StubAgent)
_register_stub("src.agents.founder_voice_agent", FounderVoiceAgent=_StubAgent)
_register_stub("src.agents.context_rag_agent", ContextRAGAgent=_StubAgent)
_register_stub("src.agents.marketing_seo_agent", MarketingSEOAgent=_StubAgent)
_register_stub("src.agents.client_profile_agent", ClientProfileAgent=_StubAgent)
_register_stub("src.agents.validation_agent", ValidationAgent=_StubAgent)
_register_stub("src.agents.educational_content_agent", EducationalContentAgent=_StubAgent)
_register_stub("src.agents.rebellion_framework_agent", RebellionFrameworkAgent=_StubAgent)
_register_stub("src.models.enlitens_schemas", EnlitensKnowledgeEntry=object)

from src.agents.supervisor_agent import SupervisorAgent
from src.agents.workflow_state import create_initial_state
from src.retrieval.embedding_ingestion import EmbeddingIngestion
from src.retrieval.vector_store import BaseVectorStore


class RefusingVectorStore(BaseVectorStore):
    def upsert(self, chunks: List[Dict[str, Any]]) -> None:  # pragma: no cover - not used
        raise NotImplementedError

    def search(
        self,
        query: Sequence[float] | str,
        limit: int = 50,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:  # pragma: no cover - not used
        return []

    def count(self) -> int:  # pragma: no cover - not used
        return 0

    def count_by_document(self, document_id: str) -> int:  # pragma: no cover - not used
        return 0

    def delete_by_document(self, document_id: str) -> None:  # pragma: no cover - not used
        return None

    def check_health(self):
        raise ConnectionError("connection refused")


def test_embedding_ingestion_records_critical_health_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    events: List[Dict[str, Any]] = []

    def fake_record_event(*_args: Any, **kwargs: Any) -> None:
        events.append(kwargs)

    monkeypatch.setattr(
        "src.monitoring.error_telemetry.telemetry_recorder.record_event",
        fake_record_event,
    )

    caplog.set_level(logging.CRITICAL)

    ingestion = EmbeddingIngestion(vector_store=RefusingVectorStore())

    assert ingestion.health_status["is_healthy"] is False
    assert ingestion.ingest_mode == "in_memory"
    assert events, "expected telemetry record for health failure"
    payload = events[-1]
    assert payload.get("severity") == "critical"
    assert payload.get("impact") == "Persistent vector store unavailable"
    assert payload.get("details", {}).get("mode") == "in_memory"
    assert any("Vector store health check failed" in record.message for record in caplog.records)


def test_run_summary_includes_vector_store_warning() -> None:
    supervisor = SupervisorAgent()
    state = create_initial_state("doc-1", "sample text")
    timestamp = datetime.utcnow()
    state["start_timestamp"] = timestamp
    state["end_timestamp"] = timestamp
    state["metadata"]["vector_store_status"] = {
        "is_healthy": False,
        "mode": "in_memory",
        "reason": "connection_refused",
    }

    output = supervisor._finalize_output(state)

    summary = output["run_summary"]
    assert summary["status"] == output["supervisor_status"]
    assert summary["vector_store_status"]["mode"] == "in_memory"
    warnings = summary.get("warnings")
    assert warnings and any("Persistent vector store unavailable" in warning for warning in warnings)
