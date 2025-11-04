import sys
import types
from pathlib import Path

import pytest

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


langgraph_graph_module = types.ModuleType("langgraph.graph")
langgraph_graph_module.StateGraph = _StubStateGraph
langgraph_graph_module.END = object()
langgraph_module = types.ModuleType("langgraph")
langgraph_module.graph = langgraph_graph_module
sys.modules.setdefault("langgraph", langgraph_module)
sys.modules.setdefault("langgraph.graph", langgraph_graph_module)


def _register_stub(path: str, **attributes: object) -> None:
    module = types.ModuleType(path)
    for name, value in attributes.items():
        setattr(module, name, value)
    sys.modules.setdefault(path, module)


class _Placeholder:
    def __init__(self, *_args, **_kwargs):
        pass


_register_stub("src.agents.supervisor_agent", SupervisorAgent=_Placeholder)
_register_stub("src.agents.extraction_team", ExtractionTeam=_Placeholder)
_register_stub("src.extraction.enhanced_pdf_extractor", EnhancedPDFExtractor=_Placeholder)
_register_stub("src.extraction.enhanced_extraction_tools", EnhancedExtractionTools=_Placeholder)
_register_stub(
    "src.retrieval.embedding_ingestion",
    EmbeddingIngestionPipeline=_Placeholder,
    EmbeddingIngestion=_Placeholder,
)
_register_stub(
    "src.models.enlitens_schemas",
    EnlitensKnowledgeBase=_Placeholder,
    EnlitensKnowledgeEntry=_Placeholder,
)
_register_stub(
    "src.utils.enhanced_logging",
    setup_enhanced_logging=lambda *_args, **_kwargs: None,
    log_startup_banner=lambda *_args, **_kwargs: None,
)

import process_multi_agent_corpus as processor_module
from process_multi_agent_corpus import MultiAgentProcessor


class _StubPDFExtractor:
    def __init__(self, report_text: str) -> None:
        self._report_text = report_text

    def extract(self, *_args, **_kwargs):
        return {"archival_content": {"full_document_text_markdown": self._report_text}}


class _StubSupervisor:
    async def process_document(self, *_args, **_kwargs):
        return {}


class _StubExtractionTeam:
    async def extract_entities(self, *_args, **_kwargs):
        return {}


@pytest.mark.parametrize("report_exists", [True, False])
def test_context_documents_ingested_on_init(monkeypatch, tmp_path: Path, report_exists: bool):
    ingested = []

    class StubIngestionPipeline:
        def __init__(self, *_args, **_kwargs):
            self.calls = []
            self.health_status = {"is_healthy": True, "mode": "persistent"}

        def ingest_document(self, *, document_id, full_text, agent_outputs=None, metadata=None, rebuild=False):
            payload = {
                "document_id": document_id,
                "full_text": full_text,
                "metadata": metadata or {},
            }
            ingested.append(payload)
            self.calls.append(payload)
            return types.SimpleNamespace(
                document_id=document_id,
                chunks_ingested=1,
                full_text_chunks=1,
                agent_chunks=0,
                metadata=metadata or {},
            )

        def health_snapshot(self):
            return dict(self.health_status)

    monkeypatch.setattr(processor_module, "SupervisorAgent", lambda: _StubSupervisor())
    monkeypatch.setattr(processor_module, "ExtractionTeam", lambda: _StubExtractionTeam())
    monkeypatch.setattr(processor_module, "EnhancedExtractionTools", lambda: object())
    monkeypatch.setattr(processor_module, "EnlitensKnowledgeBase", lambda: object())

    report_text = "st louis public health overview"
    monkeypatch.setattr(
        processor_module,
        "EnhancedPDFExtractor",
        lambda: _StubPDFExtractor(report_text),
    )
    monkeypatch.setattr(processor_module, "EmbeddingIngestion", StubIngestionPipeline)

    def fake_candidates(self, filename: str):
        return [tmp_path / filename]

    monkeypatch.setattr(MultiAgentProcessor, "_context_file_candidates", fake_candidates)

    (tmp_path / "intakes.txt").write_text("intake context", encoding="utf-8")
    (tmp_path / "transcripts.txt").write_text("transcript context", encoding="utf-8")

    stl_path = tmp_path / "stl_report.pdf"
    if report_exists:
        stl_path.write_text("pdf-bytes", encoding="utf-8")
    else:
        if stl_path.exists():
            stl_path.unlink()

    processor = MultiAgentProcessor(
        input_dir=str(tmp_path),
        output_file=str(tmp_path / "out.json"),
        st_louis_report=str(stl_path) if report_exists else None,
    )

    expected_ids = {"context:intakes", "context:transcripts"}
    if report_exists:
        expected_ids.add("context:stl_health_report")

    ingested_ids = {payload["document_id"] for payload in ingested}
    assert ingested_ids == expected_ids

    for payload in ingested:
        metadata = payload["metadata"]
        assert metadata["doc_type"] == "context_reference"
        assert metadata.get("description")
        assert metadata.get("source_type")
        assert metadata["document_id"] == payload["document_id"]

    if report_exists:
        report_payload = {
            item["metadata"]["source_type"]: item for item in ingested
        }["stl_health_report"]
        assert "st. louis" in report_payload["metadata"]["description"].lower()

    assert processor.embedding_ingestion.calls, "Expected ingestion pipeline to be invoked"
