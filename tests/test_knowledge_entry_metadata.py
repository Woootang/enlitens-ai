import sys
import types

import asyncio


# Provide minimal stubs for optional dependencies used during module import
langgraph_module = types.ModuleType("langgraph")
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
langgraph_module.graph = langgraph_graph_module

sys.modules.setdefault("langgraph", langgraph_module)
sys.modules.setdefault("langgraph.graph", langgraph_graph_module)


def _register_stub(path: str, **attributes):
    module = types.ModuleType(path)
    for name, value in attributes.items():
        setattr(module, name, value)
    sys.modules.setdefault(path, module)


class _StubSupervisorAgent:
    async def process_document(self, *_args, **_kwargs):
        return {}


class _StubExtractionTeam:
    async def extract_entities(self, *_args, **_kwargs):
        return {}


class _StubPDFExtractor:
    def extract(self, *_args, **_kwargs):  # pragma: no cover - not used in tests
        return {}


class _StubExtractionTools:
    pass


class _StubEmbeddingIngestionPipeline:
    def ingest_document(self, *_args, **_kwargs):
        return None


_register_stub("src.agents.supervisor_agent", SupervisorAgent=_StubSupervisorAgent)
_register_stub("src.agents.extraction_team", ExtractionTeam=_StubExtractionTeam)
_register_stub("src.extraction.enhanced_pdf_extractor", EnhancedPDFExtractor=_StubPDFExtractor)
_register_stub("src.extraction.enhanced_extraction_tools", EnhancedExtractionTools=_StubExtractionTools)
_register_stub("src.retrieval.embedding_ingestion", EmbeddingIngestionPipeline=_StubEmbeddingIngestionPipeline)

from process_multi_agent_corpus import MultiAgentProcessor


def test_convert_to_knowledge_entry_populates_metadata_fields():
    processor = MultiAgentProcessor.__new__(MultiAgentProcessor)

    result = {
        "document_text": "one two three four",
        "agent_outputs": {},
    }

    entry = asyncio.run(
        processor._convert_to_knowledge_entry(
            result,
            document_id="doc-123",
            processing_time=1.5,
            file_size=4096,
            page_count=12,
            filename="doc-123.pdf",
            fallback_full_text="",
        )
    )

    metadata = entry.metadata
    assert metadata.file_size == 4096
    assert metadata.page_count == 12
    assert metadata.full_text_stored is True
    assert metadata.word_count == 4
    assert metadata.filename == "doc-123.pdf"


def test_convert_to_knowledge_entry_uses_fallback_text_for_metadata():
    processor = MultiAgentProcessor.__new__(MultiAgentProcessor)

    result = {
        "agent_outputs": {},
    }

    entry = asyncio.run(
        processor._convert_to_knowledge_entry(
            result,
            document_id="doc-456",
            processing_time=2.0,
            file_size=2048,
            page_count=8,
            filename="custom-name.pdf",
            fallback_full_text="fallback text only",
        )
    )

    metadata = entry.metadata
    assert metadata.file_size == 2048
    assert metadata.page_count == 8
    assert metadata.full_text_stored is True
    assert metadata.word_count == 3
    assert metadata.filename == "custom-name.pdf"
    assert entry.full_document_text == "fallback text only"
