import asyncio
import logging
import sys
import types
from pathlib import Path

import pytest


class _Stub:
    def __init__(self, *_args, **_kwargs):
        pass


def _register_stub(path: str, **attributes: object) -> None:
    module = types.ModuleType(path)
    for name, value in attributes.items():
        setattr(module, name, value)
    sys.modules.setdefault(path, module)


_register_stub("src.agents.supervisor_agent", SupervisorAgent=_Stub)
_register_stub("src.agents.extraction_team", ExtractionTeam=_Stub)
_register_stub("src.extraction.enhanced_pdf_extractor", EnhancedPDFExtractor=_Stub)
_register_stub("src.extraction.enhanced_extraction_tools", EnhancedExtractionTools=_Stub)
_register_stub("src.retrieval.embedding_ingestion", EmbeddingIngestionPipeline=_Stub)
_register_stub(
    "src.models.enlitens_schemas",
    EnlitensKnowledgeBase=_Stub,
    EnlitensKnowledgeEntry=_Stub,
)
_register_stub(
    "src.utils.enhanced_logging",
    setup_enhanced_logging=lambda *args, **kwargs: None,
    log_startup_banner=lambda *args, **kwargs: None,
)

langgraph_graph_module = types.ModuleType("langgraph.graph")
langgraph_graph_module.StateGraph = _Stub
langgraph_graph_module.END = object()
langgraph_module = types.ModuleType("langgraph")
langgraph_module.graph = langgraph_graph_module
sys.modules.setdefault("langgraph", langgraph_module)
sys.modules.setdefault("langgraph.graph", langgraph_graph_module)

from process_multi_agent_corpus import MultiAgentProcessor


class _StubSupervisor:
    def __init__(self) -> None:
        self.last_context = None

    async def process_document(self, context):
        self.last_context = context
        parts = [
            context.get("raw_client_context"),
            context.get("raw_founder_context"),
        ]
        snippet = " | ".join([part for part in parts if part])
        return {
            "supervisor_status": "completed",
            "document_id": context["document_id"],
            "document_text": context["document_text"],
            "agent_outputs": {
                "rebellion_framework": {
                    "narrative_deconstruction": [snippet],
                    "sensory_profiling": [],
                    "executive_function": [],
                    "social_processing": [],
                    "strengths_synthesis": [],
                    "rebellion_themes": [],
                    "aha_moments": [],
                },
                "educational_content": {"explanations": [snippet]},
                "marketing_content": {"headlines": [snippet]},
            },
        }


class _StubExtractionTeam:
    async def extract_entities(self, *_args, **_kwargs):
        return {}


def test_pipeline_uses_raw_context_when_analyses_empty(tmp_path: Path):
    processor = MultiAgentProcessor.__new__(MultiAgentProcessor)
    processor.input_dir = tmp_path
    processor.output_file = tmp_path / "out.json"
    processor.temp_file = processor.output_file.with_suffix(".temp")
    processor.context_dir = tmp_path
    processor.supervisor = _StubSupervisor()
    processor.pdf_extractor = types.SimpleNamespace(
        extract=lambda _path: {
            "archival_content": {
                "full_document_text_markdown": "Document body " * 20
            }
        }
    )
    processor.extraction_tools = object()
    processor.extraction_team = _StubExtractionTeam()
    processor.knowledge_base = None
    processor.embedding_ingestion = None
    processor.retry_attempts = 1
    processor.st_louis_context = {
        "demographics": {
            "mental_health_challenges": ["challenge"],
            "socioeconomic_factors": [],
        },
        "clinical_priorities": ["priority"],
        "founder_voice": ["voice"],
    }

    processor._analyze_client_insights = types.MethodType(
        lambda self: {"raw_content": "client raw context"}, processor
    )
    processor._analyze_founder_insights = types.MethodType(
        lambda self: {"raw_content": "founder raw context"}, processor
    )
    processor._get_page_count_from_extraction = types.MethodType(
        lambda self, *_args, **_kwargs: 1,
        processor,
    )

    async def _fake_convert(self, result, *_args, **_kwargs):
        agent_outputs = result.get("agent_outputs", {}) if isinstance(result, dict) else {}

        def _extract(section: str, field: str) -> list:
            payload = agent_outputs.get(section, {})
            if isinstance(payload, dict):
                value = payload.get(field)
                if isinstance(value, list):
                    return value
            return []

        return types.SimpleNamespace(
            rebellion_framework=types.SimpleNamespace(
                narrative_deconstruction=_extract("rebellion_framework", "narrative_deconstruction")
            ),
            educational_content=types.SimpleNamespace(
                explanations=_extract("educational_content", "explanations")
            ),
            marketing_content=types.SimpleNamespace(
                headlines=_extract("marketing_content", "headlines")
            ),
        )

    processor._convert_to_knowledge_entry = types.MethodType(_fake_convert, processor)

    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("placeholder pdf content", encoding="utf-8")

    entry = asyncio.run(processor.process_document(pdf_path))

    assert entry is not None, "Expected knowledge entry"
    supervisor_context = processor.supervisor.last_context
    assert supervisor_context["raw_client_context"] == "client raw context"
    assert supervisor_context["raw_founder_context"] == "founder raw context"

    snippet = "client raw context | founder raw context"
    assert entry.rebellion_framework.narrative_deconstruction == [snippet]
    assert entry.educational_content.explanations == [snippet]
    assert entry.marketing_content.headlines == [snippet]


def test_entity_extraction_logging_includes_counts(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    processor = MultiAgentProcessor.__new__(MultiAgentProcessor)
    processor.input_dir = tmp_path
    processor.output_file = tmp_path / "out.json"
    processor.temp_file = processor.output_file.with_suffix(".temp")
    processor.context_dir = tmp_path
    processor.supervisor = _StubSupervisor()
    processor.extraction_team = object()
    processor.pdf_extractor = object()
    processor.extraction_tools = object()
    processor.knowledge_base = None
    processor.embedding_ingestion = None
    processor.retry_attempts = 1
    processor.st_louis_context = {
        "demographics": {},
        "clinical_priorities": [],
        "founder_voice": [],
    }

    sample_entities = {
        "biomedical": [{"text": "EntityA"}],
        "clinical": [{"text": "EntityB"}, {"text": "EntityC"}],
        "psychology": ({"text": "EntityD"},),
        "metadata": {"ignored": True},
        "total_entities": {
            "total": 4,
            "biomedical": 1,
            "clinical": 2,
            "psychology": 1,
        },
    }

    class _ExtractionTeamWithPayload:
        async def extract_entities(self, *_args, **_kwargs):
            return sample_entities

    processor.extraction_team = _ExtractionTeamWithPayload()

    async def _fake_extract(self, *_args, **_kwargs):
        return {
            "archival_content": {"full_document_text_markdown": "Document body " * 10}
        }

    async def _fake_context(self, text, document_id):
        return {"document_id": document_id, "document_text": text}

    async def _fake_convert(self, *_args, **_kwargs):
        return types.SimpleNamespace()

    processor._extract_pdf_text = types.MethodType(_fake_extract, processor)
    processor._create_processing_context = types.MethodType(_fake_context, processor)
    processor._convert_to_knowledge_entry = types.MethodType(_fake_convert, processor)
    processor._get_pdf_file_size = types.MethodType(lambda self, _path: 0, processor)
    processor._get_page_count_from_extraction = types.MethodType(
        lambda self, *_args, **_kwargs: 1, processor
    )

    pdf_path = tmp_path / "entity.pdf"
    pdf_path.write_text("pdf", encoding="utf-8")

    with caplog.at_level(logging.INFO):
        entry = asyncio.run(processor.process_document(pdf_path))

    assert entry is not None, "Expected knowledge entry"

    messages = [message for message in caplog.messages if "Entity extraction complete" in message]
    assert messages, "Expected entity extraction log message"
    summary_message = messages[-1]
    assert "total=4" in summary_message
    assert "'biomedical': 1" in summary_message
    assert "'clinical': 2" in summary_message
    assert "'psychology': 1" in summary_message

    supervisor_context = processor.supervisor.last_context
    assert supervisor_context["entity_counts"] == {
        "total": 4,
        "categories": {
            "biomedical": 1,
            "clinical": 2,
            "psychology": 1,
        },
    }
