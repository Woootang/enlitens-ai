from unittest.mock import MagicMock
import pytest
import asyncio
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))


class _Stub:
    def __init__(self, *args, **kwargs):
        pass


def ensure_module(name: str, attrs: dict) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    for attr_name, attr_value in attrs.items():
        setattr(module, attr_name, attr_value)
    sys.modules[name] = module


ensure_module("src.agents.supervisor_agent", {"SupervisorAgent": _Stub})
ensure_module("src.models.enlitens_schemas", {
    "EnlitensKnowledgeBase": _Stub,
    "EnlitensKnowledgeEntry": _Stub,
})
ensure_module("src.extraction.enhanced_pdf_extractor", {"EnhancedPDFExtractor": _Stub})
ensure_module("src.agents.extraction_team", {"ExtractionTeam": _Stub})
ensure_module(
    "src.utils.enhanced_logging",
    {
        "setup_enhanced_logging": lambda *args, **kwargs: None,
        "log_startup_banner": lambda *args, **kwargs: None,
    },
)

from src.extraction.enhanced_extraction_tools import (
    EnhancedExtractionTools,
    TopicDiscoveryResult,
)
from process_multi_agent_corpus import MultiAgentProcessor


@pytest.fixture
def tools(monkeypatch):
    tool = EnhancedExtractionTools(device="cpu")
    # Prevent heavy model loading during tests
    stub_model = MagicMock()
    stub_model.fit_transform.return_value = ([0, 1], [[0.7, 0.3], [0.6, 0.4]])
    stub_model.get_topic_info.return_value = {"Topic": [0, 1]}
    stub_model.get_topic.return_value = [("neuroscience", 0.5), ("therapy", 0.4)]

    def fake_load():
        tool.bertopic_model = stub_model

    monkeypatch.setattr(tool, "_load_bertopic", fake_load)
    return tool


def test_discover_topics_returns_structured_result(tools):
    texts = ["Neuroscience drives therapy.", "Clients need support."]

    result = tools.discover_topics(texts)

    assert isinstance(result, TopicDiscoveryResult)
    assert result.topic_keywords
    assert result.metadata["topic_count"] == 2
    assert result.probabilities == [[0.7, 0.3], [0.6, 0.4]]


def test_analyze_sentiment_uses_vader():
    tool = EnhancedExtractionTools(device="cpu")

    positive = tool.analyze_sentiment("I absolutely love how well this works!")
    negative = tool.analyze_sentiment("I absolutely hate how terrible this is.")

    if positive["compound"] == 0.0 and negative["compound"] == 0.0:
        pytest.skip("Sentiment analyzer unavailable in test environment")

    assert positive["compound"] > negative["compound"]
    assert positive["compound"] > 0
    assert negative["compound"] < 0


def test_extract_client_pain_points_uses_statistical_thresholds(monkeypatch, tools):
    scores = iter([-0.5, -0.5, -0.5])

    def fake_sentiment(_):
        return {"compound": next(scores)}

    monkeypatch.setattr(tools, "analyze_sentiment", fake_sentiment)
    monkeypatch.setattr(
        tools,
        "discover_topics",
        lambda texts: TopicDiscoveryResult(topic_keywords={0: ["stress"]}, metadata={"mock": True}),
    )

    result = tools.extract_client_pain_points([
        "Client feels overwhelmed by stress.",
        "Client is anxious about work.",
        "Client struggles with trauma.",
    ])

    assert result["high_priority_count"] == 3
    assert result["topic_keywords"] == {0: ["stress"]}
    assert result["topic_metadata"] == {"mock": True}


def test_processing_context_persists_insights(monkeypatch):
    processor = object.__new__(MultiAgentProcessor)
    processor.st_louis_context = {
        "demographics": {
            "mental_health_challenges": ["challenge"],
            "socioeconomic_factors": [],
        },
        "clinical_priorities": ["priority"],
        "founder_voice": ["voice"],
    }

    client_analysis = {
        "topic_modeling": {"topic_keywords": {0: ["focus"]}},
        "sentiment_analysis": {"overall_compound": -0.2},
        "pain_points": ["stress"],
        "key_themes": ["executive function"],
    }
    founder_analysis = {
        "topic_modeling": {"topic_keywords": {1: ["hope"]}},
        "sentiment_analysis": {"overall_compound": 0.3},
        "voice_characteristics": {"directness": 2},
        "key_messages": ["Neuroplasticity offers hope"],
    }

    def fake_client():
        return client_analysis

    def fake_founder():
        return founder_analysis

    monkeypatch.setattr(processor, "_analyze_client_insights", fake_client)
    monkeypatch.setattr(processor, "_analyze_founder_insights", fake_founder)

    context = asyncio.run(
        MultiAgentProcessor._create_processing_context(processor, "text", "doc-1")
    )

    assert context["client_insights"]["topic_modeling"] == client_analysis["topic_modeling"]
    assert context["client_insights"]["sentiment_analysis"] == client_analysis["sentiment_analysis"]
    assert context["founder_insights"]["topic_modeling"] == founder_analysis["topic_modeling"]
    assert context["insight_registry"]["client"] == client_analysis
    assert context["insight_registry"]["founder"] == founder_analysis
