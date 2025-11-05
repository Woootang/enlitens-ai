import asyncio
import sys
import types

import pytest

langgraph_module = types.ModuleType("langgraph")
langgraph_graph_module = types.ModuleType("langgraph.graph")
langgraph_graph_module.StateGraph = object  # type: ignore[attr-defined]
langgraph_graph_module.END = object()
langgraph_module.graph = langgraph_graph_module
sys.modules.setdefault("langgraph", langgraph_module)
sys.modules.setdefault("langgraph.graph", langgraph_graph_module)

httpx_module = types.ModuleType("httpx")


class _DummyHTTPXClient:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, *args, **kwargs):
        response = types.SimpleNamespace()
        response.raise_for_status = lambda: None
        return response


httpx_module.Client = _DummyHTTPXClient
httpx_module.RequestError = Exception
httpx_module.HTTPStatusError = Exception
sys.modules.setdefault("httpx", httpx_module)

json_repair_module = types.ModuleType("json_repair")
json_repair_module.repair_json = lambda text: text
sys.modules.setdefault("json_repair", json_repair_module)

yaml_module = types.ModuleType("yaml")
yaml_module.safe_load = lambda content: {}
sys.modules.setdefault("yaml", yaml_module)

pydantic_module = types.ModuleType("pydantic")


class _DummyBaseModel:
    def __init__(self, *args, **kwargs):
        pass


pydantic_module.BaseModel = _DummyBaseModel
pydantic_module.Field = lambda *args, **kwargs: None
pydantic_module.field_validator = lambda *args, **kwargs: (lambda func: func)


class _DummyValidationInfo:
    pass


class _DummyHttpUrl(str):
    pass


pydantic_module.ValidationInfo = _DummyValidationInfo
pydantic_module.HttpUrl = _DummyHttpUrl
pydantic_module.ValidationError = type("ValidationError", (Exception,), {})
sys.modules.setdefault("pydantic", pydantic_module)

tenacity_module = types.ModuleType("tenacity")


class _DummyAsyncRetrying:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        async def _generator():
            yield None

        return _generator()


def _identity(*args, **kwargs):
    return None


tenacity_module.AsyncRetrying = _DummyAsyncRetrying
tenacity_module.retry_if_exception = _identity
tenacity_module.stop_after_attempt = _identity
tenacity_module.wait_exponential = _identity
sys.modules.setdefault("tenacity", tenacity_module)

from process_multi_agent_corpus import MultiAgentProcessor
from src.data.locality_loader import LocalityRecord


def _locality(name: str, jurisdiction: str, income: str, descriptors: str) -> LocalityRecord:
    return LocalityRecord(
        name=name,
        jurisdiction=jurisdiction,
        median_income_band=income,
        demographic_descriptors=descriptors,
        landmark_schools=[],
        youth_sports_leagues=[],
        community_centers=[],
        health_resources=[],
        signature_eateries=[],
    )


def test_theme_landscape_detects_gaps_and_contrasts():
    processor = MultiAgentProcessor.__new__(MultiAgentProcessor)
    processor.intake_registry = {
        "weighted_theme_signals": [
            {
                "theme": "Trauma & complex PTSD",
                "frequency": 2,
                "weighted_frequency": 5.0,
                "locality_tags": [("North STL", 3), ("Delmar Loop", 1)],
            },
            {
                "theme": "Anxiety & regulation",
                "frequency": 1,
                "weighted_frequency": 2.0,
                "locality_tags": [("Clayton", 1)],
            },
        ]
    }
    processor.transcript_registry = {
        "weighted_theme_signals": [
            {
                "theme": "Trauma & complex PTSD",
                "frequency": 1,
                "weighted_frequency": 2.5,
                "locality_tags": [("Clayton", 2)],
            }
        ]
    }
    processor.health_report_summary = {
        "weighted_theme_signals": [
            {
                "theme": "Community violence",
                "frequency": 3,
                "weighted_frequency": 6.0,
                "locality_tags": [("North STL", 2)],
            },
            {
                "theme": "Trauma & complex PTSD",
                "frequency": 1,
                "weighted_frequency": 1.5,
                "locality_tags": [("North STL", 1)],
            },
        ]
    }
    processor.locality_reference = {
        "north stl": _locality(
            "North STL",
            "St. Louis city",
            "$20k-$30k",
            "Neighborhoods north of the Delmar Divide",
        ),
        "clayton": _locality(
            "Clayton",
            "St. Louis County",
            "$110k-$150k",
            "Affluent suburb with significant wealth concentration",
        ),
        "delmar loop": _locality(
            "Delmar Loop",
            "St. Louis city",
            "$40k-$60k",
            "Delmar cultural district bridging the divide",
        ),
    }

    landscape = MultiAgentProcessor._compile_theme_landscape(processor)

    dominant = {entry["theme"]: entry for entry in landscape["dominant_themes"]}
    assert "Trauma & complex PTSD" in dominant
    assert pytest.approx(dominant["Trauma & complex PTSD"]["source_breakdown"]["intake"]["weighted_frequency"], rel=1e-3) == 5.0

    gaps = {entry["theme"]: entry for entry in landscape["theme_gaps"]}
    assert "Community violence" in gaps
    assert gaps["Community violence"]["present_sources"] == ["health_report"]
    assert set(gaps["Community violence"]["missing_sources"]) == {"intake", "transcript"}

    flags = {entry["theme"]: entry for entry in landscape["socioeconomic_contrast_flags"]}
    assert "Trauma & complex PTSD" in flags
    contrast = flags["Trauma & complex PTSD"]
    assert contrast["estimated_income_gap"] is not None
    assert contrast["estimated_income_gap"] >= 80_000
    assert "income_gap" in contrast["indicators"]
    assert "delmar_indicator" in contrast["indicators"]
    flagged_localities = {profile["name"] for profile in contrast["locality_profiles"]}
    assert {"North STL", "Clayton"}.issubset(flagged_localities)


def test_processing_context_exposes_theme_landscape(monkeypatch):
    processor = MultiAgentProcessor.__new__(MultiAgentProcessor)
    processor.st_louis_context = {
        "demographics": {
            "mental_health_challenges": ["Challenge"],
            "socioeconomic_factors": [],
        },
        "clinical_priorities": ["Priority"],
        "founder_voice": ["Voice"],
    }
    processor.intake_registry = {"weighted_theme_signals": []}
    processor.transcript_registry = {"weighted_theme_signals": []}
    processor.health_report_summary = {"weighted_theme_signals": []}
    processor.regional_atlas = {}
    processor.locality_reference = {}
    processor.theme_landscape = {
        "dominant_themes": [{"theme": "Trauma & complex PTSD", "total_weight": 3.0}],
        "theme_gaps": [{"theme": "Community violence", "missing_sources": ["intake", "transcript"]}],
        "socioeconomic_contrast_flags": [{"theme": "Trauma & complex PTSD", "indicators": ["income_gap"]}],
    }

    monkeypatch.setattr(processor, "_analyze_client_insights", lambda: {})
    monkeypatch.setattr(processor, "_analyze_founder_insights", lambda: {})

    context = asyncio.run(
        MultiAgentProcessor._create_processing_context(processor, "Sample text from North STL.", "doc-001")
    )

    assert context["dominant_themes"] == processor.theme_landscape["dominant_themes"]
    assert context["theme_gaps"] == processor.theme_landscape["theme_gaps"]
    assert context["socioeconomic_contrast_flags"] == processor.theme_landscape["socioeconomic_contrast_flags"]
