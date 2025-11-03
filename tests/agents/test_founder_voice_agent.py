import sys
import types
from typing import Any, Dict, List

import pytest

pydantic = pytest.importorskip("pydantic")


if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")

    class _DummyResponse:
        def __init__(self, status_code: int = 200):
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"HTTP error: {self.status_code}")

    class _DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def request(self, *args, **kwargs):
            return _DummyResponse()

    class RequestError(Exception):
        pass

    class HTTPStatusError(Exception):
        def __init__(self, *args, response=None, **kwargs):
            super().__init__(*args)
            self.response = response or _DummyResponse(status_code=500)

    class AsyncBaseTransport:  # pragma: no cover - simple stub
        pass

    httpx_stub.AsyncClient = _DummyAsyncClient
    httpx_stub.RequestError = RequestError
    httpx_stub.HTTPStatusError = HTTPStatusError
    httpx_stub.Response = _DummyResponse
    httpx_stub.AsyncBaseTransport = AsyncBaseTransport

    sys.modules["httpx"] = httpx_stub

if "json_repair" not in sys.modules:
    json_repair_stub = types.ModuleType("json_repair")

    def repair_json(text: str, *_, **__):  # pragma: no cover - simple stub
        return text

    json_repair_stub.repair_json = repair_json
    sys.modules["json_repair"] = json_repair_stub

from src.agents.founder_voice_agent import FounderVoiceAgent
from src.models.enlitens_schemas import MarketingContent, WebsiteCopy, SocialMediaContent


def test_website_copy_schema_excludes_deprecated_fields():
    fields = set(WebsiteCopy.model_fields.keys())
    assert "feature_descriptions" not in fields
    assert "benefit_statements" not in fields
    assert "service_descriptions" not in fields
    assert {"about_sections", "faq_content", "topic_ideas"}.issubset(fields)


def test_social_media_schema_updates_remove_reels_and_carousels():
    fields = set(SocialMediaContent.model_fields.keys())
    assert "reel_ideas" not in fields
    assert "carousel_content" not in fields
    assert {"story_ideas", "hashtags", "poll_questions"}.issubset(fields)


def test_social_media_quotes_require_source_matches():
    payload = {
        "post_ideas": [],
        "captions": [],
        "quotes": ['"Neurodiversity celebrates differences." — [Source 1]'],
        "hashtags": [],
        "story_ideas": [],
        "poll_questions": [],
    }

    validated = SocialMediaContent.model_validate(
        payload,
        context={
            "source_text": "Neurodiversity celebrates differences. Regulation is key.",
            "source_segments": [
                "Neurodiversity celebrates differences. Regulation is key."
            ],
        },
    )

    assert validated.quotes == ['"Neurodiversity celebrates differences." — [Source 1]']


def test_social_media_quotes_raise_for_missing_sources():
    payload = {
        "post_ideas": [],
        "captions": [],
        "quotes": ['"Invented insight." — [Source 9]'],
        "hashtags": [],
        "story_ideas": [],
        "poll_questions": [],
    }

    with pytest.raises(ValueError):
        SocialMediaContent.model_validate(
            payload,
            context={
                "source_text": "Neurodiversity celebrates differences.",
                "source_segments": ["Neurodiversity celebrates differences."],
            },
        )


@pytest.mark.asyncio
async def test_generate_website_copy_prompt_omits_deprecated_sections(monkeypatch):
    agent = FounderVoiceAgent()
    captured_prompt = {}

    async def fake_structured_generation(prompt, response_model, context, suffix, **kwargs):
        captured_prompt["prompt"] = prompt
        assert response_model is WebsiteCopy
        return WebsiteCopy(
            about_sections=["about"],
            faq_content=["faq"],
            topic_ideas=["idea"],
        )

    monkeypatch.setattr(agent, "_structured_generation", fake_structured_generation)

    client_segments = ["I can't keep my ADHD brain organized long enough to finish a project."]
    client_summary = agent._render_client_insights(client_segments)

    result = await agent._generate_website_copy(
        clinical_data={"insight": "value"},
        context={
            "document_text": "Neuroscience insights.",
            "retrieved_passages": [
                {
                    "text": "St. Louis clients report 3x sensory load in open office plans.",
                    "chunk_id": "chunk-1",
                }
            ],
        },
        client_insights_summary=client_summary,
        client_insight_segments=client_segments,
    )

    prompt = captured_prompt.get("prompt", "")
    assert prompt, "Expected prompt to be captured"
    assert "feature_descriptions" not in prompt
    assert "benefit_statements" not in prompt
    assert "service_descriptions" not in prompt
    assert "3x sensory load" in prompt
    assert result.topic_ideas == ["idea"]


@pytest.mark.asyncio
async def test_generate_social_media_prompt_uses_sources_and_quote_rules(monkeypatch):
    agent = FounderVoiceAgent()
    captured: Dict[str, Any] = {}

    async def fake_structured_generation(prompt, response_model, context, suffix, **kwargs):
        captured["prompt"] = prompt
        captured["validation_context"] = kwargs.get("validation_context")
        assert response_model is SocialMediaContent
        return SocialMediaContent(
            post_ideas=["idea"],
            captions=["caption"],
            quotes=['"Neurodiversity celebrates differences." — [Source 1]'],
            hashtags=["#tag"],
            story_ideas=["story"],
            poll_questions=["poll"],
        )

    monkeypatch.setattr(agent, "_structured_generation", fake_structured_generation)

    context = {
        "document_text": "Neurodiversity celebrates differences. Regulation is key for resilience.",
        "final_context": {
            "research_content": {
                "findings": [
                    "Regulation is key for resilience and executive function."
                ]
            }
        },
        "retrieved_context": [
            "Neurodiversity celebrates differences and nervous systems adapt under pressure."
        ],
        "retrieved_passages": [
            {
                "text": "St. Louis ADHD brains juggle relentless sensory input, not laziness.",
                "chunk_id": "chunk-88",
            }
        ],
    }

    client_segments = [
        "My nervous system never powers down—it feels like I'm running on emergency mode 24/7.",
        "Executive dysfunction keeps wrecking my work deadlines.",
    ]
    client_summary = agent._render_client_insights(client_segments)

    result = await agent._generate_social_media_content(
        clinical_data={"insight": "value"},
        context=context,
        client_insights_summary=client_summary,
        client_insight_segments=client_segments,
    )

    prompt = captured.get("prompt", "")
    assert prompt, "Expected prompt to be captured"
    assert "reel" not in prompt.lower()
    assert "carousel" not in prompt.lower()
    assert "SOURCE MATERIAL" in prompt
    assert "[Source 1]" in prompt
    assert "relentless sensory input" in prompt
    assert "RETRIEVED PASSAGES" not in prompt
    assert "QUOTE REQUIREMENTS" in prompt
    assert "Client Intake Insights" in prompt
    assert "emergency mode 24/7" in prompt

    validation_context = captured.get("validation_context") or {}
    assert "Neurodiversity celebrates differences." in validation_context.get("source_text", "")
    assert validation_context.get("source_citation_map", {}).get("Source 1")
    assert "quote_missing_note" in validation_context
    assert "quote_validation_telemetry" in validation_context
    assert result.quotes == ['"Neurodiversity celebrates differences." — [Source 1]']


@pytest.mark.asyncio
async def test_marketing_prompt_uses_client_insights_language(monkeypatch):
    agent = FounderVoiceAgent()
    captured_prompt: Dict[str, Any] = {}

    async def fake_structured_generation(prompt, response_model, context, suffix, **kwargs):
        captured_prompt["prompt"] = prompt
        assert response_model is MarketingContent
        return MarketingContent(
            headlines=["headline"],
            taglines=["tagline"],
            value_propositions=["value"],
            benefits=["benefit"],
            pain_points=["pain"],
        )

    monkeypatch.setattr(agent, "_structured_generation", fake_structured_generation)

    client_segments = [
        "My executive function falls apart after lunch.",
        "Sensory overload at the office makes me shut down.",
    ]
    client_summary = agent._render_client_insights(client_segments)

    await agent._generate_marketing_content(
        clinical_data={},
        context={
            "document_text": "Brain-based interventions.",
            "retrieved_passages": [
                {
                    "text": "Clients describe executive shutdown when sensory input spikes.",
                    "chunk_id": "chunk-55",
                }
            ],
        },
        client_insights_summary=client_summary,
        client_insight_segments=client_segments,
    )

    prompt = captured_prompt.get("prompt", "")
    assert "executive function falls apart" in prompt
    assert "Sensory overload at the office" in prompt
    assert "No new intake insights" not in prompt
    assert "executive shutdown" in prompt


@pytest.mark.asyncio
async def test_marketing_prompt_falls_back_to_static_challenges(monkeypatch):
    agent = FounderVoiceAgent()
    captured_prompt: Dict[str, Any] = {}

    async def fake_structured_generation(prompt, response_model, context, suffix, **kwargs):
        captured_prompt["prompt"] = prompt
        return MarketingContent(
            headlines=["headline"],
            taglines=["tagline"],
            value_propositions=["value"],
            benefits=["benefit"],
            pain_points=["pain"],
        )

    monkeypatch.setattr(agent, "_structured_generation", fake_structured_generation)

    client_segments: list[str] = []
    client_summary = agent._render_client_insights(client_segments)

    await agent._generate_marketing_content(
        clinical_data={},
        context={"document_text": "Brain-based interventions."},
        client_insights_summary=client_summary,
        client_insight_segments=client_segments,
    )

    prompt = captured_prompt.get("prompt", "")
    assert "No retrieved passages" in prompt
    assert "No new intake insights provided" in prompt
    for challenge in agent.client_challenges[:2]:
        assert challenge in prompt


def test_social_media_quote_validation_accepts_retrieved_passage(monkeypatch):
    telemetry_calls: List[Dict[str, Any]] = []

    def fake_telemetry(logger_method, message, *args, **kwargs):
        telemetry_calls.append({"message": message, "kwargs": kwargs})

    monkeypatch.setattr(
        "src.models.enlitens_schemas.log_with_telemetry",
        fake_telemetry,
    )

    payload = {
        "post_ideas": [],
        "captions": [],
        "quotes": ['"St. Louis ADHD brains juggle relentless sensory input." — [Source 1]'],
        "hashtags": [],
        "story_ideas": [],
        "poll_questions": [],
    }

    context = {
        "source_text": "Neurodiversity celebrates differences.",
        "source_segments": [
            "St. Louis ADHD brains juggle relentless sensory input, not laziness."
        ],
        "source_citation_map": {
            "Source 1": "St. Louis ADHD brains juggle relentless sensory input, not laziness."
        },
        "quote_validation_telemetry": {
            "agent": "founder_voice_agent",
            "doc_id": "doc-success",
        },
    }

    validated = SocialMediaContent.model_validate(payload, context=context)
    assert validated.quotes == payload["quotes"]
    assert telemetry_calls == []


def test_social_media_quote_validation_degrades_without_evidence(monkeypatch):
    telemetry_calls: List[Dict[str, Any]] = []

    def fake_telemetry(logger_method, message, *args, **kwargs):
        telemetry_calls.append({"message": message, "kwargs": kwargs})

    monkeypatch.setattr(
        "src.models.enlitens_schemas.log_with_telemetry",
        fake_telemetry,
    )

    payload = {
        "post_ideas": [],
        "captions": [],
        "quotes": ['"Invented claim." — [Source 1]'],
        "hashtags": [],
        "story_ideas": [],
        "poll_questions": [],
    }

    fallback_note = "Evidence unavailable: no source matched this quote."

    context = {
        "source_text": "",
        "source_segments": [],
        "source_citation_map": {},
        "quote_missing_note": fallback_note,
        "quote_validation_telemetry": {
            "agent": "founder_voice_agent",
            "doc_id": "doc-missing",
        },
    }

    validated = SocialMediaContent.model_validate(payload, context=context)
    assert validated.quotes == [fallback_note]
    assert telemetry_calls, "Expected telemetry to be recorded for missing evidence"
    logged = telemetry_calls[0]
    assert logged["kwargs"].get("impact") == "quote_validation_missing_evidence"
