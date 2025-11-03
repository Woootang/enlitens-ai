import sys
import types

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
from src.models.enlitens_schemas import WebsiteCopy


def test_website_copy_schema_excludes_deprecated_fields():
    fields = set(WebsiteCopy.model_fields.keys())
    assert "feature_descriptions" not in fields
    assert "benefit_statements" not in fields
    assert "service_descriptions" not in fields
    assert {"about_sections", "faq_content", "topic_ideas"}.issubset(fields)


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

    result = await agent._generate_website_copy(
        clinical_data={"insight": "value"},
        context={"document_text": "Neuroscience insights."},
    )

    prompt = captured_prompt.get("prompt", "")
    assert prompt, "Expected prompt to be captured"
    assert "feature_descriptions" not in prompt
    assert "benefit_statements" not in prompt
    assert "service_descriptions" not in prompt
    assert result.topic_ideas == ["idea"]
