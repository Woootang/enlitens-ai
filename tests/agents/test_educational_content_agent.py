import asyncio
import sys
import types


if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def aclose(self):
            pass

        async def request(self, *args, **kwargs):
            raise NotImplementedError

    class _HTTPStatusError(Exception):
        def __init__(self, message, request=None, response=None):
            super().__init__(message)
            self.request = request
            self.response = response

    class _Response:
        def __init__(self, *args, **kwargs):
            self.status_code = kwargs.get("status_code", 200)
            self.request = kwargs.get("request")

        def raise_for_status(self):
            pass

    httpx_stub.AsyncClient = _AsyncClient
    httpx_stub.AsyncBaseTransport = object
    httpx_stub.RequestError = Exception
    httpx_stub.HTTPStatusError = _HTTPStatusError
    httpx_stub.Response = _Response
    sys.modules["httpx"] = httpx_stub

if "json_repair" not in sys.modules:
    json_repair_stub = types.ModuleType("json_repair")

    def _repair_json(payload):
        return payload

    json_repair_stub.repair_json = _repair_json
    sys.modules["json_repair"] = json_repair_stub

from src.agents.educational_content_agent import EducationalContentAgent


class DummyEducationalOllamaClient:
    def __init__(self, payload):
        self._payload = payload

    async def generate_structured_response(self, *, response_model, **kwargs):
        if self._payload is None:
            return None
        return response_model(**self._payload)

    def clone_with_model(self, model):
        return self


def _build_agent_with_payload(payload):
    agent = EducationalContentAgent()
    agent.is_initialized = True
    agent.ollama_client = DummyEducationalOllamaClient(payload)
    return agent


def test_educational_content_agent_auto_padding_from_research():
    payload = {
        "explanations": ["Short explanation"],
        "examples": [],
        "analogies": ["Analogy one", "Analogy two"],
        "definitions": ["Definition"],
        "processes": ["Process"],
        "comparisons": [],
        "visual_aids": ["Visual idea"],
        "learning_objectives": ["Objective"],
    }

    context = {
        "document_text": "doc",
        "science_data": {
            "research_content": {
                "findings": ["Finding A", "Finding B"],
                "implications": ["Implication"],
            }
        },
    }

    agent = _build_agent_with_payload(payload)

    result = asyncio.run(agent.process(context))

    content = result["educational_content"]

    for field in (
        "explanations",
        "examples",
        "analogies",
        "definitions",
        "processes",
        "comparisons",
        "visual_aids",
        "learning_objectives",
    ):
        assert len(content[field]) >= 5
        assert all(isinstance(item, str) for item in content[field])

    assert result["generation_quality"] == "medium"
    assert result["auto_padded_fields"]
    assert asyncio.run(agent.validate_output(result))


def test_educational_content_agent_padding_on_null_response():
    context = {
        "document_text": "doc",
        "science_data": {
            "research_content": {
                "findings": ["Fallback finding"],
            }
        },
    }

    agent = _build_agent_with_payload(None)

    result = asyncio.run(agent.process(context))

    content = result["educational_content"]

    for field in content:
        assert len(content[field]) >= 5

    assert result["generation_quality"] == "low"
    assert asyncio.run(agent.validate_output(result))
