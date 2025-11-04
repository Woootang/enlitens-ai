import asyncio
import json
from typing import Any, Dict, List

import pytest

httpx = pytest.importorskip("httpx")

from src.agents.base_agent import BaseAgent
from src.synthesis.ollama_client import LLMServiceError, OllamaClient, VLLMClient
from src.utils.settings import get_settings, reset_settings_cache


class _DummyAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="Dummy", role="Test")

    async def initialize(self) -> bool:
        self.is_initialized = True
        return True

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"echo": context}

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        return True


class _LLMStubAgent(BaseAgent):
    def __init__(self, client: OllamaClient) -> None:
        super().__init__(name="LLMStub", role="Test", model=client.default_model)
        self.ollama_client = client
        # We bypass initialize to focus on execute behaviour
        self.is_initialized = True

    async def initialize(self) -> bool:
        return True

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return await self.ollama_client.generate_response("ping")

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        return True


@pytest.fixture(autouse=True)
def _clear_settings() -> None:
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.mark.asyncio
async def test_agent_initialization_respects_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL_DEFAULT", "test-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://config-server/v1")
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    reset_settings_cache()

    agent = _DummyAgent()
    assert agent.model == "test-model"
    assert agent.connection_info["base_url"] == "http://config-server/v1"
    settings = get_settings()
    assert settings.llm.base_url == "http://config-server/v1"


@pytest.mark.asyncio
async def test_client_handles_offline_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    transport = httpx.MockTransport(handler)
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("LLM_BASE_URL", "http://offline")
    monkeypatch.setenv("LLM_MODEL_DEFAULT", "offline-model")
    reset_settings_cache()

    client = OllamaClient(transport=transport)
    try:
        healthy = await client.check_connection()
        assert healthy is False
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_vllm_client_targets_versioned_route(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {"count": 0, "path": None}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        calls["path"] = request.url.path
        assert request.url.path == "/v1/chat/completions"
        payload = {
            "choices": [
                {
                    "message": {"content": "ok"},
                    "finish_reason": "stop",
                }
            ]
        }
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    monkeypatch.setenv("LLM_BASE_URL", "http://router.local/v1")
    monkeypatch.setenv("LLM_MODEL_DEFAULT", "test-model")
    reset_settings_cache()

    client = VLLMClient(transport=transport)
    try:
        result = await client.generate_response("ping")
    finally:
        await client.close()

    assert result["response"] == "ok"
    assert calls["count"] == 1
    assert calls["path"] == "/v1/chat/completions"


@pytest.mark.asyncio
async def test_vllm_client_auto_continuation(monkeypatch: pytest.MonkeyPatch) -> None:
    responses: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        call_index = len(responses)
        responses.append(payload)
        if call_index == 0:
            assert payload["max_tokens"] >= 256
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "chunk-1"},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {"completion_tokens": payload["max_tokens"] - 128},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "chunk-2"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 512},
            },
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setenv("LLM_BASE_URL", "http://router.local/v1")
    monkeypatch.setenv("LLM_MODEL_DEFAULT", "test-model")
    reset_settings_cache()

    client = VLLMClient(transport=transport)
    try:
        result = await client.generate_response("long prompt")
    finally:
        await client.close()

    assert result["response"] == "chunk-1chunk-2"
    assert result["done"] is True
    assert len(responses) == 2
    assert responses[1]["messages"][-2]["role"] == "assistant"


@pytest.mark.asyncio
async def test_health_check_falls_back_to_secondary_url(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.host == "primary.local":
            return httpx.Response(404, json={"error": "missing"})
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens": 1},
                },
            )
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    monkeypatch.setenv("LLM_BASE_URL", "http://primary.local")
    monkeypatch.setenv("LLM_BASE_URL_SECONDARY", "http://secondary.local")
    monkeypatch.setenv("LLM_MODEL_DEFAULT", "test-model")
    reset_settings_cache()

    client = OllamaClient(transport=transport)
    try:
        healthy = await client.check_connection()
        assert healthy is True
        assert client.base_url == "http://secondary.local"
        resolved = client.resolved_chat_endpoint
        assert resolved is not None and resolved.endswith("/chat/completions")

        result = await client.generate_response("hello")
        assert result["response"] == "ok"
    finally:
        await client.close()

    assert any("primary.local" in call for call in calls)
    assert any("secondary.local" in call for call in calls)


@pytest.mark.asyncio
async def test_agent_raises_llm_service_error_on_client_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setenv("LLM_BASE_URL", "http://broken.local")
    monkeypatch.delenv("LLM_BASE_URL_SECONDARY", raising=False)
    monkeypatch.setenv("LLM_MODEL_DEFAULT", "test-model")
    reset_settings_cache()

    telemetry_events: List[Dict[str, Any]] = []

    def fake_record_event(*args: Any, **kwargs: Any) -> None:
        telemetry_events.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(
        "src.synthesis.ollama_client.telemetry_recorder.record_event",
        fake_record_event,
    )

    client = OllamaClient(transport=transport)
    agent = _LLMStubAgent(client)

    with pytest.raises(LLMServiceError) as excinfo:
        await agent.execute({"document_id": "doc-1"})

    assert "broken.local" in excinfo.value.endpoint
    assert telemetry_events, "expected telemetry to capture the client error"
    details = telemetry_events[-1]["kwargs"].get("details", {})
    assert details.get("status") == 404
    assert "chat/completions" in details.get("endpoint", "")

    await client.close()


@pytest.mark.asyncio
async def test_vllm_client_reprobes_after_chat_404(monkeypatch: pytest.MonkeyPatch) -> None:
    state: Dict[str, Any] = {
        "health_checks": [],
        "requests": [],
        "v1_available": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and (
            path.endswith("/chat/completions") or path.endswith("/v1/chat/completions")
        ):
            payload = json.loads(request.content.decode("utf-8"))
            is_health_check = "temperature" not in payload
            if is_health_check:
                state["health_checks"].append(path)
                if not state["v1_available"]:
                    if path.endswith("/chat/completions"):
                        return httpx.Response(200, json={"ok": True})
                    return httpx.Response(404, json={"error": "missing"})
                if path.endswith("/v1/chat/completions"):
                    return httpx.Response(200, json={"ok": True})
                return httpx.Response(404, json={"error": "moved"})

            state["requests"].append(path)
            if len(state["requests"]) == 1:
                state["v1_available"] = True
                return httpx.Response(404, json={"error": "missing"})

            assert path.endswith("/v1/chat/completions")
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "recovered"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens": 1},
                },
            )

        raise AssertionError(f"Unexpected request path: {path}")

    transport = httpx.MockTransport(handler)
    monkeypatch.setenv("LLM_BASE_URL", "http://router.local")
    monkeypatch.setenv("LLM_MODEL_DEFAULT", "test-model")
    reset_settings_cache()

    client = VLLMClient(transport=transport)
    try:
        result = await client.generate_response("ping")
    finally:
        await client.close()

    assert result["response"] == "recovered"
    assert state["requests"] == ["/chat/completions", "/v1/chat/completions"]
    assert any(path.endswith("/v1/chat/completions") for path in state["health_checks"])


@pytest.mark.asyncio
async def test_vllm_client_structured_fallback_on_persistent_404(monkeypatch: pytest.MonkeyPatch) -> None:
    state: Dict[str, Any] = {
        "requests": [],
        "health_checks": 0,
        "fail_mode": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/v1/chat/completions"):
            payload = json.loads(request.content.decode("utf-8"))
            is_health_check = "temperature" not in payload
            state["health_checks"] += 1
            if is_health_check and not state["fail_mode"]:
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(404, json={"error": "missing"})

        if request.method == "POST" and request.url.path.endswith("/chat/completions"):
            payload = json.loads(request.content.decode("utf-8"))
            is_health_check = "temperature" not in payload
            if is_health_check:
                state["health_checks"] += 1
                return httpx.Response(404, json={"error": "missing"})

            state["fail_mode"] = True
            state["requests"].append(request.url.path)
            return httpx.Response(404, json={"error": "missing"})

        raise AssertionError(f"Unexpected request path: {request.url.path}")

    transport = httpx.MockTransport(handler)
    monkeypatch.setenv("LLM_BASE_URL", "http://router.local/v1")
    monkeypatch.setenv("LLM_MODEL_DEFAULT", "test-model")
    reset_settings_cache()

    client = VLLMClient(transport=transport)
    try:
        payload = await client.generate_response("ping")
    finally:
        await client.close()

    assert payload["response"] == ""
    raw = payload.get("raw", {})
    assert raw.get("model") == ""
    telemetry = raw.get("telemetry", {})
    assert telemetry.get("status") == 404
    assert telemetry.get("message")
    assert state["requests"] == ["/chat/completions"]
