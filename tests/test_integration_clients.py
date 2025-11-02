import asyncio
from typing import Any, Dict

import pytest

httpx = pytest.importorskip("httpx")

from src.agents.base_agent import BaseAgent
from src.synthesis.ollama_client import OllamaClient, VLLMClient
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
