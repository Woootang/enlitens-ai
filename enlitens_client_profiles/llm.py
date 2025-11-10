"""Helpers around the existing Ollama/vLLM client for profile generation."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, Type

from json_repair import repair_json
from pydantic import BaseModel

from src.synthesis.ollama_client import OllamaClient
from src.utils.settings import get_settings


class ProfileLLMClient:
    """Thin wrapper that exposes a synchronous interface for profile prompts."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        resolved_model = model or settings.llm.model_for("client_profiles")
        self._client = OllamaClient(default_model=resolved_model)
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens or 3072

    async def _generate_async(self, prompt: str, system_prompt: Optional[str]) -> str:
        response = await self._client.generate_response(
            prompt,
            temperature=self._temperature,
            top_p=self._top_p,
            system_prompt=system_prompt,
            response_format="json_object",
            num_predict=self._max_tokens,
        )
        return response.get("response", "")

    def generate_json(self, prompt: str, *, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        text = loop.run_until_complete(self._generate_async(prompt, system_prompt))
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            repaired = repair_json(text, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
            raise

    def generate_structured(
        self,
        prompt: str,
        *,
        response_model: Type[BaseModel],
        system_prompt: Optional[str] = None,
        fallback_prompt: Optional[str] = None,
    ) -> BaseModel:
        loop = asyncio.get_event_loop()
        combined_prompt = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
        result = loop.run_until_complete(
            self._client.generate_structured_response(
                combined_prompt,
                response_model=response_model,
                temperature=self._temperature,
                base_num_predict=min(1024, self._max_tokens),
                max_num_predict=self._max_tokens,
                use_cot_prompt=False,
                fallback_to_unstructured=True,
                fallback_prompt=fallback_prompt,
            )
        )
        if result is None:
            raise ValueError("Structured generation returned no result")
        return result

    async def aclose(self) -> None:
        await self._client.client.close()

