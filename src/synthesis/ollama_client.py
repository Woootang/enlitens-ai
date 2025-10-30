"""vLLM Client with structured output helpers and prompt prefix caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import httpx
from json_repair import repair_json
from pydantic import BaseModel

from .prompts import (
    TEMPERATURE_FACTUAL,
    TEMPERATURE_CREATIVE,
    get_full_system_prompt,
)

logger = logging.getLogger(__name__)

VLLM_DEFAULT_URL = "http://localhost:8000/v1"
VLLM_DEFAULT_MODEL = "qwen2.5-32b-instruct-q4_k_m"
MONITORING_MODEL = "qwen2.5-3b-instruct-q4_k_m"


class VLLMClient:
    """Async helper around a vLLM OpenAI-compatible server.

    The client keeps feature parity with the original Ollama integration while
    unlocking vLLM-specific optimisations:

    * Paged attention and FlashAttention are assumed to be configured on the
      server process; ``gpu_memory_utilization`` is exposed for monitoring.
    * Prompt prefix caching is enabled for repeated system messages to minimise
      prefill latency.
    * ``benchmark_batch_sizes`` can be used to exercise continuous batching with
      batch sizes 8/16/24 as required by operations run-books.
    """

    def __init__(
        self,
        base_url: str = VLLM_DEFAULT_URL,
        default_model: str = VLLM_DEFAULT_MODEL,
        *,
        timeout_seconds: float = 900.0,
        enable_prefix_caching: bool = True,
        gpu_memory_utilization: float = 0.92,
        continuous_batch_sizes: Optional[Iterable[int]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_seconds)
        self.default_model = default_model
        self.enable_prefix_caching = enable_prefix_caching
        self.gpu_memory_utilization_target = gpu_memory_utilization
        self.continuous_batch_sizes = list(continuous_batch_sizes or (8, 16, 24))
        self._prefix_cache: Dict[str, str] = {}
        logger.info(
            "VLLMClient initialised with base_url=%s model=%s gpu_utilisation≈%.2f",
            self.base_url,
            self.default_model,
            self.gpu_memory_utilization_target,
        )

    def clone_with_model(self, model: str) -> "VLLMClient":
        clone = VLLMClient(
            self.base_url,
            default_model=model,
            timeout_seconds=self.client.timeout.read,
            enable_prefix_caching=self.enable_prefix_caching,
            gpu_memory_utilization=self.gpu_memory_utilization_target,
            continuous_batch_sizes=self.continuous_batch_sizes,
        )
        clone.client = self.client
        return clone

    def _resolve_model(self, model: Optional[str]) -> str:
        return model or getattr(self, "default_model", VLLM_DEFAULT_MODEL)

    def _prefix_cache_key(self, system_prompt: str, model: str) -> str:
        digest = hashlib.sha256(f"{model}::{system_prompt}".encode("utf-8")).hexdigest()
        self._prefix_cache.setdefault(digest, system_prompt)
        return digest

    async def generate_response(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        system_prompt: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a chat completion via vLLM."""

        model_name = self._resolve_model(model)
        if system_prompt is None:
            system_prompt = get_full_system_prompt(
                content_type="factual" if temperature <= 0.4 else "creative"
            )

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": num_predict or 2048,
        }

        # vLLM supports passing implementation-specific settings via extra_body.
        extra_body: Dict[str, Any] = {
            "cache_prompt": False,
            "best_of": 1,
            "logprobs": None,
            "stream": False,
        }

        if self.enable_prefix_caching and system_prompt:
            cache_key = self._prefix_cache_key(system_prompt, model_name)
            extra_body.update({
                "cache_prompt": True,
                "prompt_cache_key": cache_key,
            })

        if num_ctx:
            extra_body["max_model_len"] = num_ctx

        if extra_options:
            payload.update({k: v for k, v in extra_options.items() if k not in {"extra_body"}})
            extra_body.update(extra_options.get("extra_body", {}))

        payload["extra_body"] = extra_body

        try:
            response = await self.client.post("chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {
                "response": content,
                "raw": data,
                "done": True,
            }
        except httpx.RequestError as exc:
            error_msg = f"vLLM request failed: {type(exc).__name__}: {exc}"
            logger.error(error_msg)
            return {"response": f"Error: {error_msg}", "done": True}
        except httpx.HTTPStatusError as exc:
            error_msg = (
                f"vLLM API error {exc.response.status_code}: {exc.response.text[:500]}"
            )
            logger.error(error_msg)
            return {"response": f"Error: {error_msg}", "done": True}
        except Exception as exc:  # pragma: no cover - defensive
            error_msg = f"Unexpected vLLM error: {type(exc).__name__}: {exc}"
            logger.exception(error_msg)
            return {"response": f"Error: {error_msg}", "done": True}

    async def generate_text(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        system_prompt: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload = await self.generate_response(
            prompt,
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            system_prompt=system_prompt,
            extra_options=extra_options,
        )
        return payload.get("response", "")

    async def generate_structured_response(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        *,
        model: Optional[str] = None,
        temperature: float = TEMPERATURE_FACTUAL,
        max_retries: int = 3,
        base_num_predict: int = 4096,
        max_num_predict: int = 8192,
        use_cot_prompt: bool = True,
        validation_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[BaseModel]:
        if use_cot_prompt:
            system_prompt = get_full_system_prompt(
                content_type="factual" if temperature <= 0.4 else "creative"
            )
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        num_predict = base_num_predict
        attempt = 0

        while attempt < max_retries:
            try:
                response_text, truncated = await self._generate_text_with_dynamic_predict(
                    prompt=full_prompt,
                    model=self._resolve_model(model),
                    temperature=temperature,
                    num_predict=num_predict,
                    max_num_predict=max_num_predict,
                )

                if truncated:
                    raise ValueError("LLM response truncated after dynamic retries")

                try:
                    parsed = json.loads(response_text)
                except json.JSONDecodeError:
                    try:
                        parsed = repair_json(response_text, return_objects=True)
                    except Exception as repair_error:
                        if "```json" in response_text:
                            start = response_text.find("```json") + 7
                            end = response_text.find("```", start)
                            if end != -1:
                                parsed = json.loads(response_text[start:end].strip())
                            else:
                                raise ValueError("Incomplete JSON in markdown") from repair_error
                        else:
                            raise ValueError("No valid JSON found in response") from repair_error

                parsed = self._coerce_to_model_schema(parsed, response_model)

                if not isinstance(parsed, dict):
                    raise ValueError(f"Expected dict after coercion, got {type(parsed).__name__}")

                if validation_context:
                    validated = response_model.model_validate(parsed, context=validation_context)
                else:
                    validated = response_model.model_validate(parsed)

                data_dict = validated.model_dump()
                list_values = [v for v in data_dict.values() if isinstance(v, list)]
                if list_values:
                    empty_count = sum(1 for v in list_values if not v)
                    if empty_count / len(list_values) > 0.8:
                        raise ValueError(
                            f"Too many empty fields: {empty_count}/{len(list_values)} lists empty"
                        )

                return validated
            except Exception as exc:
                attempt += 1
                logger.warning("Structured generation attempt %s failed: %s", attempt, exc)
                if attempt >= max_retries:
                    logger.error("All %s structured generation attempts failed", max_retries)
                    return None
                temperature = min(1.0, temperature + 0.1)

        return None

    async def _generate_text_with_dynamic_predict(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float,
        num_predict: int,
        max_num_predict: int,
    ) -> Tuple[str, bool]:
        current = num_predict
        while True:
            payload = await self.generate_response(
                prompt,
                model=model,
                temperature=temperature,
                num_predict=current,
                system_prompt="",
                extra_options={"extra_body": {"skip_special_tokens": True}},
            )
            text = payload.get("response", "")
            raw = payload.get("raw", {})
            finish_reason = raw.get("choices", [{}])[0].get("finish_reason")
            if finish_reason and finish_reason != "length":
                return text, False
            if current >= max_num_predict:
                return text, True
            previous = current
            current = min(max_num_predict, max(current + 512, int(current * 1.5)))
            logger.warning(
                "vLLM response truncated at %s tokens; retrying with %s",
                previous,
                current,
            )

    def _coerce_to_model_schema(self, parsed: Any, response_model: Type[BaseModel]) -> Any:
        expected_fields = set(response_model.model_fields.keys())
        if isinstance(parsed, dict):
            parsed_keys = set(parsed.keys())
            if expected_fields <= parsed_keys:
                return parsed
            for wrapper in ("data", "result", "output"):
                if wrapper in parsed and isinstance(parsed[wrapper], dict):
                    wrapper_value = parsed[wrapper]
                    if expected_fields <= set(wrapper_value.keys()):
                        logger.info("Unwrapped '%s' object to match schema", wrapper)
                        return wrapper_value
            for value in parsed.values():
                if isinstance(value, dict) and expected_fields <= set(value.keys()):
                    logger.info("Unwrapped nested object to match schema")
                    return value
        return parsed

    async def benchmark_batch_sizes(self, prompt: str) -> Dict[int, float]:
        """Benchmark configured batch sizes to validate continuous batching.

        Returns a mapping ``batch_size -> seconds`` representing how long the
        server took to respond to *all* prompts in that batch.
        """

        async def _run_batch(size: int) -> Tuple[int, float]:
            import time

            start = time.perf_counter()
            await asyncio.gather(
                *[
                    self.generate_text(
                        prompt,
                        model=self.default_model,
                        temperature=TEMPERATURE_CREATIVE,
                        extra_options={"extra_body": {"skip_special_tokens": True}},
                    )
                    for _ in range(size)
                ]
            )
            return size, time.perf_counter() - start

        results = await asyncio.gather(*[_run_batch(size) for size in self.continuous_batch_sizes])
        benchmark = {size: duration for size, duration in results}
        logger.info("Continuous batching benchmark: %s", benchmark)
        return benchmark

    async def check_connection(self) -> bool:
        try:
            response = await self.client.get("models")
            response.raise_for_status()
            return True
        except Exception as exc:  # pragma: no cover - network guard
            logger.error("vLLM connection failed: %s", exc)
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        try:
            response = await self.client.get("models")
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as exc:  # pragma: no cover - network guard
            logger.error("Failed to list vLLM models: %s", exc)
            return []

    async def close(self) -> None:
        await self.client.aclose()

    async def cleanup(self) -> None:
        await self.close()


# Backwards compatibility for existing imports.
OllamaClient = VLLMClient
