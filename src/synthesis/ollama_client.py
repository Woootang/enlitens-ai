"""Unified Ollama/vLLM client with caching and schema-aware helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import weakref
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
from urllib.parse import urlparse

import httpx
from json_repair import repair_json
from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from .prompts import (
    TEMPERATURE_FACTUAL,
    TEMPERATURE_CREATIVE,
    get_full_system_prompt,
)
from .normalize import normalize_research_content_payload
from src.utils.prompt_cache import PromptCache
from src.utils.settings import get_settings
from src.monitoring.error_telemetry import TelemetrySeverity, telemetry_recorder

logger = logging.getLogger(__name__)

TELEMETRY_AGENT = "ollama_client"


class LLMServiceError(RuntimeError):
    """Exception raised when the LLM backend is unavailable or misconfigured."""

    def __init__(
        self,
        message: str,
        *,
        endpoint: str,
        status: Optional[int] = None,
        payload_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.endpoint = endpoint
        self.status = status
        self.payload_summary = payload_summary or {}


class _FailoverNeeded(RuntimeError):
    """Internal control-flow exception indicating a failover should be retried."""


def _build_payload_summary(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    summary: Dict[str, Any] = {}
    for key in ("model", "temperature", "max_tokens", "top_p"):
        if key in payload:
            summary[key] = payload[key]

    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, dict):
            content = str(last_message.get("content", ""))
            summary["last_message_preview"] = content[:160]

    return summary

_FALLBACK_BASE_URL = "http://localhost:8000/v1"
_FALLBACK_MODEL = "/home/antons-gs/enlitens-ai/models/mistral-7b-instruct"


def _resolve_default_base_url() -> str:
    """Return the best-effort default base URL respecting configuration."""

    try:
        settings = get_settings()
        base_url = settings.llm.base_url
    except Exception:  # pragma: no cover - defensive during import time
        base_url = None
    return (base_url or _FALLBACK_BASE_URL).rstrip("/")


def _resolve_default_model() -> str:
    """Return the default model path, allowing configuration overrides."""

    try:
        settings = get_settings()
        model = settings.llm.default_model
    except Exception:  # pragma: no cover - defensive during import time
        model = None
    return model or _FALLBACK_MODEL


_DEFAULT_COMPLETION_TOKENS = 4096
_MAX_COMPLETION_TOKENS = 8192
_MAX_AUTO_CONTINUATIONS = 3


def _resolve_default_completion_tokens() -> int:
    """Determine the default ``max_tokens`` budget for chat completions."""

    try:
        settings = get_settings()
        extra = getattr(settings.llm, "extra", {}) or {}
        env_override = os.environ.get("LLM_MAX_COMPLETION_TOKENS")
        candidate = env_override or extra.get("max_completion_tokens")
        if candidate is not None:
            value = int(candidate)
            if value > 0:
                return max(512, min(value, _MAX_COMPLETION_TOKENS))
    except Exception:  # pragma: no cover - defensive guard
        pass
    return _DEFAULT_COMPLETION_TOKENS


def _resolve_max_completion_tokens(default: int) -> int:
    """Return the hard upper bound for continuation attempts."""

    try:
        settings = get_settings()
        extra = getattr(settings.llm, "extra", {}) or {}
        env_override = os.environ.get("LLM_MAX_AUTO_TOKENS")
        candidate = env_override or extra.get("max_completion_limit")
        if candidate is not None:
            value = int(candidate)
            if value > 0:
                return max(default, min(value, _MAX_COMPLETION_TOKENS))
    except Exception:  # pragma: no cover - defensive guard
        pass
    return max(default, _MAX_COMPLETION_TOKENS)


VLLM_DEFAULT_URL = _resolve_default_base_url()
VLLM_DEFAULT_MODEL = _resolve_default_model()
MONITORING_MODEL = _FALLBACK_MODEL


class VLLMClient:
    """Async client for Ollama/vLLM style servers with prompt caching support."""

    _shared_prompt_cache = PromptCache()

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        *,
        timeout_seconds: float = 900.0,
        enable_prefix_caching: bool = True,
        continuous_batch_sizes: Optional[Iterable[int]] = None,
        prompt_cache: Optional[PromptCache] = None,
        headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        secondary_base_url: Optional[str] = None,
    ) -> None:
        resolved_base = (base_url or _resolve_default_base_url()).rstrip("/")
        resolved_model = default_model or _resolve_default_model()
        secondary = (secondary_base_url or "").rstrip("/") or None
        self.base_url = resolved_base
        self.default_model = resolved_model
        self.enable_prefix_caching = enable_prefix_caching
        self.prompt_cache = prompt_cache or self._shared_prompt_cache
        self.continuous_batch_sizes = list(continuous_batch_sizes or (8, 16, 24))
        self.headers = dict(headers or {})
        self.max_retries = max_retries
        self._timeout_seconds = timeout_seconds
        self._transport = transport
        self._base_url_candidates: List[str] = []
        for candidate in (resolved_base, secondary):
            if candidate and candidate not in self._base_url_candidates:
                self._base_url_candidates.append(candidate)
        if not self._base_url_candidates:
            self._base_url_candidates.append(resolved_base)
        self._active_base_index = 0
        self.default_max_tokens = _resolve_default_completion_tokens()
        self.max_completion_tokens = _resolve_max_completion_tokens(self.default_max_tokens)
        self.client = httpx.AsyncClient(
            timeout=timeout_seconds,
            headers=self.headers or None,
            transport=transport,
        )
        self._resolved_chat_path: Optional[str] = None
        self._resolved_chat_endpoint_url: Optional[str] = None
        self._health_lock = asyncio.Lock()
        self._peers: "weakref.WeakSet[VLLMClient]" = weakref.WeakSet()
        self._peers.add(self)

    def clone_with_model(self, model: str) -> "VLLMClient":
        """Create a lightweight clone that reuses the underlying HTTP client."""

        clone = object.__new__(VLLMClient)
        clone.base_url = self.base_url
        clone.client = self.client
        clone.default_model = model
        clone.enable_prefix_caching = self.enable_prefix_caching
        clone.prompt_cache = self.prompt_cache
        clone.continuous_batch_sizes = list(self.continuous_batch_sizes)
        clone.headers = dict(self.headers)
        clone.max_retries = self.max_retries
        clone.default_max_tokens = self.default_max_tokens
        clone.max_completion_tokens = self.max_completion_tokens
        clone._timeout_seconds = self._timeout_seconds
        clone._transport = self._transport
        clone._base_url_candidates = list(self._base_url_candidates)
        clone._active_base_index = self._active_base_index
        clone._resolved_chat_path = self._resolved_chat_path
        clone._resolved_chat_endpoint_url = self._resolved_chat_endpoint_url
        clone._health_lock = self._health_lock
        clone._peers = self._peers
        self._peers.add(clone)
        return clone

    def _resolve_model(self, model: Optional[str]) -> str:
        return model or self.default_model

    @staticmethod
    def _should_retry_exception(exc: BaseException) -> bool:
        if isinstance(exc, _FailoverNeeded):
            return True
        if isinstance(exc, httpx.RequestError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return status >= 500 or status in {408, 429}
        return False

    def _propagate_state(self) -> None:
        for peer in list(self._peers):
            peer.base_url = self.base_url
            peer._active_base_index = self._active_base_index
            peer._resolved_chat_path = self._resolved_chat_path
            peer._resolved_chat_endpoint_url = self._resolved_chat_endpoint_url

    @staticmethod
    def _candidate_chat_paths(base_url: str) -> List[str]:
        parsed = urlparse(base_url)
        suffix = parsed.path.rstrip("/")
        candidates: List[str] = []
        if suffix.endswith("/v1"):
            candidates.append("chat/completions")
        else:
            candidates.append("v1/chat/completions")
        candidates.append("chat/completions")
        # Ensure uniqueness preserving order
        seen: set[str] = set()
        ordered: List[str] = []
        for path in candidates:
            key = path.lstrip("/")
            if key not in seen:
                seen.add(key)
                ordered.append(key)
        return ordered

    @staticmethod
    def _build_absolute_url(base_url: str, path: str) -> str:
        base = base_url.rstrip("/")
        segment = path.lstrip("/")
        return f"{base}/{segment}" if segment else base

    def _health_check_payload(self) -> Dict[str, Any]:
        return {
            "model": self.default_model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "stream": False,
        }

    async def _ensure_chat_endpoint(
        self,
        *,
        force: bool = False,
        start_index: Optional[int] = None,
    ) -> None:
        if self._resolved_chat_path and not force:
            return

        async with self._health_lock:
            if self._resolved_chat_path and not force:
                return

            last_exception: Optional[Exception] = None
            index_start = start_index if start_index is not None else (0 if force else self._active_base_index)

            for idx in range(index_start, len(self._base_url_candidates)):
                base = self._base_url_candidates[idx]
                for path in self._candidate_chat_paths(base):
                    url = self._build_absolute_url(base, path)
                    try:
                        response = await self.client.post(url, json=self._health_check_payload())
                        if response.status_code < 400:
                            self.base_url = base
                            self._active_base_index = idx
                            self._resolved_chat_path = path
                            self._resolved_chat_endpoint_url = str(response.request.url)
                            self._propagate_state()
                            return
                        if response.status_code in {400, 404}:
                            last_exception = httpx.HTTPStatusError(
                                "Health check failed",
                                request=response.request,
                                response=response,
                            )
                        else:
                            response.raise_for_status()
                    except Exception as exc:
                        last_exception = exc
                        continue

            endpoint = self._build_absolute_url(self._base_url_candidates[min(len(self._base_url_candidates) - 1, index_start)], "chat/completions")
            summary = _build_payload_summary(self._health_check_payload())
            raise LLMServiceError(
                "No responsive chat completions endpoint discovered",
                endpoint=endpoint,
                status=getattr(getattr(last_exception, "response", None), "status_code", None),
                payload_summary=summary,
            )

    async def _attempt_failover(self) -> bool:
        if self._active_base_index + 1 >= len(self._base_url_candidates):
            return False

        self._active_base_index += 1
        self.base_url = self._base_url_candidates[self._active_base_index]
        self._resolved_chat_path = None
        self._resolved_chat_endpoint_url = None
        self._propagate_state()

        try:
            await self._ensure_chat_endpoint(force=True, start_index=self._active_base_index)
            return True
        except LLMServiceError:
            return False

    def _record_client_error(
        self,
        *,
        response: httpx.Response,
        payload: Optional[Dict[str, Any]],
    ) -> None:
        try:
            summary = _build_payload_summary(payload)
            telemetry_recorder.record_event(
                doc_id=None,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.CRITICAL,
                impact="LLM chat completion returned client error",
                details={
                    "status": response.status_code,
                    "endpoint": str(response.request.url),
                    "payload": summary,
                },
            )
        except Exception:  # pragma: no cover - telemetry should not break requests
            logger.exception("Failed to record telemetry for client error")

    @property
    def resolved_chat_endpoint(self) -> Optional[str]:
        return self._resolved_chat_endpoint_url

    async def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        is_chat_completion = url == "chat/completions"
        if is_chat_completion:
            await self._ensure_chat_endpoint()

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
            retry=retry_if_exception(self._should_retry_exception),
            reraise=True,
        ):
            with attempt:
                request_url = url
                if is_chat_completion and self._resolved_chat_path:
                    request_url = self._resolved_chat_path
                if not str(request_url).startswith("http"):
                    request_url = self._build_absolute_url(self.base_url, str(request_url))

                request_url_str = str(request_url)
                response = await self.client.request(method, request_url_str, **kwargs)

                if response.status_code >= 400:
                    if is_chat_completion and response.status_code in {400, 404}:
                        payload = kwargs.get("json")
                        summary = _build_payload_summary(payload)

                        self._record_client_error(
                            response=response,
                            payload=payload,
                        )

                        # Clear the cached endpoint so reprobe starts from scratch.
                        self._resolved_chat_path = None
                        self._resolved_chat_endpoint_url = None
                        self._propagate_state()

                        reprobe_error: Optional[LLMServiceError] = None
                        try:
                            await self._ensure_chat_endpoint(force=True, start_index=0)
                        except LLMServiceError as exc:
                            reprobe_error = exc
                        else:
                            raise _FailoverNeeded("chat endpoint reprobe triggered")

                        if await self._attempt_failover():
                            raise _FailoverNeeded("chat endpoint failover triggered")

                        if reprobe_error is not None:
                            return self._structured_fallback_response(
                                method=method,
                                request_url=request_url_str,
                                payload_summary=summary,
                                status_code=response.status_code,
                                error=reprobe_error,
                            )

                        raise LLMServiceError(
                            f"LLM chat completion failed with status {response.status_code}",
                            endpoint=str(response.request.url),
                            status=response.status_code,
                            payload_summary=summary,
                        )

                    response.raise_for_status()

                return response
        raise RuntimeError("Retry loop exited unexpectedly")

    def _structured_fallback_response(
        self,
        *,
        method: str,
        request_url: str,
        payload_summary: Dict[str, Any],
        status_code: int,
        error: LLMServiceError,
    ) -> httpx.Response:
        request = httpx.Request(method, request_url)

        telemetry_payload = {
            "status": status_code,
            "endpoint": str(request.url),
            "message": "Returned empty structured fallback after chat client error",
            "error": str(error),
            "payload": payload_summary,
        }

        try:
            telemetry_recorder.record_event(
                doc_id=None,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.CRITICAL,
                impact="LLM chat completion fallback response generated",
                details=telemetry_payload,
            )
        except Exception:  # pragma: no cover - telemetry best effort
            logger.exception("Failed to record telemetry for fallback response")

        fallback_payload = {
            "id": "fallback-empty-response",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "fallback",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "telemetry": telemetry_payload,
        }

        return httpx.Response(200, json=fallback_payload, request=request)

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
        top_p: Optional[float] = 0.9,
        grammar: Optional[str] = None,
        response_format: Optional[object] = None,
    ) -> Dict[str, Any]:
        """Generate a completion from the serving endpoint."""

        model_name = self._resolve_model(model)

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if self._resolved_chat_path is None:
            await self._ensure_chat_endpoint()

        if num_predict is not None and num_predict > self.max_completion_tokens:
            logger.info(
                "Requested num_predict %s exceeds max completion budget %s; clamping",
                num_predict,
                self.max_completion_tokens,
            )

        total_budget = min(self.max_completion_tokens, num_predict or self.max_completion_tokens)
        chunk_budget = min(num_predict or self.default_max_tokens, total_budget)
        chunk_budget = max(256, chunk_budget)
        remaining_budget = total_budget

        base_messages = list(messages)
        conversation = list(base_messages)
        aggregated_chunks: List[str] = []
        attempts = 0
        truncated = False
        last_data: Optional[Dict[str, Any]] = None

        while remaining_budget > 0:
            payload: Dict[str, Any] = {
                "model": model_name,
                "messages": conversation,
                "temperature": temperature,
                "max_tokens": min(chunk_budget, remaining_budget),
                "stream": False,
            }

            if top_p is not None:
                payload["top_p"] = top_p
            if response_format:
                if isinstance(response_format, dict):
                    payload["response_format"] = response_format
                else:
                    payload["response_format"] = {"type": response_format}

            # Don't use extra_body - vLLM OpenAI API doesn't support it
            # This causes 400 errors: "Extra inputs are not permitted"
            # if num_ctx is not None:
            #     payload.setdefault("extra_body", {})["max_model_len"] = num_ctx
            # if grammar:
            #     payload.setdefault("extra_body", {})["grammar"] = grammar
            # if extra_options:
            #     payload.setdefault("extra_body", {}).update(extra_options)

            # Primary request
            response = await self._request_with_retry("POST", "chat/completions", json=payload)
            data = response.json()

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason")
            chunk_text = message.get("content", "")
            aggregated_chunks.append(chunk_text)
            last_data = data

            usage = data.get("usage") or {}
            completion_tokens = usage.get("completion_tokens")
            if completion_tokens is None:
                completion_tokens = payload["max_tokens"]
            remaining_budget = max(0, remaining_budget - int(completion_tokens))

            if finish_reason == "length" and num_predict is None and attempts < _MAX_AUTO_CONTINUATIONS and remaining_budget > 0:
                attempts += 1
                truncated = True
                next_chunk_budget = max(chunk_budget + 512, int(chunk_budget * 1.5))
                chunk_budget = min(max(256, next_chunk_budget), self.max_completion_tokens)
                chunk_budget = min(chunk_budget, remaining_budget)
                conversation = list(base_messages)
                conversation.append({"role": "assistant", "content": "".join(aggregated_chunks)})
                conversation.append(
                    {
                        "role": "user",
                        "content": "Please continue the previous response without repeating yourself. Continue exactly where you left off.",
                    }
                )
                logger.info(
                    "🔁 Continuing truncated response (attempt %s, remaining budget %s)",
                    attempts,
                    remaining_budget,
                )
                continue

            truncated = finish_reason == "length"
            break

        aggregated_text = "".join(aggregated_chunks)
        if last_data is None:
            last_data = {}
        else:
            try:
                last_choice = last_data.setdefault("choices", [{}])[0]
                last_message = last_choice.setdefault("message", {})
                last_message["content"] = aggregated_text
                if truncated:
                    last_choice["finish_reason"] = "length"
                else:
                    last_choice["finish_reason"] = last_choice.get("finish_reason", "stop")
            except Exception:  # pragma: no cover - defensive guard
                pass

        return {
            "response": aggregated_text,
            "raw": last_data,
            "done": not truncated,
        }

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
        top_p: Optional[float] = None,
        max_retries: int = 3,
        base_num_predict: int = 2048,
        max_num_predict: int = 8192,
        use_cot_prompt: bool = True,
        validation_context: Optional[Dict[str, Any]] = None,
        cache_prefix: Optional[str] = None,
        cache_chunk_id: Optional[str] = None,
        enforce_grammar: bool = True,
        fallback_to_unstructured: bool = True,
    ) -> Optional[BaseModel]:
        """Generate a structured response and validate it against a model.

        Args:
            fallback_to_unstructured: If True, falls back to plain completion without
                grammar enforcement when structured generation fails (useful for models
                that don't support JSON schema constraints).
        """

        system_prompt = (
            get_full_system_prompt("factual" if temperature <= 0.4 else "creative")
            if use_cot_prompt
            else None
        )
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        effective_top_p = 0.9 if top_p is None else top_p

        logger.debug(
            "🧪 Structured generation requested for %s (temp=%s, top_p=%s)",
            response_model.__name__,
            temperature,
            effective_top_p,
        )

        def _format_param(value: float) -> str:
            text = f"{value:.4f}"
            text = text.rstrip("0").rstrip(".")
            return text or "0"

        base_namespace = cache_prefix or f"{self._resolve_model(model)}:{response_model.__name__}"
        cache_namespace = (
            f"{base_namespace}|temp={_format_param(temperature)}|top_p={_format_param(effective_top_p)}"
        )
        cache_chunk = cache_chunk_id or "global"

        if self.enable_prefix_caching:
            cached = self.prompt_cache.get(cache_namespace, cache_chunk, full_prompt)
            if cached is not None:
                try:
                    logger.info(
                        "🔁 Using cached structured response for prefix '%s' chunk '%s' (temp=%s, top_p=%s)",
                        cache_namespace,
                        cache_chunk,
                        temperature,
                        effective_top_p,
                    )
                    return response_model.model_validate(cached)
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.warning("Cached payload failed validation: %s", exc)

        attempts = 0
        current_temperature = temperature
        grammar: Optional[str] = None
        use_json_schema: bool = True
        json_schema_rf: Optional[Dict[str, Any]] = None
        http_errors = 0  # Track HTTP 400/500 errors

        if enforce_grammar:
            # Prefer JSON Schema response_format when supported by the backend
            try:
                schema = response_model.model_json_schema()
                json_schema_rf = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": schema,
                    },
                }
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Failed to build JSON schema for %s: %s", response_model.__name__, exc)
                json_schema_rf = None
                use_json_schema = False
            # Keep grammar build available for future backends that accept grammars
            try:
                grammar = self._build_grammar_for_model(response_model)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Failed to build grammar for %s: %s", response_model.__name__, exc)
                grammar = None

        while attempts < max_retries:
            try:
                text, truncated = await self._generate_text_with_dynamic_predict(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=self._resolve_model(model),
                    temperature=current_temperature,
                    num_predict=base_num_predict,
                    max_num_predict=max_num_predict,
                    top_p=effective_top_p,
                    grammar=grammar,
                    response_format=(json_schema_rf if use_json_schema and json_schema_rf else "json_object"),
                )
                if truncated:
                    raise ValueError("LLM response truncated after dynamic retries")

                parsed = self._parse_structured_payload(text)
                parsed = self._coerce_to_model_schema(parsed, response_model)
                # Some models occasionally return a single-item list wrapping the object
                if not isinstance(parsed, dict) and isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    logger.info("Unwrapped single-item list to match schema")
                    parsed = parsed[0]
                if not isinstance(parsed, dict):
                    raise ValueError(f"Expected dict after coercion, got {type(parsed).__name__}")

                # Lightweight sanitation before validation to reduce avoidable failures
                try:
                    if response_model.__name__ == "BlogContent":
                        stats = parsed.get("statistics")
                        source = (validation_context or {}).get("source_text", "")
                        if isinstance(stats, list) and source:
                            import difflib
                            sentences = source.split('. ')
                            def _ok(item: Any) -> bool:
                                try:
                                    quote = str((item or {}).get("quote", "")).strip()
                                    if not quote:
                                        return False
                                    if quote in source:
                                        return True
                                    best = difflib.get_close_matches(quote, sentences, n=1, cutoff=0.8)
                                    return bool(best)
                                except Exception:
                                    return False
                            before = len(stats)
                            parsed["statistics"] = [it for it in stats if _ok(it)]
                            after = len(parsed["statistics"])
                            if before and after < before:
                                logger.info("Filtered %s invalid statistics prior to validation", before - after)
                    elif response_model.__name__ == "RebellionFramework":
                        # Flatten nested lists the model sometimes returns
                        def _flatten(items: Any) -> List[str]:
                            result: List[str] = []
                            if isinstance(items, list):
                                for it in items:
                                    if isinstance(it, list):
                                        result.append(" ".join([str(x) for x in it]))
                                    else:
                                        result.append(str(it))
                            return result
                        for key in ("narrative_deconstruction","sensory_profiling","executive_function","social_processing","strengths_synthesis","rebellion_themes","aha_moments"):
                            if key in parsed and isinstance(parsed[key], list) and any(isinstance(x, list) for x in parsed[key]):
                                parsed[key] = _flatten(parsed[key])
                    elif response_model.__name__ == "ResearchContent":
                        parsed = normalize_research_content_payload(parsed)
                except Exception:
                    # Best-effort sanitation; fall through to validation
                    pass

                if validation_context:
                    validated = response_model.model_validate(parsed, context=validation_context)
                else:
                    validated = response_model.model_validate(parsed)

                data_dict = validated.model_dump()
                list_values = [value for value in data_dict.values() if isinstance(value, list)]
                if list_values:
                    empty = sum(1 for value in list_values if not value)
                    if empty / len(list_values) > 0.8:
                        raise ValueError(
                            f"Too many empty fields: {empty}/{len(list_values)} lists are empty"
                        )

                if self.enable_prefix_caching:
                    self.prompt_cache.set(cache_namespace, cache_chunk, full_prompt, data_dict)

                return validated

            except httpx.HTTPStatusError as exc:
                attempts += 1
                http_errors += 1
                status_code = exc.response.status_code

                logger.warning(
                    "HTTP %s error on attempt %s: %s",
                    status_code,
                    attempts,
                    exc.response.text[:200] if hasattr(exc.response, 'text') else str(exc)
                )

                # Dynamically shrink completion budget if we exceeded context window
                try:
                    error_payload = exc.response.json()
                    error_message = error_payload.get("message", "")
                except Exception:  # pragma: no cover - defensive
                    error_message = exc.response.text if hasattr(exc.response, 'text') else ""

                if error_message and "maximum context length" in error_message:
                    match = re.search(r"\((\d+) in the messages, (\d+) in the completion\)", error_message)
                    if match:
                        message_tokens = int(match.group(1))
                        completion_tokens = int(match.group(2))
                        available = max(512, 8192 - message_tokens - 128)
                        if available < base_num_predict:
                            logger.info(
                                "🔧 Reducing completion budget from %s to %s tokens (messages=%s)",
                                base_num_predict,
                                available,
                                message_tokens,
                            )
                            base_num_predict = available
                        if available < max_num_predict:
                            max_num_predict = available

                # If we get repeated HTTP errors, disable schema/grammar constraints
                if http_errors >= 2 and fallback_to_unstructured:
                    logger.info("🔄 Falling back to unstructured generation due to repeated HTTP errors")
                    grammar = None
                    enforce_grammar = False
                    use_json_schema = False
                    # Simplify the prompt for plain completion
                    base_num_predict = min(base_num_predict, 2048)

                if attempts >= max_retries:
                    logger.error("All %s structured generations failed with HTTP errors", max_retries)
                    return None

                # Add backoff delay for server errors
                await asyncio.sleep(2 ** attempts)

            except Exception as exc:
                attempts += 1
                logger.warning("Structured generation attempt %s failed: %s", attempts, exc)
                grammar = None  # Disable grammar after first failure
                current_temperature = min(1.0, current_temperature + 0.1)
                # If we are still failing due to truncation, fall back to incremental field assembly
                if (
                    (isinstance(exc, ValueError) and "truncated" in str(exc).lower())
                    or (attempts >= max_retries and "truncated" in str(exc).lower())
                    or (attempts >= max_retries - 1)
                ):
                    try:
                        logger.info("🧩 Falling back to incremental field assembly for %s", response_model.__name__)
                        partial = await self._generate_fields_incrementally(
                            base_prompt=prompt,
                            system_prompt=system_prompt,
                            model=self._resolve_model(model),
                            response_model=response_model,
                            temperature=current_temperature,
                            top_p=effective_top_p,
                            validation_context=validation_context,
                        )
                        if partial is not None:
                            if self.enable_prefix_caching:
                                self.prompt_cache.set(cache_namespace, cache_chunk, full_prompt, partial)
                            if validation_context:
                                return response_model.model_validate(partial, context=validation_context)
                            return response_model.model_validate(partial)
                    except Exception as inc_exc:  # pragma: no cover - best effort fallback
                        logger.warning("Incremental assembly failed: %s", inc_exc)
                if attempts >= max_retries:
                    # Do not fail the entire pipeline; return an empty validated object
                    logger.warning("All %s structured generations failed; returning empty %s", max_retries, response_model.__name__)
                    try:
                        return response_model()
                    except Exception:
                        return None

        return None

    async def _generate_fields_incrementally(
        self,
        *,
        base_prompt: str,
        system_prompt: Optional[str],
        model: str,
        response_model: Type[BaseModel],
        temperature: float,
        top_p: float,
        validation_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build large JSON outputs by requesting one field at a time.

        This avoids long single-turn completions that hit context/length limits.
        """

        assembled: Dict[str, Any] = {}
        expected_fields = list(response_model.model_fields.keys())

        for field_name in expected_fields:
            field_prompt = (
                "You are generating a JSON object one field at a time to avoid truncation."\
                " Return ONLY a valid JSON object containing the single key '" + field_name + "'"\
                " with its best-value for this task. Do not include any other keys.\n\n"\
                + "Task: " + base_prompt + "\n\n"\
                + ("Existing fields already collected: " + json.dumps(assembled) + "\n\n" if assembled else "")\
                + "Respond strictly as a JSON object."
            )

            try:
                payload = await self.generate_response(
                    field_prompt,
                    model=model,
                    temperature=temperature,
                    num_predict=1024,
                    system_prompt=system_prompt,
                    top_p=top_p,
                    grammar=None,
                    response_format="json_object",
                )
                text = payload.get("response", "")
                part = self._parse_structured_payload(text)
                if not isinstance(part, dict):
                    raise ValueError("Incremental field response was not a JSON object")
                # Coerce and unwrap if needed
                part = self._coerce_to_model_schema(part, response_model)
                if field_name in part:
                    assembled[field_name] = part[field_name]
                else:
                    # Some models wrap under 'data'/'result'
                    for wrapper_key in ("data", "result", "output"):
                        wrapper = part.get(wrapper_key)
                        if isinstance(wrapper, dict) and field_name in wrapper:
                            assembled[field_name] = wrapper[field_name]
                            break
                    if field_name not in assembled:
                        # As a last resort, accept the entire dict if it's a single-key object
                        if len(part.keys()) == 1:
                            assembled[field_name] = next(iter(part.values()))
                        else:
                            logger.warning("Field '%s' missing in incremental step; inserting null", field_name)
                            assembled[field_name] = None
            except Exception as exc:
                logger.warning("Incremental generation for field '%s' failed: %s", field_name, exc)
                assembled[field_name] = None

        # Final validation
        try:
            if validation_context:
                validated = response_model.model_validate(assembled, context=validation_context)
            else:
                validated = response_model.model_validate(assembled)
            return validated.model_dump()
        except Exception as exc:  # pragma: no cover
            logger.warning("Final validation after incremental assembly failed: %s", exc)
            return assembled

    async def _generate_text_with_dynamic_predict(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str],
        model: str,
        temperature: float,
        num_predict: int,
        max_num_predict: int,
        top_p: float,
        grammar: Optional[str],
        response_format: Optional[object],
    ) -> Tuple[str, bool]:
        current = num_predict
        while True:
            payload = await self.generate_response(
                prompt,
                model=model,
                temperature=temperature,
                num_predict=current,
                system_prompt=system_prompt,
                top_p=top_p,
                grammar=grammar,
                # Encourage JSON outputs; prefer stricter schema when available
                response_format=response_format or "json_object",
            )
            text = payload.get("response", "")
            if payload.get("done", True):
                return text, False
            if current >= max_num_predict:
                return text, True
            previous = current
            current = min(max_num_predict, max(current + 512, int(current * 1.5)))
            logger.warning(
                "LLM response truncated at %s tokens; retrying with %s",
                previous,
                current,
            )

    def _parse_structured_payload(self, response_text: str) -> Any:
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            try:
                return repair_json(response_text, return_objects=True)
            except Exception:
                if "```json" in response_text:
                    start = response_text.find("```json") + 7
                    end = response_text.find("```", start)
                    if end != -1:
                        return json.loads(response_text[start:end].strip())
                raise

    def _coerce_to_model_schema(self, parsed: Any, response_model: Type[BaseModel]) -> Any:
        expected_fields = set(response_model.model_fields.keys())
        if isinstance(parsed, dict):
            parsed_keys = set(parsed.keys())
            if expected_fields <= parsed_keys:
                return parsed
            for wrapper_key in ("data", "result", "output"):
                wrapper = parsed.get(wrapper_key)
                if isinstance(wrapper, dict) and expected_fields <= set(wrapper.keys()):
                    logger.info("Unwrapped '%s' object to match schema", wrapper_key)
                    return wrapper
            for value in parsed.values():
                if isinstance(value, dict) and expected_fields <= set(value.keys()):
                    logger.info("Unwrapped nested object to match schema")
                    return value
        return parsed

    def _build_grammar_for_model(self, response_model: Type[BaseModel]) -> str:
        builder = _GBNFBuilder()
        return builder.build(response_model.model_json_schema())

    async def benchmark_batch_sizes(self, prompt: str) -> Dict[int, float]:
        """Run simple continuous batching benchmarks for observability."""

        async def _run_batch(size: int) -> Tuple[int, float]:
            import time

            start = time.perf_counter()
            await asyncio.gather(
                *[
                    self.generate_text(
                        prompt,
                        model=self.default_model,
                        temperature=TEMPERATURE_CREATIVE,
                        extra_options={"skip_special_tokens": True},
                    )
                    for _ in range(size)
                ]
            )
            return size, time.perf_counter() - start

        results = await asyncio.gather(*[_run_batch(size) for size in self.continuous_batch_sizes])
        benchmark = {size: duration for size, duration in results}
        logger.info("Continuous batching benchmark results: %s", benchmark)
        return benchmark

    async def check_connection(self) -> bool:
        """Verify the backing server is reachable."""

        try:
            await self._ensure_chat_endpoint()
            await self._request_with_retry("GET", "models")
            return True
        except LLMServiceError as exc:
            logger.error("LLM health check failed: %s", exc)
            return False
        except Exception as exc:
            logger.error("LLM connection failed: %s", exc)
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        try:
            response = await self._request_with_retry("GET", "models")
            data = response.json()
            return data.get("data", data.get("models", []))
        except Exception as exc:
            logger.error("Failed to list models: %s", exc)
            return []

    async def close(self) -> None:
        await self.client.aclose()

    async def cleanup(self) -> None:
        await self.close()


class OllamaAPIClient(VLLMClient):
    """Specialised client for Ollama's OpenAI-compatible bridge."""

    async def check_connection(self) -> bool:
        if await super().check_connection():
            return True
        try:
            await self._request_with_retry("GET", "api/tags")
            return True
        except Exception as exc:
            logger.error("Ollama health check failed: %s", exc)
            return False


class OpenAIClient(VLLMClient):
    """Client targeting OpenAI's native API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is required when using the OpenAI provider")

        headers = dict(kwargs.pop("headers", {}))
        headers.setdefault("Authorization", f"Bearer {api_key}")
        headers.setdefault("Content-Type", "application/json")

        resolved_base = (base_url or "https://api.openai.com/v1").rstrip("/")
        super().__init__(
            base_url=resolved_base,
            default_model=default_model,
            headers=headers,
            **kwargs,
        )


class OllamaClient:
    """Provider-aware wrapper that returns the appropriate backend client."""

    _PROVIDERS = {
        "vllm": VLLMClient,
        "ollama": OllamaAPIClient,
        "openai": OpenAIClient,
    }

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        *,
        provider: Optional[str] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        **kwargs: Any,
    ) -> None:
        settings = get_settings()
        self._settings = settings
        provider_name = (provider or settings.llm.provider or "vllm").lower()
        resolved_model = default_model or settings.llm.default_model
        resolved_url = base_url or settings.llm.endpoint_for(provider_name) or settings.llm.base_url
        secondary_url = settings.llm.secondary_base_url

        client_kwargs: Dict[str, Any] = dict(kwargs)
        if transport is not None:
            client_kwargs["transport"] = transport

        client_cls = self._PROVIDERS.get(provider_name, VLLMClient)

        if client_cls is OpenAIClient:
            api_key = client_kwargs.pop("api_key", None) or settings.llm.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required when provider=openai")
            resolved_base = (resolved_url or "https://api.openai.com/v1").rstrip("/")
            self._client = client_cls(
                api_key=api_key,
                base_url=resolved_base,
                default_model=resolved_model or "gpt-4o-mini",
                **client_kwargs,
            )
            resolved_url = resolved_base
        else:
            default_url = "http://localhost:11434/v1" if provider_name == "ollama" else _resolve_default_base_url()
            resolved_url = (resolved_url or default_url).rstrip("/")
            self._client = client_cls(
                base_url=resolved_url,
                default_model=resolved_model or _resolve_default_model(),
                secondary_base_url=secondary_url,
                **client_kwargs,
            )

        self.provider = provider_name
        self.default_model = getattr(self._client, "default_model", resolved_model)
        self._secondary_base_url = secondary_url

    def __getattr__(self, item: str) -> Any:
        return getattr(self._client, item)

    def clone_with_model(self, model: str) -> "OllamaClient":
        cloned = self._client.clone_with_model(model)
        wrapper = object.__new__(OllamaClient)
        wrapper._client = cloned
        wrapper.provider = self.provider
        wrapper.default_model = getattr(cloned, "default_model", model)
        wrapper._settings = self._settings
        wrapper._secondary_base_url = getattr(self, "_secondary_base_url", None)
        return wrapper

    @property
    def base_url(self) -> Optional[str]:
        return getattr(self._client, "base_url", None)

    @property
    def resolved_chat_endpoint(self) -> Optional[str]:
        return getattr(self._client, "resolved_chat_endpoint", None)

    async def close(self) -> None:
        if hasattr(self._client, "close"):
            await self._client.close()

    async def cleanup(self) -> None:
        if hasattr(self._client, "cleanup"):
            await self._client.cleanup()


class _GBNFBuilder:
    """Utility to convert a JSON schema into a restrictive GBNF grammar."""

    def __init__(self) -> None:
        self.rules: Dict[str, str] = {}
        self.order: List[str] = []

    def build(self, schema: Dict[str, Any]) -> str:
        root_rule = "rule_root"
        self._emit_rule(root_rule, schema)

        base_rules = [
            "root ::= ws " + root_rule + " ws",
            'ws ::= (" " | "\\t" | "\\n" | "\\r")*',
            'string ::= "\"" characters "\""',
            'characters ::= character*',
            'character ::= [^"\\] | "\\\\" ("\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t")',
            'number ::= "-"? int frac? exp?',
            'int ::= "0" | digit19 digit*',
            'frac ::= "." digit+',
            'exp ::= ("e" | "E") ("+" | "-")? digit+',
            'digit ::= "0" | digit19',
            'digit19 ::= "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"',
            'boolean ::= "true" | "false"',
            'null ::= "null"',
        ]

        rule_lines = [self.rules[name] for name in self.order]
        return "\n".join(base_rules + rule_lines)

    def _emit_rule(self, name: str, schema: Dict[str, Any]) -> None:
        if name in self.rules:
            return

        schema = schema or {}
        schema_type = schema.get("type")

        if schema_type == "object":
            properties = schema.get("properties", {}) or {}
            if not properties:
                rule = f'{name} ::= "{{" ws "}}"'
            else:
                parts = []
                for key, subschema in properties.items():
                    child_name = f"{name}_{key}"
                    self._emit_rule(child_name, subschema)
                    parts.append(
                        f'"\"{key}\""' + ' ws ":" ws ' + child_name
                    )
                joined = ' ws "," ws '.join(parts)
                rule = f'{name} ::= "{{" ws {joined} ws "}}"'
        elif schema_type == "array":
            items = schema.get("items", {}) or {}
            child_name = f"{name}_item"
            self._emit_rule(child_name, items)
            rule = f'{name} ::= "[" ws ({child_name} (ws "," ws {child_name})*)? ws "]"'
        elif schema_type in {"number", "integer"}:
            rule = f"{name} ::= number"
        elif schema_type == "boolean":
            rule = f"{name} ::= boolean"
        elif schema.get("enum"):
            options = " | ".join(f'"{value}"' for value in schema["enum"])
            rule = f"{name} ::= {options}"
        elif schema.get("anyOf"):
            option_names = []
            for idx, option in enumerate(schema["anyOf"]):
                option_name = f"{name}_alt_{idx}"
                self._emit_rule(option_name, option)
                option_names.append(option_name)
            rule = f"{name} ::= {' | '.join(option_names)}"
        else:
            rule = f"{name} ::= string"

        self.rules[name] = rule
        self.order.append(name)
