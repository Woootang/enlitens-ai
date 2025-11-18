"""Unified Ollama/vLLM client with caching and schema-aware helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import httpx
from json_repair import repair_json
from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from .prompts import (
    TEMPERATURE_FACTUAL,
    TEMPERATURE_CREATIVE,
    get_full_system_prompt,
)
from src.utils.prompt_cache import PromptCache
from src.utils.kv_cache_compressor import KVCacheCompressor
from src.utils.settings import get_settings

logger = logging.getLogger(__name__)

VLLM_DEFAULT_URL = "http://localhost:8000/v1"
VLLM_DEFAULT_MODEL = "/home/antons-gs/enlitens-ai/models/llama-3.1-8b-instruct"
MONITORING_MODEL = "/home/antons-gs/enlitens-ai/models/llama-3.1-8b-instruct"


class VLLMClient:
    """Async client for Ollama/vLLM style servers with prompt caching support."""

    _shared_prompt_cache = PromptCache()

    def __init__(
        self,
        base_url: str = VLLM_DEFAULT_URL,
        default_model: str = VLLM_DEFAULT_MODEL,
        *,
        timeout_seconds: float = 900.0,
        enable_prefix_caching: bool = True,
        continuous_batch_sizes: Optional[Iterable[int]] = None,
        prompt_cache: Optional[PromptCache] = None,
        headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.enable_prefix_caching = enable_prefix_caching
        self.prompt_cache = prompt_cache or self._shared_prompt_cache
        self.continuous_batch_sizes = list(continuous_batch_sizes or (8, 16, 24))
        self.headers = dict(headers or {})
        self.max_retries = max_retries
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout_seconds,
            headers=self.headers or None,
            transport=transport,
        )
        self.kv_compressor: Optional[KVCacheCompressor] = None
        try:
            compressor = KVCacheCompressor.shared()
            if compressor.is_enabled():
                self.kv_compressor = compressor
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("KV compressor initialisation failed: %s", exc)

        self._explicit_server_cap = self._read_env_optional_int("VLLM_SERVER_MAX_TOKENS")
        self._explicit_completion_cap = self._read_env_optional_int("VLLM_COMPLETION_CAP_TOKENS")
        self._fallback_completion_cap = (
            self._explicit_completion_cap or self._explicit_server_cap or 60000
        )
        self._prompt_guard_tokens = self._read_env_int("VLLM_PROMPT_GUARD_TOKENS", 2048)
        self._min_completion_tokens = self._read_env_int("VLLM_MIN_COMPLETION_TOKENS", 512)
        self._chars_per_token = self._read_env_float("VLLM_CHARS_PER_TOKEN", 3.6)
        self._effective_cap_tokens: Optional[int] = None
        self._logged_cap_info = False
        self._default_completion_request = 24576

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
        clone.kv_compressor = self.kv_compressor
        return clone

    def _resolve_model(self, model: Optional[str]) -> str:
        return model or self.default_model

    @staticmethod
    def _should_retry_exception(exc: BaseException) -> bool:
        if isinstance(exc, httpx.RequestError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return status >= 500 or status in {408, 429}
        return False

    async def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
            retry=retry_if_exception(self._should_retry_exception),
            reraise=True,
        ):
            with attempt:
                response = await self.client.request(method, url, **kwargs)
                if response.status_code >= 400:
                    response.raise_for_status()
                return response
        raise RuntimeError("Retry loop exited unexpectedly")

    @staticmethod
    def _read_env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            value = int(raw)
            return value if value > 0 else default
        except ValueError:
            logger.warning("Invalid integer for %s=%s; using %s", name, raw, default)
            return default

    @staticmethod
    def _read_env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid float for %s=%s; using %s", name, raw, default)
            return default

    @staticmethod
    def _read_env_optional_int(name: str) -> Optional[int]:
        raw = os.environ.get(name)
        if raw is None:
            return None
        try:
            value = int(raw)
            return value if value > 0 else None
        except ValueError:
            logger.warning("Invalid integer for %s=%s; ignoring override", name, raw)
            return None

    def _content_char_count(self, content: Any) -> int:
        if content is None:
            return 0
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            total = 0
            for item in content:
                if isinstance(item, dict):
                    total += len(str(item.get("text", "")))
                else:
                    total += len(str(item))
            return total
        if isinstance(content, dict):
            return len("".join(str(part) for part in content.values()))
        return len(str(content))

    def _estimate_tokens_from_messages(self, messages: List[Dict[str, Any]], *, fallback: int = 2048) -> int:
        total_chars = 0
        for message in messages:
            total_chars += self._content_char_count(message.get("content"))
        approx = int(total_chars / self._chars_per_token) if total_chars else 0
        return max(fallback, approx)

    def _estimate_text_tokens(self, *chunks: Optional[str], fallback: int = 2048) -> int:
        total_chars = sum(len(chunk) for chunk in chunks if chunk)
        approx = int(total_chars / self._chars_per_token) if total_chars else 0
        return max(fallback, approx)

    async def _get_effective_cap_tokens(self) -> int:
        if self._effective_cap_tokens is not None:
            return self._effective_cap_tokens
        discovered: Optional[int] = None
        try:
            models = await self.list_models()
            for entry in models:
                candidate = entry.get("max_model_len")
                if isinstance(candidate, int) and candidate > 0:
                    discovered = candidate
                    break
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Context length discovery failed: %s", exc)
        candidates: List[int] = []
        if discovered:
            candidates.append(discovered)
        if self._explicit_server_cap:
            candidates.append(self._explicit_server_cap)
        if self._explicit_completion_cap:
            candidates.append(self._explicit_completion_cap)
        if not candidates:
            candidates.append(self._fallback_completion_cap)
        cap = max(self._min_completion_tokens, min(candidates))
        self._effective_cap_tokens = cap
        if not self._logged_cap_info:
            logger.info(
                "📏 Using completion cap=%s tokens (server=%s, overrides=%s/%s)",
                cap,
                discovered,
                self._explicit_server_cap,
                self._explicit_completion_cap,
            )
            self._logged_cap_info = True
        return cap

    def _apply_completion_cap(self, requested: int, prompt_tokens: int, cap_tokens: int) -> int:
        available = cap_tokens - prompt_tokens - self._prompt_guard_tokens
        available = max(self._min_completion_tokens, available)
        if requested > available:
            logger.warning(
                "✂️ Clamping completion budget from %s→%s tokens (prompt≈%s, cap=%s)",
                requested,
                available,
                prompt_tokens,
                cap_tokens,
            )
        return min(requested, available)

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
        response_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a completion from the serving endpoint."""

        model_name = self._resolve_model(model)

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        requested_tokens = num_predict or self._default_completion_request
        prompt_tokens_estimate = int(num_ctx) if num_ctx is not None else self._estimate_tokens_from_messages(messages)
        cap_tokens = await self._get_effective_cap_tokens()
        max_tokens = self._apply_completion_cap(requested_tokens, prompt_tokens_estimate, cap_tokens)

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if top_p is not None:
            payload["top_p"] = top_p
        if response_format:
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
        compression_meta: Optional[Dict[str, Any]] = None
        if self.kv_compressor:
            compression_meta = self.kv_compressor.before_request(prompt)

        try:
            response = await self._request_with_retry("POST", "/chat/completions", json=payload)
            data = response.json()
        except httpx.HTTPStatusError as exc:
            # Intermittent 404s can happen depending on how the server exposes the OpenAI routes.
            # If we hit a 404, retry by toggling the /v1 prefix heuristically.
            if exc.response is not None and exc.response.status_code == 404:
                base = self.base_url.rstrip("/")
                candidates: List[str] = []
                if base.endswith("/v1"):
                    base_no = base[:-3]
                    candidates.append(f"{base_no}/v1/chat/completions")
                    candidates.append(f"{base_no}/chat/completions")
                else:
                    candidates.append(f"{base}/v1/chat/completions")
                data = None
                for url in candidates:
                    try:
                        fb = await httpx.AsyncClient(
                            timeout=self.client.timeout,
                            headers=self.headers or None,
                        ).post(url, json=payload)
                        fb.raise_for_status()
                        data = fb.json()
                        break
                    except Exception:
                        continue
                if data is None:
                    raise
            else:
                raise
        finally:
            if self.kv_compressor and compression_meta is not None:
                try:
                    self.kv_compressor.after_request(compression_meta)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("KV compressor logging failed: %s", exc)
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")
        return {
            "response": message.get("content", ""),
            "raw": data,
            "done": finish_reason != "length",
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
        max_retries: int = 3,
        base_num_predict: int = 24576,  # 24k default output (up from 8k)
        max_num_predict: int = 32768,   # 32k max output (up from 16k)
        use_cot_prompt: bool = True,
        validation_context: Optional[Dict[str, Any]] = None,
        cache_prefix: Optional[str] = None,
        cache_chunk_id: Optional[str] = None,
        enforce_grammar: bool = False,
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

        prompt_tokens = self._estimate_text_tokens(system_prompt, prompt)
        cap_tokens = await self._get_effective_cap_tokens()
        base_num_predict = self._apply_completion_cap(base_num_predict, prompt_tokens, cap_tokens)
        max_num_predict = self._apply_completion_cap(max_num_predict, prompt_tokens, cap_tokens)
        if max_num_predict < base_num_predict:
            max_num_predict = base_num_predict

        cache_namespace = cache_prefix or f"{self._resolve_model(model)}:{response_model.__name__}"
        cache_chunk = cache_chunk_id or "global"

        if self.enable_prefix_caching:
            cached = self.prompt_cache.get(cache_namespace, cache_chunk, full_prompt)
            if cached is not None:
                try:
                    logger.info(
                        "🔁 Using cached structured response for prefix '%s' chunk '%s'",
                        cache_namespace,
                        cache_chunk,
                    )
                    return response_model.model_validate(cached)
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.warning("Cached payload failed validation: %s", exc)

        attempts = 0
        current_temperature = temperature
        grammar: Optional[str] = None
        http_errors = 0  # Track HTTP 400/500 errors

        if enforce_grammar:
            try:
                grammar = self._build_grammar_for_model(response_model)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Failed to build grammar for %s: %s", response_model.__name__, exc)
                grammar = None

        use_response_format = True
        while attempts < max_retries:
            try:
                text = ""
                text, truncated = await self._generate_text_with_dynamic_predict(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=self._resolve_model(model),
                    temperature=current_temperature,
                    num_predict=base_num_predict,
                    max_num_predict=max_num_predict,
                    top_p=0.9,
                    grammar=grammar,
                    use_response_format=use_response_format,
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
                    def _stringify_sequence(items: List[Any]) -> List[str]:
                        stringified: List[str] = []
                        for item in items:
                            if isinstance(item, str):
                                stringified.append(item)
                            elif isinstance(item, dict):
                                parts = []
                                for k, v in item.items():
                                    if v is None:
                                        continue
                                    parts.append(f"{k}: {v}")
                                stringified.append(" | ".join(parts) if parts else str(item))
                            elif isinstance(item, list):
                                parts = [str(elem) for elem in item if elem is not None]
                                stringified.append(" | ".join(parts) if parts else str(item))
                            else:
                                stringified.append(str(item))
                        return stringified

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
                        string_fields = [
                            "article_ideas",
                            "blog_outlines",
                            "talking_points",
                            "expert_quotes",
                            "case_studies",
                            "how_to_guides",
                            "myth_busting",
                        ]
                        for key in string_fields:
                            if key in parsed and isinstance(parsed[key], list):
                                parsed[key] = _stringify_sequence(parsed[key])
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
                    else:
                        if isinstance(parsed, dict):
                            for key, value in list(parsed.items()):
                                if isinstance(value, list) and any(not isinstance(it, str) for it in value):
                                    parsed[key] = _stringify_sequence(value)
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
                    filled = len(list_values) - empty
                    
                    # VERY lenient: Accept if ANY field has content
                    # (For complex schemas with 7-8 list fields, partial completion is acceptable)
                    if filled == 0:
                        raise ValueError(
                            f"No content generated: all {len(list_values)} lists are empty"
                        )
                    
                    # Log warning if less than 50% filled, but don't reject
                    if filled < len(list_values) // 2:
                        logger.info(f"⚠️ Partial completion: {filled}/{len(list_values)} lists filled (acceptable)")

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
                        available = max(
                            self._min_completion_tokens,
                            cap_tokens - message_tokens - self._prompt_guard_tokens,
                        )
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

                # If we get repeated HTTP errors, disable grammar and structured constraints
                if http_errors >= 2 and fallback_to_unstructured:
                    logger.info("🔄 Falling back to unstructured generation due to repeated HTTP errors")
                    grammar = None
                    enforce_grammar = False
                    use_response_format = False
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
                
                # Log what was actually generated for debugging (changed to INFO level)
                if text and len(text) > 100:
                    logger.info(f"📝 Model generated {len(text)} chars but failed validation. Sample: {text[:300]}...")
                elif text:
                    logger.info(f"📝 Model generated short response ({len(text)} chars): {text}")
                
                grammar = None  # Disable grammar after first failure
                use_response_format = False
                current_temperature = min(1.0, current_temperature + 0.1)
                if attempts >= max_retries:
                    # Try one last time to salvage partial content
                    logger.warning("All %s structured generations failed; attempting fallback extraction", max_retries)
                    try:
                        # Try to create a minimal valid object with whatever we have
                        fallback_obj = response_model()
                        logger.info("✓ Returning minimal %s object as fallback", response_model.__name__)
                        return fallback_obj
                    except Exception as fallback_exc:
                        logger.error("Fallback object creation failed: %s", fallback_exc)
                        return None

        return None

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
        use_response_format: bool,
    ) -> Tuple[str, bool]:
        prompt_tokens = self._estimate_text_tokens(system_prompt, prompt)
        cap_tokens = await self._get_effective_cap_tokens()
        base_clamped = self._apply_completion_cap(num_predict, prompt_tokens, cap_tokens)
        max_clamped = self._apply_completion_cap(max_num_predict, prompt_tokens, cap_tokens)
        if max_clamped < base_clamped:
            max_clamped = base_clamped
        current = base_clamped
        while True:
            payload = await self.generate_response(
                prompt,
                model=model,
                temperature=temperature,
                num_predict=current,
                system_prompt=system_prompt,
                top_p=top_p,
                grammar=grammar,
                # Encourage JSON object responses to reduce top-level list outputs
                response_format="json_object" if use_response_format else None,
            )
            text = payload.get("response", "")
            if payload.get("done", True):
                return text, False
            if current >= max_clamped:
                return text, True
            previous = current
            current = min(max_clamped, max(current + 512, int(current * 1.5)))
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
            await self._request_with_retry("GET", "/models")
            return True
        except Exception as exc:
            logger.error("LLM connection failed: %s", exc)
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        try:
            response = await self._request_with_retry("GET", "/models")
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
            await self._request_with_retry("GET", "/api/tags")
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
            resolved_url = resolved_url or (
                "http://localhost:11434/v1" if provider_name == "ollama" else VLLM_DEFAULT_URL
            )
            self._client = client_cls(
                base_url=resolved_url,
                default_model=resolved_model or VLLM_DEFAULT_MODEL,
                **client_kwargs,
            )

        self.provider = provider_name
        self.base_url = getattr(self._client, "base_url", resolved_url)
        self.default_model = getattr(self._client, "default_model", resolved_model)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._client, item)

    def clone_with_model(self, model: str) -> "OllamaClient":
        cloned = self._client.clone_with_model(model)
        wrapper = object.__new__(OllamaClient)
        wrapper._client = cloned
        wrapper.provider = self.provider
        wrapper.base_url = getattr(cloned, "base_url", self.base_url)
        wrapper.default_model = getattr(cloned, "default_model", model)
        wrapper._settings = self._settings
        return wrapper

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
