#!/usr/bin/env python3
"""
LLM Client Utility
Uses vLLM with Qwen3-14B
"""
import logging
import json
import os
import shlex
import subprocess
from typing import Dict, Optional, List, Any
import httpx
from json_repair import repair_json

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM client for vLLM server"""
    
    def __init__(self, base_url="http://localhost:8000/v1", model_name="/home/antons-gs/enlitens-ai/models/llama-3.1-8b-instruct"):
        """
        Initialize LLM client for vLLM
        
        Args:
            base_url: vLLM server URL
            model_name: Path to model
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.client = httpx.Client(timeout=1800.0)
        self.external_timeout = int(os.getenv("EXTERNAL_JSON_FORMATTER_TIMEOUT", "180"))
        self.external_formatters: List[Dict[str, str]] = []
        gemini_cmd = os.getenv("GEMINI_JSON_FORMATTER_CMD")
        if gemini_cmd:
            self.external_formatters.append({"name": "Gemini CLI", "cmd": gemini_cmd})
        codex_cmd = os.getenv("CODEX_JSON_FORMATTER_CMD")
        if codex_cmd:
            self.external_formatters.append({"name": "Codex CLI", "cmd": codex_cmd})
        logger.info(f"Initialized vLLM client: {base_url} with model: {model_name}")
    
    def _chat_request(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        timeout: int,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Internal helper to send chat completions request."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            raise RuntimeError(f"vLLM generation timed out after {timeout}s")
        except httpx.ConnectError:
            raise RuntimeError("vLLM server not reachable at http://localhost:8000 - run: bash scripts/start_vllm_llama_8b.sh")
        except Exception as exc:
            raise RuntimeError(f"vLLM error: {exc}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.2,
        timeout: int = 1200,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate completion from prompt using vLLM chat endpoint.
        """
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        result = self._chat_request(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )

        return (
            result["choices"][0]["message"]["content"]
            .strip()
        )
    
    def generate_json(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.2,
        timeout: int = 1200,
        max_attempts: int = 3,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Generate JSON output from prompt
        
        Args:
            prompt: Input prompt (should request JSON output)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            timeout: Timeout in seconds
            
        Returns:
            Parsed JSON dictionary
        """
        conversation: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You MUST output a single valid JSON object that matches the requested schema. "
                    "Do not include explanations, markdown, code fences, or natural language outside of the JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        if json_schema:
            response_format = {"type": "json_schema", "json_schema": json_schema}
        else:
            response_format = {"type": "json_object"}

        for attempt in range(1, max_attempts + 1):
            result = self._chat_request(
                messages=list(conversation),
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                response_format=response_format,
            )
            raw = result["choices"][0]["message"]["content"].strip()

            # Strip Qwen3 reasoning tags (<think>…</think>) and any residual wrappers
            cleaned = raw
            if "<think>" in cleaned:
                if "</think>" in cleaned:
                    cleaned = cleaned.split("</think>", 1)[-1]
                else:
                    cleaned = cleaned.split("<think>", 1)[-1]
            cleaned = cleaned.replace("<think>", "").replace("</think>", "").strip()

            # Remove code fences if present
            if "```json" in cleaned:
                cleaned = cleaned.split("```json", 1)[1]
            if "```" in cleaned:
                cleaned = cleaned.split("```", 1)[0]

            cleaned = cleaned.strip()

            # Fallback: grab substring between first { and last }
            if cleaned and not cleaned.lstrip().startswith("{"):
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1 and end > start:
                    cleaned = cleaned[start : end + 1]

            cleaned = cleaned.strip()

            try:
                if not cleaned:
                    raise json.JSONDecodeError("empty payload", cleaned, 0)
                return json.loads(cleaned)
            except json.JSONDecodeError as exc:
                log_fn = logger.error if attempt >= max_attempts else logger.info
                log_fn(f"Failed to parse JSON (attempt {attempt}/{max_attempts}): {exc}")
                if attempt >= max_attempts:
                    logger.error(f"Raw response (first 1000 chars): {raw[:1000]}...")
                    logger.error(f"Cleaned payload (first 1000 chars): {cleaned[:1000]}...")
                else:
                    logger.debug(f"Raw response (first 1000 chars): {raw[:1000]}...")
                    logger.debug(f"Cleaned payload (first 1000 chars): {cleaned[:1000]}...")

                # Attempt to repair malformed JSON before retrying
                if cleaned:
                    try:
                        repaired = repair_json(cleaned)
                        if repaired:
                            repaired_obj = json.loads(repaired)
                            logger.info(
                                "Repaired malformed JSON response on attempt %s: used json-repair",
                                attempt,
                            )
                            return repaired_obj
                    except Exception as repair_exc:
                        logger.debug(
                            "json-repair failed to fix payload: %s", repair_exc, exc_info=True
                        )

                if attempt >= max_attempts:
                    external_fixed = self._format_with_external_agents(
                        raw_payload=raw,
                        cleaned_payload=cleaned,
                        schema=json_schema,
                        error_message=str(exc),
                    )
                    if external_fixed is not None:
                        return external_fixed
                    raise ValueError(f"Invalid JSON response: {exc}") from exc

                # Append assistant reply and ask for correction
                conversation.append({"role": "assistant", "content": raw})
                conversation.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous reply was NOT valid JSON. "
                            "Respond again with a SINGLE JSON object that matches the required schema exactly. "
                            f"Error was: {exc}. No narration, no code fences—only valid JSON."
                        ),
                    }
                )

    def _format_with_external_agents(
        self,
        raw_payload: str,
        cleaned_payload: str,
        schema: Optional[Dict[str, Any]],
        error_message: str,
    ) -> Optional[Dict[str, Any]]:
        if not self.external_formatters:
            return None

        schema_text = json.dumps(schema, indent=2) if schema else "N/A (use JSON object)"
        base_prompt = (
            "You are a JSON formatter. The following model output failed to parse.\n"
            f"Parsing error: {error_message}\n"
            "Your job is to return a single JSON object that matches the provided schema."
            " Do not include any commentary or markdown—only the JSON.\n\n"
            f"Schema:\n{schema_text}\n\n"
            "Problematic output:\n"
        )

        for formatter in self.external_formatters:
            cmd = formatter["cmd"]
            prompt = base_prompt + (cleaned_payload or raw_payload)
            try:
                logger.info(
                    "Attempting JSON repair via %s command: %s",
                    formatter["name"],
                    cmd,
                )
                proc = subprocess.run(
                    shlex.split(cmd),
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.external_timeout,
                )
                if proc.returncode != 0:
                    logger.warning(
                        "%s formatter failed (exit %s): %s",
                        formatter["name"],
                        proc.returncode,
                        proc.stderr.strip(),
                    )
                    continue

                candidate = proc.stdout.strip()
                if not candidate:
                    logger.warning("%s formatter returned empty output", formatter["name"])
                    continue

                try:
                    result = json.loads(candidate)
                    logger.info(
                        "%s formatter produced valid JSON fallback (len=%s chars)",
                        formatter["name"],
                        len(candidate),
                    )
                    return result
                except json.JSONDecodeError as parse_exc:
                    logger.warning(
                        "%s formatter output not valid JSON: %s\nOutput preview: %s",
                        formatter["name"],
                        parse_exc,
                        candidate[:500],
                    )
            except subprocess.TimeoutExpired:
                logger.warning(
                    "%s formatter timed out after %ss",
                    formatter["name"],
                    self.external_timeout,
                )
            except FileNotFoundError:
                logger.warning(
                    "%s formatter command not found: %s",
                    formatter["name"],
                    cmd,
                )
            except Exception as formatter_exc:
                logger.warning(
                    "%s formatter raised unexpected error: %s",
                    formatter["name"],
                    formatter_exc,
                )

        return None

