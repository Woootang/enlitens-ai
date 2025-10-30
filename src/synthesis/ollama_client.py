"""
Enhanced Ollama Client with JSON Repair and Schema Enforcement

This module provides an advanced client for interacting with Ollama models,
including JSON repair, schema enforcement, and structured output generation.

HALLUCINATION PREVENTION:
- Chain-of-Thought prompting (53% reduction per Frontiers AI 2025)
- Temperature lowered to 0.3 for factual content (research optimal)
- Validation context support for citation checking
"""

import httpx
import json
import logging
from typing import List, Dict, Any, Optional, Type, Tuple
from json_repair import repair_json
from pydantic import BaseModel
from .prompts import (
    CHAIN_OF_THOUGHT_SYSTEM_PROMPT,
    TEMPERATURE_FACTUAL,
    TEMPERATURE_CREATIVE,
    get_full_system_prompt,
    get_generation_params
)
from src.utils.prompt_cache import PromptCache

logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Enhanced Ollama client with JSON repair and schema enforcement.
    """
    
    _shared_prompt_cache = PromptCache()

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "qwen3:32b",
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.base_url = base_url
        # Increase timeout to 15 minutes (900s) for large documents and complex generations
        if client is None:
            self.client = httpx.AsyncClient(base_url=self.base_url, timeout=900.0)
            logger.info(
                "OllamaClient initialized with base_url: %s and default_model: %s",
                self.base_url,
                default_model,
            )
        else:
            self.client = client
            logger.debug(
                "OllamaClient cloned with shared HTTP client for model %s", default_model
            )

        self.default_model = default_model
        self.prompt_cache = OllamaClient._shared_prompt_cache

    def clone_with_model(self, model: str) -> "OllamaClient":
        """Create a lightweight clone that reuses the underlying HTTP client."""

        return OllamaClient(
            base_url=self.base_url,
            default_model=model,
            client=self.client,
        )

    def _resolve_model(self, model: Optional[str]) -> str:
        return model or getattr(self, "default_model", "qwen3:32b")

    async def generate_response(
        self,
        prompt: str,
        model: str = "qwen3:32b",
        temperature: float = 0.7,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate response from Ollama with optional schema enforcement.
        
        Args:
            prompt: The input prompt
            model: The model to use
            temperature: Sampling temperature (0.0 to 1.0)
            num_ctx: Optional override for context window
            num_predict: Optional override for maximum generated tokens
            extra_options: Additional Ollama generation options
        """
        try:
            model_name = self._resolve_model(model)
            logger.info(f"Generating response with model: {model_name}")
            
            # Prepare request payload
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_gpu": -1,
                    "temperature": temperature,
                    "num_ctx": num_ctx or 4096,
                    "num_predict": num_predict or 2048,
                },
            }

            if extra_options:
                payload["options"].update(extra_options)
            
            response = await self.client.post("/api/generate", json=payload)
            response.raise_for_status()
            
            response_data = response.json()
            full_response = response_data.get("response", "")
            logger.info(f"Generated response length: {len(full_response)} characters")
            
            # Check if generation was truncated
            if not response_data.get("done", True):
                logger.warning("Response may be truncated - generation not complete")
            
            return response_data

        except httpx.RequestError as e:
            error_msg = f"Ollama request failed - Type: {type(e).__name__}, Details: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "response": f"Error: {error_msg}",
                "done": True,
            }
        except httpx.HTTPStatusError as e:
            error_msg = f"Ollama API error - Status: {e.response.status_code}, Response: {e.response.text}"
            logger.error(error_msg)
            return {
                "response": f"Error: {error_msg}",
                "done": True,
            }
        except Exception as e:
            error_msg = f"Unexpected error during Ollama generation - Type: {type(e).__name__}, Details: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "response": f"Error: {error_msg}",
                "done": True,
            }

    async def generate_structured_response(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        model: str = "qwen3:32b",
        temperature: float = 0.3,  # LOWERED from 0.7: Research shows 0.3 optimal for factual
        max_retries: int = 3,
        base_num_predict: int = 4096,  # Increased default for better generation
        max_num_predict: int = 8192,  # Increased max for complex extractions
        use_cot_prompt: bool = True,  # NEW: Enable Chain-of-Thought prompting
        validation_context: Optional[Dict[str, Any]] = None,  # NEW: For citation checking
        cache_prefix: Optional[str] = None,
        cache_chunk_id: Optional[str] = None,
    ) -> Optional[BaseModel]:
        """
        Generate structured response with automatic JSON repair and validation.

        HALLUCINATION PREVENTION (Research-backed):
        - Chain-of-Thought prompting: 53% reduction (Frontiers AI 2025)
        - Temperature 0.3: Optimal for factual content
        - Validation context: Enables citation verification

        Args:
            prompt: The input prompt
            model: The model to use
            response_model: Pydantic model for validation
            temperature: Sampling temperature (default 0.3 for factual)
            max_retries: Maximum number of retry attempts
            use_cot_prompt: Whether to prepend Chain-of-Thought system prompt
            validation_context: Context for Pydantic validators (e.g., source_text for citations)

        Returns:
            Validated Pydantic model instance or None if all retries failed
        """
        # Prepend Chain-of-Thought system prompt if enabled
        if use_cot_prompt:
            system_prompt = get_full_system_prompt(
                content_type="factual" if temperature <= 0.4 else "creative"
            )
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        cache_namespace = cache_prefix or f"{model}:{response_model.__name__}"
        cache_chunk = cache_chunk_id or "global"

        cached_payload = self.prompt_cache.get(cache_namespace, cache_chunk, full_prompt)
        if cached_payload is not None:
            try:
                logger.info(
                    "🔁 Using cached structured response for prefix '%s' chunk '%s'",
                    cache_namespace,
                    cache_chunk,
                )
                return response_model.model_validate(cached_payload)
            except Exception:
                logger.warning("Cached payload failed validation, regenerating")

        num_predict = base_num_predict
        attempt = 0

        while attempt < max_retries:
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} for structured generation")
                response_text, was_truncated = await self._generate_text_with_dynamic_predict(
                    prompt=full_prompt,  # Use full prompt with CoT system prompt
                    model=model,
                    temperature=temperature,
                    num_predict=num_predict,
                    max_num_predict=max_num_predict,
                )

                if was_truncated:
                    raise ValueError("LLM response truncated after dynamic retries")
                
                # Multi-stage JSON repair
                try:
                    # Stage 1: Direct JSON parsing
                    parsed_data = json.loads(response_text)
                    logger.info("Direct JSON parsing successful")
                except json.JSONDecodeError:
                    logger.info("Direct parsing failed, attempting JSON repair")
                    try:
                        # Stage 2: JSON repair
                        repaired_json = repair_json(response_text, return_objects=True)
                        parsed_data = repaired_json
                        logger.info("JSON repair successful")
                    except Exception as repair_error:
                        logger.warning(f"JSON repair failed: {repair_error}")
                        # Stage 3: Extract JSON from markdown code blocks
                        if "```json" in response_text:
                            start_idx = response_text.find("```json") + 7
                            end_idx = response_text.find("```", start_idx)
                            if end_idx != -1:
                                json_text = response_text[start_idx:end_idx].strip()
                                try:
                                    parsed_data = json.loads(json_text)
                                    logger.info("Extracted JSON from markdown successful")
                                except json.JSONDecodeError:
                                    raise ValueError("All JSON parsing methods failed")
                            else:
                                raise ValueError("Incomplete JSON in markdown")
                        else:
                            raise ValueError("No valid JSON found in response")

                parsed_data = self._coerce_to_model_schema(parsed_data, response_model)

                # Ensure we have a dictionary for Pydantic validation
                if not isinstance(parsed_data, dict):
                    value_repr = repr(parsed_data)[:500]  # First 500 chars of repr
                    logger.error(f"Coerced data is not a dictionary:")
                    logger.error(f"  Type: {type(parsed_data).__name__}")
                    logger.error(f"  Value: {value_repr}")
                    logger.error(f"  Length: {len(str(parsed_data))}")
                    raise ValueError(f"Expected dict after coercion, got {type(parsed_data).__name__}")

                # Validate with Pydantic model (with optional validation_context for citation checking)
                if validation_context:
                    validated_model = response_model.model_validate(parsed_data, context=validation_context)
                else:
                    validated_model = response_model.model_validate(parsed_data)
                data_dict = validated_model.model_dump()

                # Check if ALL lists are empty (more lenient check)
                list_values = [v for v in data_dict.values() if isinstance(v, list)]
                if list_values:
                    non_empty_count = sum(1 for v in list_values if v)
                    empty_count = len(list_values) - non_empty_count

                    # Only fail if more than 80% of lists are empty
                    if empty_count / len(list_values) > 0.8:
                        logger.warning(f"{empty_count}/{len(list_values)} list fields are empty - may indicate poor LLM output")
                        raise ValueError(f"Too many empty fields: {empty_count}/{len(list_values)} lists are empty")
                    elif empty_count > 0:
                        logger.info(f"Partial extraction: {non_empty_count}/{len(list_values)} fields populated ({empty_count} empty)")

                logger.info(f"Successfully validated response with {len(parsed_data)} keys")

                # Persist deterministic cache payload for identical prompts
                self.prompt_cache.set(cache_namespace, cache_chunk, full_prompt, data_dict)
                return validated_model
                
            except Exception as e:
                attempt += 1
                # Log detailed error information for debugging
                logger.warning(f"Attempt {attempt} failed: {e}")

                # Log response sample for debugging (first 1000 chars for better context)
                if 'response_text' in locals():
                    logger.warning(f"LLM response sample (first 1000 chars): {response_text[:1000]}...")
                    logger.warning(f"Full response length: {len(response_text)} characters")

                # Log parsed data if available
                if 'parsed_data' in locals():
                    logger.debug(f"Parsed data type: {type(parsed_data)}, keys: {parsed_data.keys() if isinstance(parsed_data, dict) else 'N/A'}")

                if attempt >= max_retries:
                    logger.error(f"All {max_retries} attempts failed for structured generation")
                    logger.error(f"Final error: {e}")
                    # Log full response text on final failure for debugging
                    if 'response_text' in locals():
                        logger.error(f"Full LLM response on final attempt:\n{response_text}")
                    return None

                # Increase temperature slightly to encourage variation
                temperature = min(1.0, temperature + 0.1)
                logger.info(f"Retrying with temperature {temperature}")

    async def check_connection(self) -> bool:
        """Check if Ollama server is accessible."""
        try:
            response = await self.client.get("/")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def cleanup(self):
        """Clean up the HTTP client."""
        await self.close()

    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload = await self.generate_response(
            prompt=prompt,
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            extra_options=extra_options,
        )
        return payload.get("response", "")

    async def _generate_text_with_dynamic_predict(
        self,
        prompt: str,
        model: str,
        temperature: float,
        num_predict: int,
        max_num_predict: int,
    ) -> Tuple[str, bool]:
        """Generate text with retries while increasing num_predict when truncated."""
        current_num_predict = num_predict

        while True:
            payload = await self.generate_response(
                prompt=prompt,
                model=model,
                temperature=temperature,
                num_ctx=None,
                num_predict=current_num_predict,
                extra_options=None,
            )

            text = payload.get("response", "")
            if payload.get("done", True):
                return text, False

            if current_num_predict >= max_num_predict:
                return text, True

            previous = current_num_predict
            current_num_predict = min(
                max_num_predict,
                max(current_num_predict + 512, int(current_num_predict * 1.5)),
            )
            logger.warning(
                "LLM response incomplete (num_predict=%s); retrying with num_predict=%s",
                previous,
                current_num_predict,
            )

    def _coerce_to_model_schema(
        self,
        parsed_data: Any,
        response_model: Type[BaseModel]
    ) -> Any:
        """Attempt to align parsed JSON with the expected response model schema."""
        expected_fields = set(response_model.model_fields.keys())

        if isinstance(parsed_data, dict):
            parsed_keys = set(parsed_data.keys())
            if expected_fields <= parsed_keys:
                return parsed_data

            # Check for common wrapper keys
            for key in ("data", "result", "output"):
                if key in parsed_data and isinstance(parsed_data[key], dict):
                    wrapper_value = parsed_data[key]
                    wrapper_keys = set(wrapper_value.keys())
                    if expected_fields <= wrapper_keys:
                        logger.info("Unwrapped '%s' object to match schema", key)
                        return wrapper_value

            # Check other nested dictionaries
            for value in parsed_data.values():
                if isinstance(value, dict):
                    value_keys = set(value.keys())
                    if expected_fields <= value_keys:
                        logger.info("Unwrapped nested object to match schema")
                        return value

        # Return as-is if we can't coerce
        return parsed_data
