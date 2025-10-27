"""
Enhanced Ollama Client with JSON Repair and Schema Enforcement

This module provides an advanced client for interacting with Ollama models,
including JSON repair, schema enforcement, and structured output generation.
"""

import httpx
import json
import logging
from typing import List, Dict, Any, Optional, Type, Tuple
from json_repair import repair_json
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Enhanced Ollama client with JSON repair and schema enforcement.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "qwen3:32b"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=300.0)
        self.default_model = default_model
        logger.info(f"OllamaClient initialized with base_url: {self.base_url} and default_model: {self.default_model}")

    def clone_with_model(self, model: str) -> "OllamaClient":
        clone = OllamaClient(self.base_url, default_model=model)
        clone.client = self.client
        return clone

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
            logger.error(f"Ollama request failed: {e}")
            return {
                "response": f"Error: Ollama request failed - {e}",
                "done": True,
            }
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API error: {e.response.status_code} - {e.response.text}")
            return {
                "response": f"Error: Ollama API error - {e.response.status_code}",
                "done": True,
            }
        except Exception as e:
            logger.error(f"An unexpected error occurred during Ollama generation: {e}")
            return {
                "response": f"Error: An unexpected error occurred - {e}",
                "done": True,
            }

    async def generate_structured_response(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        model: str = "qwen3:32b",
        temperature: float = 0.7,
        max_retries: int = 3,
        base_num_predict: int = 2048,
        max_num_predict: int = 4096,
    ) -> Optional[BaseModel]:
        """
        Generate structured response with automatic JSON repair and validation.
        
        Args:
            prompt: The input prompt
            model: The model to use
            response_model: Pydantic model for validation
            temperature: Sampling temperature
            max_retries: Maximum number of retry attempts
            
        Returns:
            Validated Pydantic model instance or None if all retries failed
        """
        num_predict = base_num_predict
        attempt = 0

        while attempt < max_retries:
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} for structured generation")
                response_text, was_truncated = await self._generate_text_with_dynamic_predict(
                    prompt=prompt,
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
                    logger.error(f"Coerced data is not a dictionary: type={type(parsed_data)}, value={parsed_data}")
                    raise ValueError(f"Expected dict after coercion, got {type(parsed_data).__name__}")

                # Validate with Pydantic model
                validated_model = response_model.model_validate(parsed_data)
                data_dict = validated_model.model_dump()

                # Check if ALL lists are empty (not just ANY - some empty lists are fine)
                list_values = [v for v in data_dict.values() if isinstance(v, list)]
                if list_values and all(not v for v in list_values):
                    logger.warning(f"All {len(list_values)} list fields are empty - may indicate poor LLM output")
                    raise ValueError("All list fields are empty - no content generated")

                logger.info(f"Successfully validated response with {len(parsed_data)} keys")
                return validated_model
                
            except Exception as e:
                attempt += 1
                # Log detailed error information for debugging
                logger.warning(f"Attempt {attempt} failed: {e}")

                # Log response sample for debugging (first 500 chars)
                if 'response_text' in locals():
                    logger.debug(f"LLM response sample: {response_text[:500]}...")
                    logger.debug(f"Response length: {len(response_text)} characters")

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