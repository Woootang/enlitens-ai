"""Gemini 2.5 Pro API client for persona generation."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type

import google.generativeai as genai
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for Gemini 2.5 Pro API with JSON schema enforcement."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key. If None, reads from credentials/gemini_api_key.txt
            model_name: Gemini model to use (gemini-2.0-flash-exp, gemini-exp-1206, etc.)
        """
        if api_key is None:
            key_path = Path(__file__).parent / "credentials" / "gemini_api_key.txt"
            api_key = key_path.read_text().strip()

        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"Initialized Gemini client with model: {model_name}")

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> Optional[BaseModel]:
        """
        Generate a structured response matching the Pydantic model schema.

        Args:
            prompt: User prompt
            response_model: Pydantic model class to validate against
            system_instruction: Optional system instruction
            temperature: Sampling temperature (0.0-2.0)
            max_retries: Number of retries on validation failure

        Returns:
            Validated Pydantic model instance or None on failure
        """
        # Use text generation with JSON instructions instead of response_schema
        # because Gemini's schema enforcement is very strict
        
        # Build prompt with clear JSON format instructions (but don't show the schema itself)
        json_prompt = f"{prompt}\n\n**IMPORTANT:** Respond with a complete, valid JSON document following the structure described above. Ensure all required fields are populated with rich, context-specific content. Do NOT return a schema or template—generate actual persona data."

        # Combine system instruction with prompt
        full_prompt = json_prompt
        if system_instruction:
            full_prompt = f"{system_instruction}\n\n{json_prompt}"

        # Build generation config (request JSON mime type without strict schema)
        generation_config = {
            "temperature": temperature,
            "response_mime_type": "application/json",
        }

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Generating structured response (attempt {attempt}/{max_retries})")

                response = self.model.generate_content(full_prompt, generation_config=generation_config)

                # Extract JSON from response
                if not response.text:
                    logger.warning(f"Attempt {attempt}: Empty response from Gemini")
                    continue

                # Parse and validate
                try:
                    response_dict = json.loads(response.text)
                except json.JSONDecodeError as e:
                    logger.warning(f"Attempt {attempt}: JSON decode error: {e}")
                    logger.debug(f"Response text: {response.text[:500]}")
                    continue

                # Validate with Pydantic
                validated = response_model.model_validate(response_dict)
                logger.info(f"Successfully generated and validated {response_model.__name__}")
                return validated

            except Exception as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    logger.error(f"All {max_retries} attempts failed for {response_model.__name__}")
                    return None

        return None

    def _clean_schema_for_gemini(self, schema: Any) -> Any:
        """
        Recursively remove fields that Gemini doesn't support.
        
        Gemini only supports: type, properties, required, items, description (in some contexts)
        """
        if isinstance(schema, dict):
            cleaned = {}
            # Keep only essential fields
            allowed_keys = {"type", "properties", "required", "items", "description", "enum", "format"}
            
            for key, value in schema.items():
                if key in allowed_keys:
                    cleaned[key] = self._clean_schema_for_gemini(value) if isinstance(value, (dict, list)) else value
                    
            return cleaned
        elif isinstance(schema, list):
            return [self._clean_schema_for_gemini(item) for item in schema]
        else:
            return schema

    def _inline_defs(self, schema: Dict[str, Any], defs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively inline $ref definitions into the schema.
        
        Args:
            schema: The schema dict to process
            defs: The $defs dict containing definitions
            
        Returns:
            Schema with all $refs inlined
        """
        if isinstance(schema, dict):
            if "$ref" in schema:
                # Extract the definition name
                ref_path = schema["$ref"]
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path.split("/")[-1]
                    if def_name in defs:
                        # Replace the $ref with the actual definition
                        resolved = defs[def_name].copy()
                        # Recursively inline any nested $refs
                        return self._inline_defs(resolved, defs)
                return schema
            else:
                # Recursively process all dict values
                return {k: self._inline_defs(v, defs) for k, v in schema.items()}
        elif isinstance(schema, list):
            # Recursively process list items
            return [self._inline_defs(item, defs) for item in schema]
        else:
            return schema

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None, temperature: float = 0.7) -> Optional[str]:
        """
        Generate plain text response.

        Args:
            prompt: User prompt
            system_instruction: Optional system instruction
            temperature: Sampling temperature

        Returns:
            Generated text or None on failure
        """
        generation_config = {"temperature": temperature}

        full_prompt = prompt
        if system_instruction:
            full_prompt = f"{system_instruction}\n\n{prompt}"

        try:
            response = self.model.generate_content(full_prompt, generation_config=generation_config)
            return response.text if response.text else None
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return None

