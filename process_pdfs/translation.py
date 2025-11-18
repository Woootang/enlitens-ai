#!/usr/bin/env python3
"""
Clinical Translation Module
Translates scientific findings into COMPREHENSIVE actionable clinical guidance
"""
import logging
import json
from pathlib import Path
from typing import Dict
import sys
sys.path.insert(0, '/home/antons-gs/enlitens-ai')

from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


def load_translation_prompt() -> str:
    """Load the translation prompt template"""
    prompt_path = Path(__file__).parent / "prompts" / "translation_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Translation prompt not found: {prompt_path}")
    return prompt_path.read_text()


def translate_to_clinical(
    scientific_extraction: Dict,
    llm_client: LLMClient,
    max_retries: int = 2
) -> Dict:
    """
    Translate scientific findings into clinical guidance
    
    Args:
        scientific_extraction: Output from extraction module
        llm_client: Initialized LLM client
        max_retries: Number of retry attempts if translation fails
        
    Returns:
        Dictionary with clinical translation
    """
    prompt_template = load_translation_prompt()
    
    extraction_str = json.dumps(scientific_extraction, indent=2)
    prompt = prompt_template.replace("{SCIENTIFIC_EXTRACTION}", extraction_str)
    
    logger.info("Translating findings to clinical guidance")
    
    # Attempt translation with retries
    for attempt in range(max_retries + 1):
        try:
            result = llm_client.generate_json(
                prompt=prompt,
                max_tokens=4096,  # Deep outputs without overwhelming VRAM
                temperature=0.3,
                timeout=1800  # 30 minute timeout
            )
            
            # Validate required fields
            required_fields = ["interventions", "protocols", "assessments", "contraindications", "monitoring", "evidence_summary"]
            missing_fields = [f for f in required_fields if f not in result or not result[f]]
            
            if missing_fields:
                logger.warning(f"Missing fields: {missing_fields}")
                if attempt < max_retries:
                    logger.info(f"Retrying translation (attempt {attempt + 2}/{max_retries + 1})")
                    continue
            
            # Check for shallow translation (MINIMUM 1000 chars for major fields)
            major_fields = ["interventions", "protocols", "evidence_summary"]
            shallow_fields = [f for f in major_fields if len(str(result.get(f, ""))) < 1000]
            
            if shallow_fields:
                logger.warning(f"Shallow translation detected in fields: {shallow_fields}")
                if attempt < max_retries:
                    prompt += f"\n\nIMPORTANT: Your previous response was TOO BRIEF in these fields: {shallow_fields}. You MUST provide MINIMUM 1000 characters with EXHAUSTIVE detail. Write FULL PARAGRAPHS with step-by-step protocols."
                    continue
            
            avg_length = sum(len(str(result.get(f, ""))) for f in required_fields) / len(required_fields)
            logger.info(f"✅ Translation successful (avg {avg_length:.0f} chars per field)")
            return result
            
        except Exception as e:
            logger.error(f"Translation attempt {attempt + 1} failed: {e}")
            if attempt == max_retries:
                raise
    
    raise RuntimeError("Translation failed after all retries")

