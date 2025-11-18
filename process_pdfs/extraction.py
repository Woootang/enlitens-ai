#!/usr/bin/env python3
"""
Scientific Extraction Module
Uses LLM to extract COMPREHENSIVE scientific content from papers
"""
import copy
import json
import logging
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import sys
sys.path.insert(0, '/home/antons-gs/enlitens-ai')

from src.utils.llm_client import LLMClient
from src.utils.usage_tracker import record_usage

logger = logging.getLogger(__name__)

DIRECT_TEXT_CHAR_LIMIT = 10000
CHUNK_CHAR_LIMIT = 2500
CHUNK_CHAR_OVERLAP = 150
CONTEXT_SNIPPET_CHAR_LIMIT = 8000

DEFAULT_FIELD_RULES: Dict[str, Dict[str, Any]] = {
    "background": {"type": "string", "min_chars": 1000, "max_chars": 8000},
    "methods": {"type": "string", "min_chars": 1500, "max_chars": 9000},
    "findings": {"type": "string", "min_chars": 1500, "max_chars": 9000},
    "statistics": {"type": "string", "min_chars": 500, "max_chars": 4000},
    "limitations": {"type": "string", "min_chars": 500, "max_chars": 4000},
    "conclusions": {"type": "string", "min_chars": 1000, "max_chars": 6000},
    "citations": {"type": "array", "min_items": 10, "max_items": 30},
}

MEDGEMMA_FIELD_RULES: Dict[str, Dict[str, Any]] = {
    "background": {"type": "string", "min_chars": 800, "max_chars": 6000},
    "methods": {"type": "string", "min_chars": 1200, "max_chars": 6000},
    "findings": {"type": "string", "min_chars": 1200, "max_chars": 6000},
    "statistics": {"type": "string", "min_chars": 300, "max_chars": 2500},
    "limitations": {"type": "string", "min_chars": 400, "max_chars": 2500},
    "conclusions": {"type": "string", "min_chars": 800, "max_chars": 4000},
    "citations": {"type": "array", "min_items": 8, "max_items": 24},
}

MAJOR_FIELDS = ["background", "methods", "findings"]

SECTION_ORDER = [
    "background",
    "methods",
    "findings",
    "statistics",
    "limitations",
    "conclusions",
    "citations",
]

SECTION_INSTRUCTIONS: Dict[str, str] = {
    "background": (
        "Provide the complete theoretical framework, prior work, knowledge gaps, and stated hypotheses. "
        "Reference concrete details from the source (populations, biomarkers, prior studies). "
        "Write in academic paragraphs, no bullet points."
    ),
    "methods": (
        "Describe study design, participants (N, demographics, inclusion/exclusion), recruitment, procedures, timing, "
        "instruments, and analytic strategy. Include every numerical detail (sample sizes, sessions, dosages, software, "
        "thresholds). Use full sentences and paragraph structure."
    ),
    "findings": (
        "Cover all results: primary, secondary, subgroup, unexpected, including effect sizes and interpretations. "
        "Pair every result with exact statistics (means, deltas, p-values, confidence intervals). Explain what the numbers mean in context."
    ),
    "statistics": (
        "Enumerate all descriptive and inferential statistics (Ns, means, SDs, test statistics, p-values, CIs, effect sizes, corrections). "
        "Present as rich sentences with clear context."
    ),
    "limitations": (
        "Detail every limitation, caveat, bias, generalizability issue, measurement constraint, and why it matters. "
        "Use paragraphs, no bullets."
    ),
    "conclusions": (
        "Summarize authors' interpretations, clinical/practical implications, theoretical impact, and future work recommendations. "
        "Ensure full narrative paragraphs."
    ),
    "citations": (
        "Return an array with the key references cited in the paper. "
        "Format each entry as 'Author et al. (Year). Title' or 'Author (Year)'. "
        "Ensure entries are unique and relevant."
    ),
}


def _describe_field_requirements(field: str, field_rules: Dict[str, Dict[str, Any]]) -> str:
    rules = field_rules[field]
    base = SECTION_INSTRUCTIONS[field]
    if rules["type"] == "string":
        return (
            f"{field}: deliver {rules['min_chars']:,}–{rules['max_chars']:,} characters of dense paragraphs. "
            f"Cover: {base}"
        )
    return (
        f"{field}: return an array with {rules['min_items']}–{rules['max_items']} unique citation strings. "
        f"Guidance: {base}"
    )


def _build_requirements_hint(fields: List[str], field_rules: Dict[str, Dict[str, Any]]) -> str:
    if not fields:
        return ""
    descriptions = [_describe_field_requirements(field, field_rules) for field in fields]
    return " | ".join(descriptions)


def _extract_json_blob(payload: str) -> Optional[Dict[str, Any]]:
    if not payload:
        return None
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = payload[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _gemini_section_fallback(
    field: str,
    context_text: str,
    field_rules: Dict[str, Dict[str, Any]],
    current_value: Optional[Any],
) -> Optional[Any]:
    executable = os.getenv("GEMINI_CLI_PATH", "gemini")
    if shutil.which(executable) is None:
        return None

    rules = field_rules[field]
    requirements = _describe_field_requirements(field, field_rules)
    existing = current_value if current_value else "<empty>"
    context_snippet = context_text[:CONTEXT_SNIPPET_CHAR_LIMIT]

    prompt = textwrap.dedent(
        f"""
        You are repairing a JSON extraction for a neuroscience paper.
        Fill ONLY the '{field}' field according to the requirements below.

        Requirements:
        - {requirements}
        - Output exactly one JSON object with the single key "{field}".
        - Do NOT wrap the JSON in quotes or provide any commentary.

        Existing value (may be invalid or empty):
        {existing}

        Aggregated scientific context:
        {context_snippet}
        """
    ).strip()

    cmd = [executable, "--output-format", "json"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            input=prompt,
            timeout=240,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.info("Gemini CLI fallback unavailable or timed out for section %s.", field)
        return None

    if result.returncode != 0:
        logger.info(
            "Gemini CLI fallback exited with code %s for section %s: %s",
            result.returncode,
            field,
            result.stderr.strip(),
        )
        return None

    parsed = _extract_json_blob(result.stdout)
    if not parsed:
        logger.info("Gemini CLI fallback did not return parseable JSON for %s.", field)
        return None

    record_usage("gemini_cli", metadata={"task": "section_repair", "field": field})
    return parsed.get(field)

CHUNK_SUMMARY_JSON_SCHEMA = {
    "name": "chunk_summary",
    "schema": {
        "type": "object",
        "properties": {
            "chunk_index": {"type": "integer"},
            "chunk_summary": {
                "type": "string",
                "minLength": 400,
                "maxLength": 6000,
                "pattern": '^[^"]*$',
            },
            "background_points": {
                "type": "array",
                "items": {
                    "type": "string",
                    "pattern": '^[^"]*$',
                    "maxLength": 1000,
                },
                "default": [],
            },
            "method_points": {
                "type": "array",
                "items": {
                    "type": "string",
                    "pattern": '^[^"]*$',
                    "maxLength": 1200,
                },
                "default": [],
            },
            "finding_points": {
                "type": "array",
                "items": {
                    "type": "string",
                    "pattern": '^[^"]*$',
                    "maxLength": 1200,
                },
                "default": [],
            },
            "statistics": {
                "type": "array",
                "items": {
                    "type": "string",
                    "pattern": '^[^"]*$',
                    "maxLength": 800,
                },
                "default": [],
            },
            "limitations": {
                "type": "array",
                "items": {
                    "type": "string",
                    "pattern": '^[^"]*$',
                    "maxLength": 600,
                },
                "default": [],
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "string",
                    "pattern": '^[^"]*$',
                    "maxLength": 200,
                },
                "default": [],
            },
            "verbatim_quotes": {
                "type": "array",
                "items": {
                    "type": "string",
                    "pattern": '^[^"]*$',
                    "maxLength": 800,
                },
                "default": [],
            },
        },
        "required": ["chunk_index", "chunk_summary"],
        "additionalProperties": False,
    },
}

EXTRACTION_JSON_SCHEMA = {
    "name": "scientific_extraction",
    "schema": {
        "type": "object",
        "properties": {
            "background": {
                "type": "string",
                "minLength": 800,
                "maxLength": 8000,
                "pattern": '^[^"]*$',
            },
            "methods": {
                "type": "string",
                "minLength": 1200,
                "maxLength": 9000,
                "pattern": '^[^"]*$',
            },
            "findings": {
                "type": "string",
                "minLength": 1200,
                "maxLength": 9000,
                "pattern": '^[^"]*$',
            },
            "statistics": {
                "type": "string",
                "minLength": 400,
                "maxLength": 4000,
                "pattern": '^[^"]*$',
            },
            "limitations": {
                "type": "string",
                "minLength": 400,
                "maxLength": 4000,
                "pattern": '^[^"]*$',
            },
            "conclusions": {
                "type": "string",
                "minLength": 800,
                "maxLength": 6000,
                "pattern": '^[^"]*$',
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "string",
                    "pattern": '^[^"]*$',
                    "maxLength": 200,
                },
                "minItems": 5,
            },
        },
        "required": [
            "background",
            "methods",
            "findings",
            "statistics",
            "limitations",
            "conclusions",
            "citations",
        ],
        "additionalProperties": False,
    },
}


def load_extraction_prompt() -> str:
    """Load the extraction prompt template."""
    prompt_path = Path(__file__).parent / "prompts" / "extraction_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Extraction prompt not found: {prompt_path}")
    return prompt_path.read_text()


def load_chunk_summary_prompt() -> str:
    """Load the chunk summary prompt template."""
    prompt_path = Path(__file__).parent / "prompts" / "chunk_summary_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Chunk summary prompt not found: {prompt_path}")
    return prompt_path.read_text()


def _chunk_document_text(text: str) -> List[str]:
    """Split document into overlapping character chunks to control prompt size."""
    if len(text) <= CHUNK_CHAR_LIMIT:
        return [text]

    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + CHUNK_CHAR_LIMIT, length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == length:
            break
        start = max(end - CHUNK_CHAR_OVERLAP, 0)
    return chunks


def _sanitize_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if value is None:
        return []
    return [str(value).strip()]


def _summarize_long_document(
    document_text: str,
    llm_client: LLMClient
) -> Tuple[str, str]:
    """
    Summarize long documents into chunk overviews to stay within context limits.

    Returns:
        aggregated_summary_text, source_description
    """
    if len(document_text) <= DIRECT_TEXT_CHAR_LIMIT:
        return document_text, "Full verbatim paper text"

    chunk_prompt_template = load_chunk_summary_prompt()
    is_medgemma = "medgemma" in llm_client.model_name.lower()
    chunk_temperature = 0.12 if is_medgemma else 0.15
    chunks = _chunk_document_text(document_text)
    summaries: List[Dict[str, Any]] = []

    logger.info(f"Document length {len(document_text)} chars exceeds limit; chunking into {len(chunks)} segments.")

    for index, chunk in enumerate(chunks, start=1):
        prompt = (
            chunk_prompt_template
            .replace("{CHUNK_INDEX}", str(index))
            .replace("{CHUNK_TEXT}", chunk)
        )
        try:
            summary = llm_client.generate_json(
                prompt=prompt,
                max_tokens=512,
                temperature=chunk_temperature,
                timeout=900,
                json_schema=CHUNK_SUMMARY_JSON_SCHEMA,
            )
            summary["chunk_index"] = summary.get("chunk_index") or index
            summaries.append(summary)
            logger.info(f"✅ Chunk {index} summarized (len={len(chunk)} chars)")
        except Exception as exc:
            logger.error(f"Chunk summarization failed for chunk {index}: {exc}")
            raise

    context_lines: List[str] = [
        "The following consolidated digest is derived from exhaustive chunk-level summaries.",
        "Treat each chunk section as authoritative context sourced from the original PDF."
    ]

    for summary in summaries:
        idx = summary.get("chunk_index", len(context_lines))
        context_lines.append(f"\n### Chunk {idx} Summary")

        chunk_summary = summary.get("chunk_summary", "")
        if chunk_summary:
            context_lines.append(chunk_summary.strip())

        background = _sanitize_list(summary.get("background_points"))
        if background:
            context_lines.append("Background Points: " + " | ".join(background))

        methods = _sanitize_list(summary.get("method_points"))
        if methods:
            context_lines.append("Method Notes: " + " | ".join(methods))

        findings = _sanitize_list(summary.get("finding_points"))
        if findings:
            context_lines.append("Findings: " + " | ".join(findings))

        stats = _sanitize_list(summary.get("statistics"))
        if stats:
            context_lines.append("Statistics: " + " | ".join(stats))

        limitations = _sanitize_list(summary.get("limitations"))
        if limitations:
            context_lines.append("Limitations: " + " | ".join(limitations))

        quotes = _sanitize_list(summary.get("verbatim_quotes"))
        if quotes:
            context_lines.append("Verbatim Quotes: " + " | ".join(quotes))

        citations = _sanitize_list(summary.get("citations"))
        if citations:
            context_lines.append("Citations: " + ", ".join(citations))

    aggregated_summary = "\n".join(context_lines).strip()
    description = (
        "Aggregated chunk summaries synthesized from the full paper; "
        "each chunk preserves detailed findings, statistics, and citations."
    )
    return aggregated_summary, description


def _build_single_field_schema(field: str, field_rules: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rules = field_rules[field]
    if rules["type"] == "string":
        field_schema: Dict[str, Any] = {"type": "string"}
    else:
        field_schema = {"type": "array", "items": {"type": "string"}}

    return {
        "name": f"{field}_section",
        "schema": {
            "type": "object",
            "properties": {field: field_schema},
            "required": [field],
            "additionalProperties": False,
        },
    }


def _normalize_citations(value: Any) -> List[str]:
    if isinstance(value, list):
        entries = value
    elif isinstance(value, str):
        entries = re.split(r"[\n;]+", value)
    else:
        entries = []
    cleaned = []
    for entry in entries:
        if not entry:
            continue
        text = str(entry).strip(" \t-•")
        if text:
            cleaned.append(text)
    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for item in cleaned:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _get_field_rules(model_name: str) -> Dict[str, Dict[str, Any]]:
    if "medgemma" in model_name.lower():
        return MEDGEMMA_FIELD_RULES
    return DEFAULT_FIELD_RULES


def _analyze_result(result: Dict[str, Any], field_rules: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    missing: List[str] = []
    shallow: List[str] = []
    for field, rules in field_rules.items():
        value = result.get(field)
        if value is None:
            missing.append(field)
            continue
        if rules["type"] == "string":
            text = str(value).strip()
            if not text:
                missing.append(field)
                continue
            if len(text) < rules["min_chars"]:
                shallow.append(field)
        else:
            normalized = _normalize_citations(value)
            if not normalized:
                missing.append(field)
                continue
            result[field] = normalized
            if len(normalized) < rules["min_items"]:
                shallow.append(field)
    return missing, shallow


def _validate_field(field: str, value: Any, field_rules: Dict[str, Dict[str, Any]]) -> bool:
    rules = field_rules[field]
    if rules["type"] == "string":
        if not isinstance(value, str):
            return False
        length = len(value.strip())
        if length < rules["min_chars"]:
            return False
        if length > rules["max_chars"]:
            return False
        return True
    if rules["type"] == "array":
        if isinstance(value, str):
            value = _normalize_citations(value)
        if not isinstance(value, list):
            return False
        length = len(value)
        if length < rules["min_items"]:
            return False
        if length > rules["max_items"]:
            # Trim but keep order
            del value[rules["max_items"] :]
        return True
    return False


def _build_section_prompt(
    field: str,
    context_text: str,
    field_rules: Dict[str, Dict[str, Any]],
    model_name: str,
) -> str:
    rules = field_rules[field]
    if rules["type"] == "string":
        length_clause = (
            f"- Target length: at least {rules['min_chars']} characters (keep under {rules['max_chars']} characters).\n"
        )
    else:
        length_clause = (
            f"- Return an array with {rules['min_items']} to {rules['max_items']} citation strings.\n"
            "- Each entry should follow 'Author et al. (Year). Title' or 'Author (Year)'.\n"
        )
    instructions = SECTION_INSTRUCTIONS[field]
    context_snippet = context_text[:15000]
    return (
        "You are an expert scientific extraction specialist.\n"
        f"Focus EXCLUSIVELY on the `{field}` section of the paper. "
        "Do not mention any other keys.\n\n"
        "Requirements:\n"
        f"{length_clause}"
        f"- {instructions}\n"
        "- Use complete sentences and paragraphs (no bullets) unless returning an array.\n"
        "- Output raw JSON. Do not wrap the object in quotes or escape it.\n"
        "- Output ONLY a JSON object containing the single key.\n"
        "- Avoid unescaped double quotes inside string values.\n\n"
        "SOURCE CONTEXT (aggregated chunk summaries):\n"
        f"{context_snippet}"
    )


def _rescue_sections(
    context_text: str,
    llm_client: LLMClient,
    existing_result: Dict[str, Any],
) -> Dict[str, Any]:
    logger.info("Attempting targeted section rescue to fill missing or shallow fields.")
    repaired = dict(existing_result)
    field_rules = _get_field_rules(llm_client.model_name)
    model_name = llm_client.model_name.lower()
    is_medgemma = "medgemma" in model_name
    temperature = 0.12 if is_medgemma else 0.2
    max_attempts = 4 if is_medgemma else 3

    fields_to_fix = set()
    for field in field_rules:
        value = repaired.get(field)
        if value is None or (field != "citations" and not str(value).strip()):
            fields_to_fix.add(field)
            continue
        if field == "citations":
            normalized = _normalize_citations(value)
            if not _validate_field(field, normalized, field_rules):
                fields_to_fix.add(field)
            else:
                repaired[field] = normalized
        elif not _validate_field(field, value, field_rules):
            fields_to_fix.add(field)

    for field in fields_to_fix:
        section_schema = _build_single_field_schema(field, field_rules)
        prompt_base = _build_section_prompt(field, context_text, field_rules, llm_client.model_name)
        reminder = "\n\nFORMAT: Return exactly {\"" + field + "\": \"...\"} or an array for citations."
        success = False
        for attempt in range(max_attempts):
            prompt = prompt_base + reminder
            try:
                response = llm_client.generate_json(
                    prompt=prompt,
                    max_tokens=2200 if field_rules[field]["type"] == "string" else 1024,
                    temperature=temperature,
                    timeout=1200,
                    json_schema=section_schema,
                    max_attempts=max_attempts,
                )
                candidate = response.get(field)
                if field_rules[field]["type"] == "array":
                    candidate = _normalize_citations(candidate)
                if _validate_field(field, candidate, field_rules):
                    repaired[field] = candidate
                    success = True
                    logger.info("✅ Section %s reconstructed (attempt %s)", field, attempt + 1)
                    break
                logger.info("Section %s still invalid after attempt %s", field, attempt + 1)
            except Exception as exc:
                logger.error("Section %s reconstruction attempt %s failed: %s", field, attempt + 1, exc)
            reminder += (
                f"\n\nREMINDER: You must output JSON with only the '{field}' key meeting the stated length/detail requirements. "
                "Return raw JSON, no code fences or narration."
            )
        if not success:
            gemini_candidate = _gemini_section_fallback(
                field=field,
                context_text=context_text,
                field_rules=field_rules,
                current_value=repaired.get(field),
            )
            if gemini_candidate is not None:
                if field_rules[field]["type"] == "array":
                    gemini_candidate = _normalize_citations(gemini_candidate)
                if _validate_field(field, gemini_candidate, field_rules):
                    repaired[field] = gemini_candidate
                    logger.info("✅ Section %s reconstructed via Gemini CLI fallback", field)
                    continue
                logger.info("Gemini CLI fallback produced invalid content for %s.", field)
            logger.error("Unable to repair section %s after targeted attempts.", field)
            raise RuntimeError(f"Failed to reconstruct section '{field}'")

    return repaired


def _extract_with_sequential_sections(
    context_text: str,
    llm_client: LLMClient,
    field_rules: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    logger.info("Using sequential section generation strategy.")
    result: Dict[str, Any] = {}
    model_name = llm_client.model_name.lower()
    is_medgemma = "medgemma" in model_name
    temperature = 0.12 if is_medgemma else 0.2
    max_attempts = 4 if is_medgemma else 3

    for field in SECTION_ORDER:
        section_schema = _build_single_field_schema(field, field_rules)
        prompt_base = _build_section_prompt(field, context_text, field_rules, llm_client.model_name)
        reminder = ""
        success = False
        for attempt in range(max_attempts):
            prompt = prompt_base + reminder
            try:
                response = llm_client.generate_json(
                    prompt=prompt,
                    max_tokens=1600 if field_rules[field]["type"] == "string" else 800,
                    temperature=temperature,
                    timeout=1200,
                    json_schema=section_schema,
                    max_attempts=max_attempts,
                )
                candidate = response.get(field)
                if field_rules[field]["type"] == "array":
                    candidate = _normalize_citations(candidate)
                if _validate_field(field, candidate, field_rules):
                    result[field] = candidate
                    success = True
                    logger.info("✅ Generated %s section (attempt %s)", field, attempt + 1)
                    break
                logger.info("Section %s validation failed (attempt %s)", field, attempt + 1)
            except Exception as exc:
                logger.error("Section %s generation attempt %s failed: %s", field, attempt + 1, exc)
            reminder += (
                f"\n\nREMINDER: Return JSON with only the '{field}' key and satisfy the length/format requirements. "
                "Do not include narrative outside the JSON object."
            )
        if not success:
            raise RuntimeError(f"Failed to generate section '{field}'")
    return result


def extract_scientific_content(
    document_text: str,
    llm_client: LLMClient,
    max_retries: int = 2,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Extract scientific content from document using LLM
    
    Args:
        document_text: Full text of the research paper
        llm_client: Initialized LLM client
        max_retries: Number of retry attempts if extraction fails
        
    Returns:
        Dictionary with extracted scientific content
    """
    prompt_template = load_extraction_prompt()

    # Hard safety guard against extremely long raw text
    max_chars = 400000  # ~100k tokens
    working_text = document_text
    if len(working_text) > max_chars:
        logger.warning(f"Document too long ({len(working_text)} chars), truncating to {max_chars}")
        working_text = working_text[:max_chars] + "\n\n[Document truncated due to length]"

    context_text, source_description = _summarize_long_document(working_text, llm_client)

    base_prompt = (
        prompt_template
        .replace("{SOURCE_DESCRIPTION}", source_description)
        .replace("{DOCUMENT_TEXT}", context_text)
    )

    logger.info(
        "Extracting scientific content "
        f"(original chars: {len(document_text)}, context chars: {len(context_text)})"
    )

    model_lower = llm_client.model_name.lower()
    field_rules = _get_field_rules(llm_client.model_name)
    is_medgemma = "medgemma" in model_lower

    sequential_threshold = int(os.getenv("ENLITENS_SEQUENTIAL_THRESHOLD_CHARS", "180000"))
    requires_sequential = len(context_text) >= sequential_threshold

    if is_medgemma or requires_sequential:
        if requires_sequential and not is_medgemma:
            logger.info(
                "Context length %d chars exceeds threshold %d; using sequential extraction.",
                len(context_text),
                sequential_threshold,
            )
        result = _extract_with_sequential_sections(context_text, llm_client, field_rules)
        logger.info("✅ Sequential extraction successful for %s.", llm_client.model_name)
        if metadata:
            result["metadata"] = copy.deepcopy(metadata)
        return result

    local_max_retries = max_retries
    generation_temperature = 0.2
    generation_attempts = 3

    retry_suffix = ""

    for attempt in range(local_max_retries + 1):
        prompt = base_prompt + retry_suffix
        try:
            result = llm_client.generate_json(
                prompt=prompt,
                max_tokens=4096,
                temperature=generation_temperature,
                timeout=1800,
                json_schema=EXTRACTION_JSON_SCHEMA,
                max_attempts=generation_attempts,
            )

            missing_fields, shallow_fields = _analyze_result(result, field_rules)

            if missing_fields:
                logger.info(f"Missing fields: {missing_fields}")
                if attempt < local_max_retries:
                    hint = _build_requirements_hint(missing_fields, field_rules)
                    retry_suffix += (
                        "\n\nCRITICAL NOTICE: Your previous reply was missing the following JSON keys: "
                        f"{missing_fields}. You MUST return a SINGLE JSON object containing EVERY required key "
                        "with detailed, paragraph-level content meeting the character minimums. Do not omit any field."
                    )
                    if hint:
                        retry_suffix += f"\nDETAILED REQUIREMENTS: {hint}"
                    logger.info("Retrying extraction (attempt %s/%s)", attempt + 2, local_max_retries + 1)
                    continue
                try:
                    result = _rescue_sections(context_text, llm_client, result)
                    missing_fields, shallow_fields = _analyze_result(result, field_rules)
                except Exception as rescue_exc:
                    logger.error("Targeted section rescue failed: %s", rescue_exc)
                    # Graceful degradation: accept partial results with placeholder content
                    logger.warning("⚠️ Accepting partial extraction with placeholder content for missing fields")
                    for field in missing_fields:
                        if field not in result or not result[field]:
                            result[field] = f"[Content unavailable - extraction failed for {field}]"
                    missing_fields = []  # Clear to allow processing to continue

            if shallow_fields:
                logger.info(f"Shallow extraction detected in fields: {shallow_fields}")
                if attempt < local_max_retries:
                    hint = _build_requirements_hint(shallow_fields, field_rules)
                    retry_suffix += (
                        "\n\nIMPORTANT REMINDER: The following fields were TOO BRIEF: "
                        f"{shallow_fields}. Provide exhaustive paragraphs with all quantitative details. "
                        "Minimum character requirements are non-negotiable."
                    )
                    if hint:
                        retry_suffix += f"\nDETAILED REQUIREMENTS: {hint}"
                    continue
                try:
                    result = _rescue_sections(context_text, llm_client, result)
                    missing_fields, shallow_fields = _analyze_result(result, field_rules)
                except Exception as rescue_exc:
                    logger.error("Section depth repair failed: %s", rescue_exc)
                    # Graceful degradation: accept shallow content rather than failing
                    logger.warning("⚠️ Accepting shallow extraction - some fields may be brief")
                    shallow_fields = []  # Clear to allow processing to continue

            if missing_fields or shallow_fields:
                # Final check: if we still have issues, log but don't fail
                logger.warning(
                    f"⚠️ Extraction quality degraded (missing={missing_fields}, shallow={shallow_fields}) - proceeding with available content"
                )
                # Fill any remaining missing fields with placeholders
                for field in missing_fields:
                    if field not in result or not result[field]:
                        result[field] = f"[Content unavailable - extraction incomplete for {field}]"

            narrative_fields = [f for f, cfg in field_rules.items() if cfg["type"] == "string"]
            avg_length = sum(len(str(result.get(f, ""))) for f in narrative_fields) / len(narrative_fields)
            logger.info(f"✅ Extraction successful (avg {avg_length:.0f} chars across narrative fields)")
            if metadata:
                result["metadata"] = copy.deepcopy(metadata)
            return result

        except Exception as e:
            logger.error(f"Extraction attempt {attempt + 1} failed: {e}")
            if attempt == local_max_retries:
                raise

    raise RuntimeError("Extraction failed after all retries")

