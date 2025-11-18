from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import textwrap
from typing import Any, Dict, Optional

from src.utils.usage_tracker import record_usage

logger = logging.getLogger(__name__)


class GeminiJSONAssembler:
    """
    Helper for sending aggregated agent outputs to Gemini CLI to build a
    consolidated knowledge entry.
    """

    def __init__(self, executable: Optional[str] = None, timeout: int = 600) -> None:
        # Default to the installed Gemini CLI path
        default_path = "/home/antons-gs/.nvm/versions/node/v20.19.5/bin/gemini"
        self.executable = executable or os.getenv("GEMINI_CLI_PATH", default_path)
        self.timeout = timeout  # 10 minutes for large documents

    @property
    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def assemble_entry(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.available:
            logger.debug("Gemini CLI executable not found; skipping JSON assembly.")
            return None

        prompt = self._build_prompt(payload)
        cmd = [
            self.executable,
            "-p",
            prompt,
            "--output-format",
            "json",
        ]

        logger.info("🤝 Invoking Gemini CLI to consolidate agent outputs for %s", payload.get("document_id"))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError:
            logger.warning("Gemini CLI not installed or not in PATH.")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("Gemini CLI call exceeded %ss timeout.", self.timeout)
            return None

        if result.returncode != 0:
            logger.warning("Gemini CLI exited with code %s: %s", result.returncode, result.stderr.strip())
            return None

        raw_output = result.stdout.strip()
        parsed = self._extract_json(raw_output)
        if not parsed:
            logger.warning("Gemini CLI did not return valid JSON output.")
            return None

        record_usage("gemini_cli", metadata={"document_id": payload.get("document_id")})
        return parsed

    def _build_prompt(self, payload: Dict[str, Any]) -> str:
        metadata = payload.get("metadata") or {}
        agent_outputs = payload.get("agent_outputs") or {}
        quality = payload.get("quality") or {}
        document_text = payload.get("document_text") or ""

        def truncated_json(data: Any, limit: int = 50000) -> str:
            text = json.dumps(data, ensure_ascii=False, indent=2)
            if len(text) > limit:
                return text[: limit - 500] + "\n...<truncated>..."
            return text

        # Truncate document text to fit within Gemini's 1M token context window
        # Approximately 4 chars per token, so 800K chars ≈ 200K tokens (leaving room for prompt + response)
        max_doc_length = 800000
        if len(document_text) > max_doc_length:
            document_text = document_text[:max_doc_length] + "\n\n[Document truncated for length...]"

        agent_json = truncated_json(agent_outputs)

        prompt = textwrap.dedent(
            f"""
            You are an expert Enlitens data steward working inside Gemini CLI.
            You have access to the FULL document text and all extraction/enrichment outputs.
            
            Your task:
            1. Review the full document text for accuracy and completeness
            2. Validate and enhance the extraction outputs
            3. Create a comprehensive knowledge entry that will be stored in PostgreSQL, ChromaDB, and Neo4j
            4. Ensure all scientific claims are grounded in the document text
            
            Requirements:
            - Preserve factual accuracy; do not invent new claims.
            - Respect trauma-informed and neurodiversity-affirming language.
            - Include any warning metadata under a `verification.deep_research` object if relevant.
            - Return ONLY JSON with no commentary.
            - The JSON should include: metadata, extracted_entities, educational_content, clinical_implications, key_findings

            Document metadata:
            {json.dumps(metadata, ensure_ascii=False, indent=2)}

            Quality metrics:
            {json.dumps(quality, ensure_ascii=False, indent=2)}

            FULL DOCUMENT TEXT:
            {document_text}

            Agent extraction outputs to validate and merge:
            {agent_json}
            """
        ).strip()
        return prompt

    @staticmethod
    def _extract_json(blob: str) -> Optional[Dict[str, Any]]:
        if not blob:
            return None
        start = blob.find("{")
        end = blob.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = blob[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

