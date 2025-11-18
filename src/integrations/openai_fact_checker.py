from __future__ import annotations

import json
import logging
import os
import textwrap
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

from src.utils.usage_tracker import record_openai_usage

logger = logging.getLogger(__name__)


class OpenAIFactChecker:
    """
    Lightweight wrapper around the OpenAI Responses API to run an automated
    fact-checking pass on the knowledge entry.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if OpenAI and api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
            if not api_key:
                logger.info("OpenAI fact checker disabled (missing OPENAI_API_KEY).")
        self.model = model or os.getenv("OPENAI_FACTCHECK_MODEL", "gpt-4o-mini")

    @property
    def available(self) -> bool:
        return self.client is not None

    def fact_check(self, document_id: str, entry_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None

        prompt = self._build_prompt(document_id, entry_payload)
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": "You are a meticulous fact-checking analyst for mental health content."},
                    {"role": "user", "content": prompt},
                ],
                reasoning={"effort": "medium"},
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("OpenAI fact check failed: %s", exc)
            return None

        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "input_tokens", 0) if usage else 0
        tokens_out = getattr(usage, "output_tokens", 0) if usage else 0
        record_openai_usage(
            "openai_factcheck",
            model=self.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            metadata={"document_id": document_id},
        )

        try:
            analysis = response.output_text.strip()
        except AttributeError:
            analysis = json.dumps(response.model_dump(), indent=2)

        return {
            "model": self.model,
            "analysis": analysis,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }

    def _build_prompt(self, document_id: str, entry_payload: Dict[str, Any]) -> str:
        serialized = json.dumps(entry_payload, ensure_ascii=False, indent=2)
        if len(serialized) > 20000:
            serialized = serialized[:19500] + "\n...<truncated>..."

        return textwrap.dedent(
            f"""
            Document ID: {document_id}

            Task:
            - Identify any claims that require citations or appear questionable.
            - Cross-check against credible neuroscience, trauma-informed therapy, or St. Louis regional sources.
            - Respond in Markdown with:
              * Verdict (OK / Needs Review)
              * Bullet list of flagged statements with supporting citations or recommended corrections.

            Knowledge entry JSON:
            {serialized}
            """
        ).strip()

