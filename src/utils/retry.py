"""Intelligent retry utilities with exponential backoff and self-correction prompts."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetryOutcome:
    success: bool
    attempts: int
    result: Any = None
    error: Optional[str] = None


class IntelligentRetryManager:
    """Reusable retry helper supporting exponential backoff and graceful degradation."""

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        base_delay: float = 2.0,
        backoff_factor: float = 2.0,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor

    def run(
        self,
        operation_name: str,
        func: Callable[[], Any],
        *,
        correction_prompt: Optional[str] = None,
        degradation: Optional[Callable[[], Any]] = None,
    ) -> RetryOutcome:
        attempts = 0
        delay = self.base_delay
        last_error: Optional[str] = None

        while attempts < self.max_attempts:
            attempts += 1
            try:
                logger.info("Executing %s (attempt %s/%s)", operation_name, attempts, self.max_attempts)
                result = func()
                logger.info("%s succeeded on attempt %s", operation_name, attempts)
                return RetryOutcome(True, attempts, result=result)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "%s failed on attempt %s/%s: %s", operation_name, attempts, self.max_attempts, exc
                )
                if correction_prompt:
                    logger.info("Self-correction guidance for %s: %s", operation_name, correction_prompt)

                if attempts < self.max_attempts:
                    logger.info("Waiting %.1f seconds before retrying %s", delay, operation_name)
                    time.sleep(delay)
                    delay *= self.backoff_factor

        logger.error("%s exhausted retries. Last error: %s", operation_name, last_error)
        fallback_result = None
        if degradation is not None:
            try:
                fallback_result = degradation()
                logger.info("Applied graceful degradation for %s", operation_name)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Degradation for %s failed: %s", operation_name, exc)
        return RetryOutcome(False, attempts, result=fallback_result, error=last_error)

