"""Schema for storing structured prediction errors tied to client profiles."""
from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class PredictionErrorEntry(BaseModel):
    """Unexpected-yet-relevant pivot that reframes an intake assumption."""

    trigger_context: str = Field(
        ...,
        description=(
            "Short description of the intake belief, habit, or regional pattern"
            " that primes the surprise."
        ),
    )
    surprising_pivot: str = Field(
        ...,
        description=(
            "Locally grounded twist that subverts the trigger while staying"
            " faithful to research citations or community knowledge."
        ),
    )
    intended_cognitive_effect: str = Field(
        ...,
        description=(
            "Explanation of the mindset shift, reframing, or curiosity the pivot"
            " should spark for the reader."
        ),
    )

    _MIN_LENGTH: ClassVar[int] = 8

    @field_validator("trigger_context", "surprising_pivot", "intended_cognitive_effect")
    @classmethod
    def _ensure_descriptive_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if len(text) < cls._MIN_LENGTH:
            raise ValueError(
                "Prediction error fields must contain descriptive text of at"
                " least eight characters."
            )
        return text

    @field_validator("surprising_pivot")
    @classmethod
    def _pivot_differs_from_trigger(cls, value: str, info: ValidationInfo) -> str:
        text = value.strip()
        trigger = info.data.get("trigger_context")
        if trigger and trigger.strip().lower() == text.lower():
            raise ValueError("Surprising pivot must meaningfully differ from the trigger context.")
        return text
