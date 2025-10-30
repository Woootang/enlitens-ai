"""Testing harness for evaluating prompt variants against golden criteria."""

from __future__ import annotations

import difflib
from typing import Any, Dict

from src.synthesis.few_shot_library import FEW_SHOT_LIBRARY, PromptCriterion


class PromptHarness:
    """Evaluate prompt variants for a task using the golden few-shot criteria."""

    def __init__(self) -> None:
        self.library = FEW_SHOT_LIBRARY

    def evaluate_prompt_variants(self, task: str, prompt_variants: Dict[str, str]) -> Dict[str, Any]:
        criteria = self.library.aggregate_criteria(task)
        variant_results: Dict[str, Any] = {}

        for name, prompt in prompt_variants.items():
            variant_results[name] = self._evaluate_single(prompt, criteria)

        average_score = (
            sum(result["score"] for result in variant_results.values()) / len(variant_results)
            if variant_results
            else 0.0
        )

        return {
            "task": task,
            "criteria": [
                {"name": criterion.name, "keywords": criterion.keywords, "description": criterion.description}
                for criterion in criteria
            ],
            "variants": variant_results,
            "average_score": average_score,
            "best_variant": self.rank_variants(variant_results)[0] if variant_results else None,
        }

    def rank_variants(self, variant_results: Dict[str, Any]) -> list[str]:
        return [
            name
            for name, _ in sorted(
                variant_results.items(),
                key=lambda item: (item[1]["score"], item[1]["lexical_similarity"]),
                reverse=True,
            )
        ]

    def _evaluate_single(self, prompt: str, criteria: list[PromptCriterion]) -> Dict[str, Any]:
        coverage_details = []
        met_count = 0
        for criterion in criteria:
            met = criterion.matches(prompt)
            if met:
                met_count += 1
            missing_keywords = [kw for kw in criterion.keywords if kw.lower() not in prompt.lower()]
            coverage_details.append(
                {
                    "criterion": criterion.name,
                    "met": met,
                    "missing_keywords": missing_keywords,
                }
            )

        score = met_count / len(criteria) if criteria else 1.0
        reference_text = " ".join(criterion.description for criterion in criteria)
        lexical_similarity = difflib.SequenceMatcher(None, prompt.lower(), reference_text.lower()).ratio()

        missing_criteria = [detail["criterion"] for detail in coverage_details if not detail["met"]]

        return {
            "score": score,
            "lexical_similarity": lexical_similarity,
            "coverage": coverage_details,
            "missing_criteria": missing_criteria,
        }
