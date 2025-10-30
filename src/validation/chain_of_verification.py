"""Chain-of-Verification utilities for validating multi-agent outputs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List


logger = logging.getLogger(__name__)


@dataclass
class VerificationStepResult:
    name: str
    passed: bool
    evidence: str
    issues: List[str]


class ChainOfVerification:
    """Runs sequential verification checks across agent outputs."""

    def __init__(self) -> None:
        self.steps: List[Callable[[Dict[str, Any]], VerificationStepResult]] = [
            self._check_research_grounding,
            self._check_clinical_alignment,
            self._check_marketing_compliance,
            self._check_self_consistency_support,
        ]

    def run(self, output: Dict[str, Any]) -> Dict[str, Any]:
        results: List[VerificationStepResult] = []
        aggregated_issues: List[str] = []

        for step in self.steps:
            result = step(output)
            results.append(result)
            aggregated_issues.extend(result.issues)
            logger.debug("Chain-of-Verification step '%s': %s", result.name, "passed" if result.passed else "failed")

        overall = all(result.passed for result in results)
        return {
            "overall_passed": overall,
            "steps": [result.__dict__ for result in results],
            "issues": aggregated_issues,
        }

    def _check_research_grounding(self, output: Dict[str, Any]) -> VerificationStepResult:
        research = output.get("research_content", {}) or {}
        statistics: List[str] = research.get("statistics", []) if isinstance(research, dict) else []
        findings: List[str] = research.get("findings", []) if isinstance(research, dict) else []

        if not statistics and not findings:
            return VerificationStepResult(
                name="Research grounding",
                passed=False,
                evidence="No research statistics or findings available for verification.",
                issues=["Research content missing; cannot validate grounding."],
            )

        total_items = len(statistics)
        grounded_items = sum(1 for item in statistics if "[Source:" in item and "According to" in item)
        coverage = grounded_items / total_items if total_items else 1.0

        passed = coverage >= 0.7 and bool(findings)
        issues: List[str] = []
        if coverage < 0.7:
            issues.append(f"Only {grounded_items}/{total_items} statistics include citation pattern.")
        if not findings:
            issues.append("Findings section empty; unable to cross-reference claims.")

        evidence = (
            f"Citation coverage {coverage:.0%} across {total_items} statistics." if total_items else "No statistics provided."
        )

        return VerificationStepResult(
            name="Research grounding",
            passed=passed,
            evidence=evidence,
            issues=issues,
        )

    def _check_clinical_alignment(self, output: Dict[str, Any]) -> VerificationStepResult:
        research = output.get("research_content", {}) or {}
        clinical = output.get("clinical_content", {}) or {}

        research_terms = self._extract_keywords(research.get("implications", []))
        clinical_terms = self._extract_keywords(clinical.get("interventions", []))

        overlap = research_terms & clinical_terms
        passed = bool(overlap)
        evidence = (
            f"Shared terminology: {', '.join(sorted(list(overlap))[:5])}" if overlap else "No shared terminology detected."
        )
        issues = [] if passed else ["Clinical interventions do not reference research implications by shared terminology."]

        return VerificationStepResult(
            name="Clinical alignment",
            passed=passed,
            evidence=evidence,
            issues=issues,
        )

    def _check_marketing_compliance(self, output: Dict[str, Any]) -> VerificationStepResult:
        marketing = output.get("marketing_content", {}) or {}

        banned_patterns = [
            r"\bguarantee\b",
            r"\bcure\b",
            r"\btestimonial\b",
            r"\breview\b",
        ]

        flagged: List[str] = []
        for field, values in marketing.items():
            if not isinstance(values, list):
                continue
            for value in values:
                lower_value = value.lower()
                if any(re.search(pattern, lower_value) for pattern in banned_patterns):
                    flagged.append(f"{field}: {value}")

        passed = not flagged
        evidence = "No compliance issues found." if passed else f"Flagged content: {flagged[:3]}"
        issues = flagged if flagged else []

        return VerificationStepResult(
            name="Marketing compliance",
            passed=passed,
            evidence=evidence,
            issues=issues,
        )

    def _check_self_consistency_support(self, output: Dict[str, Any]) -> VerificationStepResult:
        metadata = output.get("self_consistency", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}

        num_samples = metadata.get("num_samples", 0)
        vote_threshold = metadata.get("vote_threshold", 0)
        passed = num_samples >= 2 and vote_threshold >= 1

        issues = []
        if not passed:
            issues.append("Self-consistency metadata unavailable or insufficient.")

        evidence = (
            f"{num_samples} samples evaluated with threshold {vote_threshold}."
            if num_samples
            else "No self-consistency metadata."
        )

        return VerificationStepResult(
            name="Self-consistency evidence",
            passed=passed,
            evidence=evidence,
            issues=issues,
        )

    @staticmethod
    def _extract_keywords(values: List[str]) -> set[str]:
        keywords: set[str] = set()
        for value in values or []:
            for token in re.findall(r"[a-zA-Z]{4,}", value or ""):
                keywords.add(token.lower())
        return keywords
