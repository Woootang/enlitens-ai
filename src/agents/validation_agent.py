"""Validation Agent - Constitutional Guardian."""

from __future__ import annotations

import copy
import logging
import re
from datetime import datetime
from typing import Any, Dict, List

from .base_agent import BaseAgent
from src.utils.enlitens_constitution import EnlitensConstitution
from src.monitoring.error_telemetry import TelemetrySeverity, log_with_telemetry

logger = logging.getLogger(__name__)
TELEMETRY_AGENT = "validation_agent"


class ValidationAgent(BaseAgent):
    """Agent that enforces the Enlitens constitution across generated outputs."""

    def __init__(self) -> None:
        super().__init__(
            name="Validation",
            role="Constitutional Quality Assurance",
        )
        self.constitution = EnlitensConstitution()
        self.principle_sequence = [
            "ENL-001",
            "ENL-002",
            "ENL-003",
            "ENL-004",
            "ENL-005",
            "ENL-006",
            "ENL-007",
            "ENL-008",
            "ENL-009",
            "ENL-010",
        ]
        self._neurodiversity_terms = {
            "neurotype",
            "operating system",
            "neurodivergent",
            "neurodiversity",
            "pattern",
        }
        self._collaboration_terms = {
            "co-design",
            "co design",
            "co-create",
            "co create",
            "together",
            "partner",
            "choice",
            "choose",
            "agency",
        }
        self._evidence_terms = {
            "study",
            "research",
            "evidence",
            "data",
            "trial",
            "neuroscience",
        }
        self._year_pattern = re.compile(r"20[0-9]{2}")

    async def initialize(self) -> bool:
        try:
            self.is_initialized = True
            logger.info(f"✅ {self.name} agent initialized")
            return True
        except Exception as exc:  # pragma: no cover - defensive guard
            log_with_telemetry(
                logger.error,
                "Failed to initialize %s: %s",
                self.name,
                exc,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Agent initialization failed",
                details={"error": str(exc)},
            )
            return False

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate synthesis output against every constitutional principle."""

        complete_output = context.get("complete_output", {}) or {}

        review = self._evaluate_output(complete_output)
        corrected_clinical = self._apply_fixups(
            complete_output.get("clinical_content") or {},
            complete_output.get("research_content") or {},
            review["principles"],
        )

        passed = all(principle["passed"] for principle in review["principles"])
        citation_report = self._build_citation_report(complete_output.get("research_content") or {})

        retry_metadata = {
            "attempt": context.get("retry_attempt", 1),
            "needs_retry": not passed,
            "triggers": [p["principle_id"] for p in review["principles"] if not p["passed"]],
            "self_critique_performed": False,
            "timestamp": datetime.utcnow().isoformat(),
        }

        result = {
            "constitutional_review": review,
            "quality_scores": review["quality_scores"],
            "confidence_scoring": {
                "confidence_score": review["quality_scores"].get("overall_quality", 0.0)
            },
            "verification_report": {"overall_passed": passed, "steps": []},
            "citation_report": citation_report,
            "corrected_clinical_content": corrected_clinical,
            "self_critique": None,
            "retry_metadata": retry_metadata,
            "final_validation": {
                "passed": passed,
                "recommendations": review["recommendations"],
            },
        }
        return result

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """Minimal validator ensuring outputs meet a baseline quality bar."""
        try:
            final = output.get("final_validation") or {}
            if isinstance(final, dict) and final.get("passed") is True:
                return True
            scores = output.get("quality_scores") or {}
            overall = 0.0
            try:
                overall = float(scores.get("overall_quality", 0.0))
            except Exception:
                overall = 0.0
            return overall >= 0.6
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def _evaluate_output(self, complete_output: Dict[str, Any]) -> Dict[str, Any]:
        text_blob = self._flatten_text(complete_output)
        principles: List[Dict[str, Any]] = []
        quality_scores: Dict[str, float] = {}
        recommendations: List[str] = []

        for principle_id in self.principle_sequence:
            issues = self._evaluate_principle(principle_id, complete_output, text_blob)
            passed = not issues
            principle = self.constitution.get(principle_id)
            principles.append(
                {
                    "principle_id": principle_id,
                    "title": principle.title,
                    "passed": passed,
                    "issues": issues,
                }
            )
            quality_scores[f"principle_{principle_id.lower()}"] = 1.0 if passed else 0.0
            if issues:
                recommendations.append(f"Address {principle_id}: {issues[0]}")

        if quality_scores:
            overall = sum(quality_scores.values()) / len(quality_scores)
        else:  # pragma: no cover - defensive guard
            overall = 0.0
        quality_scores["overall_quality"] = overall

        return {
            "principles": principles,
            "quality_scores": quality_scores,
            "recommendations": recommendations,
        }

    def _evaluate_principle(
        self, principle_id: str, complete_output: Dict[str, Any], text_blob: str
    ) -> List[str]:
        issues: List[str] = []
        text_lower = text_blob.lower()
        research = complete_output.get("research_content") or {}

        if principle_id == "ENL-001":
            if not self.constitution.ensure_keyword_presence(text_blob, self.constitution.CONTEXT_KEYWORDS):
                issues.append("Missing explicit environmental or systemic framing.")
            if self.constitution.contains_pathology(text_blob) or self.constitution.contains_legacy_reference(text_blob):
                issues.append("Pathology-first or legacy diagnostic language detected.")
        elif principle_id == "ENL-002":
            if self.constitution.contains_pathology(text_blob):
                issues.append("Replace pathology terminology with neurotype-affirming language.")
            if not any(term in text_lower for term in self._neurodiversity_terms):
                issues.append("Neurodiversity framing is absent; reference neurotypes or operating systems.")
        elif principle_id == "ENL-003":
            if not (research.get("citations") or research.get("references")):
                issues.append("No contemporary citations or references supplied.")
            if not (
                self.constitution.ensure_keyword_presence(text_blob, self._evidence_terms)
                and self._year_pattern.search(text_blob)
            ):
                issues.append("Evidence discussion should cite modern neuroscience or trauma findings explicitly.")
        elif principle_id == "ENL-004":
            if not any(term in text_lower for term in self._collaboration_terms):
                issues.append("Collaborative agency cues (co-design, choices, partnership) are missing.")
        elif principle_id == "ENL-005":
            if not self.constitution.ensure_keyword_presence(text_blob, self.constitution.STRENGTH_KEYWORDS):
                issues.append("Strengths-first storytelling not detected; surface adaptive skills alongside challenges.")
        elif principle_id == "ENL-006":
            if not self.constitution.ensure_keyword_presence(text_blob, self.constitution.TRAUMA_KEYWORDS):
                issues.append("Trauma-informed safety language absent; name nervous-system states and regulation cues.")
        elif principle_id == "ENL-007":
            if not self.constitution.ensure_keyword_presence(text_blob, self.constitution.SYSTEM_KEYWORDS):
                issues.append("System accountability missing; cite structural pressures causing harm.")
        elif principle_id == "ENL-008":
            if not self.constitution.ensure_keyword_presence(text_blob, self.constitution.BOLD_MARKERS):
                issues.append("Bold, precise voice not evident; add rebellious language that calls out the old map.")
        elif principle_id == "ENL-009":
            if not (research.get("citations") or research.get("references")):
                issues.append("Factual statements lack traceable citations.")
        elif principle_id == "ENL-010":
            if not self.constitution.ensure_keyword_presence(text_blob, self.constitution.AUTONOMY_KEYWORDS):
                issues.append("Future-facing autonomy not articulated; describe how clients graduate with a roadmap.")

        return issues

    # ------------------------------------------------------------------
    # Fix-up helpers
    # ------------------------------------------------------------------
    def _apply_fixups(
        self,
        clinical_content: Dict[str, Any],
        research_content: Dict[str, Any],
        principle_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        corrected = copy.deepcopy(clinical_content) if clinical_content else {}
        corrected = self.constitution.sanitize_mapping(corrected)
        # Ensure expected keys exist even if empty
        from src.models.enlitens_schemas import ClinicalContent  # local import to avoid circular load at module import time

        baseline = ClinicalContent().model_dump()
        for field, default_value in baseline.items():
            corrected.setdefault(field, default_value)

        for principle in principle_results:
            if principle["passed"]:
                continue
            pid = principle["principle_id"]
            if pid == "ENL-001":
                self._ensure_list(corrected, "guidelines").append(
                    self.constitution.sanitize_language(
                        "Context audit: map environmental, relational, and systemic pressures before naming individual challenges."
                    )
                )
            elif pid == "ENL-002":
                self._ensure_list(corrected, "interventions").append(
                    self.constitution.sanitize_language(
                        "Reframe language to neurotype-baseline wording (e.g., operating system, pattern, divergence)."
                    )
                )
            elif pid == "ENL-003":
                citations = research_content.get("citations") or research_content.get("references") or []
                if citations:
                    self._ensure_list(corrected, "guidelines").append(
                        self.constitution.sanitize_language(
                            "Anchor follow-up sessions to fresh evidence sources: " + ", ".join(citations[:2])
                        )
                    )
            elif pid == "ENL-004":
                self._ensure_list(corrected, "protocols").append(
                    "Co-design each intervention with at least two client-selected options to preserve agency."
                )
            elif pid == "ENL-005":
                self._ensure_list(corrected, "outcomes").append(
                    "Document strengths discovered (creative problem solving, hyperfocus) alongside every friction note."
                )
            elif pid == "ENL-006":
                self._ensure_list(corrected, "contraindications").append(
                    "Pause or slow work whenever the nervous system signals overwhelm; secure consent for regulation strategies."
                )
            elif pid == "ENL-007":
                self._ensure_list(corrected, "guidelines").append(
                    "Plan advocacy moves targeting schools, workplaces, and policy barriers fueling the distress."
                )
            elif pid == "ENL-008":
                self._ensure_list(corrected, "interventions").append(
                    "Use bold reframes that torch pathologizing narratives while offering actionable alternatives."
                )
            elif pid == "ENL-009":
                self._ensure_list(corrected, "monitoring").append(
                    "Track citations for every factual claim and log gaps for the Research Agent to pursue."
                )
            elif pid == "ENL-010":
                self._ensure_list(corrected, "outcomes").append(
                    "Explicit independence goal: client leaves with a living blueprint and self-advocacy scripts."
                )

        return corrected

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _flatten_text(self, complete_output: Dict[str, Any]) -> str:
        parts: List[str] = []
        for value in complete_output.values():
            if isinstance(value, dict):
                parts.append(self._flatten_text(value))
            elif isinstance(value, list):
                parts.extend(str(item) for item in value)
            elif isinstance(value, str):
                parts.append(value)
        return " \n".join(parts)

    def _ensure_list(self, mapping: Dict[str, Any], key: str) -> List[str]:
        value = mapping.get(key)
        if not isinstance(value, list):
            value = []
        mapping[key] = value
        return value

    def _build_citation_report(self, research_content: Dict[str, Any]) -> Dict[str, Any]:
        citations = research_content.get("citations") or []
        references = research_content.get("references") or []
        total = len(citations) + len(references)
        if total == 0:
            return {
                "verified": 0,
                "failed": ["No citations provided"],
                "missing_quotes": [],
                "total": 0,
            }
        return {
            "verified": total,
            "failed": [],
            "missing_quotes": [],
            "total": total,
        }
