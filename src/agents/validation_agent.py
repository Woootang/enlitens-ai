"""Validation Agent - Validates and scores all generated content."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent
from src.synthesis.ollama_client import OllamaClient
from src.validation.chain_of_verification import ChainOfVerification

logger = logging.getLogger(__name__)


class ValidationAgent(BaseAgent):
    """Agent specialized in content validation and quality scoring."""

    def __init__(self):
        super().__init__(
            name="Validation",
            role="Content Validation and Quality Assurance",
        )
        self.chain_of_verification = ChainOfVerification()
        self.self_critique_thresholds = {
            "overall_quality": 0.75,
        }
        self._llm_client = OllamaClient()

    async def initialize(self) -> bool:
        """Initialize the validation agent."""
        try:
            self.is_initialized = True
            logger.info(f"✅ {self.name} agent initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            return False

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all generated content."""
        try:
            complete_output = context.get("complete_output", {})

            # Calculate quality scores
            quality_scores = self._calculate_quality_scores(complete_output)
            verification_report = self.chain_of_verification.run(complete_output)
            quality_scores["verification_chain"] = 1.0 if verification_report["overall_passed"] else 0.0
            quality_scores["overall_quality"] = self._compute_overall_quality(quality_scores)
            confidence_scoring = self._calculate_confidence_scores(complete_output)

            document_text = context.get("document_text") or complete_output.get("full_document_text", "")
            citation_report = self._verify_citations(complete_output, document_text)

            self_critique: Optional[Dict[str, Any]] = None
            if self._needs_self_critique(quality_scores, verification_report, citation_report):
                self_critique = await self._run_self_critique(
                    complete_output,
                    quality_scores,
                    verification_report,
                    citation_report,
                )

            validation_passed = (
                quality_scores.get("overall_quality", 0) >= 0.6
                and verification_report["overall_passed"]
                and not citation_report["failed"]
            )

            retry_metadata = self._build_retry_metadata(
                context,
                quality_scores,
                verification_report,
                citation_report,
                bool(self_critique),
                validation_passed,
            )

            result = {
                "quality_scores": quality_scores,
                "confidence_scoring": confidence_scoring,
                "verification_report": verification_report,
                "citation_report": citation_report,
                "self_critique": self_critique,
                "retry_metadata": retry_metadata,
                "final_validation": {
                    "passed": validation_passed,
                    "recommendations": self._generate_recommendations(quality_scores)
                }
            }

            self._log_quality_event(
                context,
                quality_scores,
                confidence_scoring,
                verification_report,
                citation_report,
                retry_metadata,
                self_critique,
            )

            return result

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "quality_scores": {"overall_quality": 0},
                "confidence_scoring": {"confidence_score": 0},
                "verification_report": {"overall_passed": False, "issues": []},
                "citation_report": {"verified": 0, "failed": [], "missing_quotes": [], "total": 0},
                "retry_metadata": {
                    "attempt": context.get("retry_attempt", 1),
                    "needs_retry": True,
                    "triggers": ["exception"],
                    "self_critique_performed": False,
                    "timestamp": datetime.utcnow().isoformat(),
                },
                "final_validation": {"passed": False, "recommendations": []},
            }

    def _calculate_quality_scores(self, output: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality scores for different content types."""
        scores: Dict[str, float] = {}

        # Check research content
        research = self._ensure_mapping(output.get("research_content"))
        scores["research_quality"] = self._score_content(research, ["findings", "methodologies"])

        # Check clinical content
        clinical = self._ensure_mapping(output.get("clinical_content"))
        scores["clinical_accuracy"] = self._score_content(clinical, ["interventions", "protocols"])

        # Check marketing content
        marketing = self._ensure_mapping(output.get("marketing_content"))
        scores["marketing_effectiveness"] = self._score_content(marketing, ["headlines", "value_propositions"])

        # Check SEO content coverage
        seo = self._ensure_mapping(output.get("seo_content"))
        scores["seo_readiness"] = self._score_content(seo, ["meta_descriptions", "primary_keywords"])

        # Check founder voice (if available)
        scores["founder_voice_authenticity"] = 0.8  # Default score

        # Check completeness
        scores["completeness"] = self._score_completeness(output)

        # Check fact checking
        scores["fact_checking"] = 0.85  # Default score

        return scores

    def _compute_overall_quality(self, scores: Dict[str, float]) -> float:
        metrics = [value for key, value in scores.items() if key != "overall_quality"]
        return sum(metrics) / len(metrics) if metrics else 0.0

    def _needs_self_critique(
        self,
        scores: Dict[str, float],
        verification_report: Dict[str, Any],
        citation_report: Dict[str, Any],
    ) -> bool:
        if scores.get("overall_quality", 0.0) < self.self_critique_thresholds["overall_quality"]:
            return True
        if not verification_report.get("overall_passed", False):
            return True
        if citation_report.get("failed"):
            return True
        return False

    async def _run_self_critique(
        self,
        output: Dict[str, Any],
        scores: Dict[str, float],
        verification_report: Dict[str, Any],
        citation_report: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Request additional evidence chains from the LLM when thresholds are not met."""

        prompt = (
            "You are the validation watchdog for a multi-agent content pipeline. "
            "Review the provided content summary, quality metrics, and citation findings. "
            "Identify why validation is struggling and propose concrete evidence chains (with specific sections to re-read) "
            "that another pass of the generators should pursue. Respond with concise JSON containing 'issues', 'evidence_chains', "
            "and 'next_actions'."
        )

        request_payload = {
            "content_summary": {k: v for k, v in output.items() if k != "full_document_text"},
            "quality_scores": scores,
            "verification": verification_report,
            "citation_report": citation_report,
        }

        try:
            response = await self._llm_client.generate_text(
                prompt + "\n\nINPUT:\n" + json.dumps(request_payload, default=str) + "\n\nJSON:",
                temperature=0.2,
            )
            parsed = self._parse_json_response(response)
            if parsed:
                parsed.setdefault("generated_at", datetime.utcnow().isoformat())
            return parsed
        except Exception as exc:
            logger.warning("Self-critique request failed: %s", exc)
            return None

    @staticmethod
    def _parse_json_response(response: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                start = response.index("{")
                end = response.rindex("}") + 1
                return json.loads(response[start:end])
            except Exception:
                return None

    def _verify_citations(self, output: Dict[str, Any], source_text: str) -> Dict[str, Any]:
        """Fuzzy match citation quotes against the archived document text."""
        blog = self._ensure_mapping(output.get("blog_content"))
        statistics: List[Dict[str, Any]] = blog.get("statistics", []) if isinstance(blog, dict) else []

        verified = 0
        failures: List[Dict[str, Any]] = []
        missing_quotes: List[str] = []

        if not isinstance(statistics, list):
            return {"verified": 0, "failed": [], "missing_quotes": [], "total": 0}

        for stat in statistics:
            citation = stat.get("citation") if isinstance(stat, dict) else None
            if not citation:
                missing_quotes.append(stat.get("claim", ""))
                continue
            quote = citation.get("quote", "")
            if not quote:
                missing_quotes.append(stat.get("claim", ""))
                continue

            if quote and quote in source_text:
                verified += 1
                continue

            if quote and source_text:
                matcher = SequenceMatcher(None, quote.lower(), source_text.lower())
                ratio = matcher.quick_ratio()
                if ratio >= 0.8:
                    verified += 1
                    continue
                failures.append({
                    "claim": stat.get("claim"),
                    "quote": quote,
                    "similarity": round(ratio, 3),
                })
            else:
                failures.append({
                    "claim": stat.get("claim"),
                    "quote": quote,
                    "similarity": 0,
                })

        return {
            "verified": verified,
            "failed": failures,
            "missing_quotes": missing_quotes,
            "total": len(statistics),
        }

    def _build_retry_metadata(
        self,
        context: Dict[str, Any],
        quality_scores: Dict[str, float],
        verification_report: Dict[str, Any],
        citation_report: Dict[str, Any],
        self_critique_performed: bool,
        validation_passed: bool,
    ) -> Dict[str, Any]:
        triggers: List[str] = []
        if quality_scores.get("overall_quality", 0.0) < self.self_critique_thresholds["overall_quality"]:
            triggers.append("low_quality")
        if not verification_report.get("overall_passed", False):
            triggers.append("verification_failed")
        if citation_report.get("failed"):
            triggers.append("citation_mismatch")
        if citation_report.get("missing_quotes"):
            triggers.append("missing_quotes")

        attempt = int(context.get("retry_attempt", 1) or 1)
        needs_retry = bool(triggers) or not validation_passed

        return {
            "attempt": attempt,
            "needs_retry": needs_retry,
            "triggers": triggers,
            "self_critique_performed": self_critique_performed,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _log_quality_event(
        self,
        context: Dict[str, Any],
        quality_scores: Dict[str, float],
        confidence: Dict[str, float],
        verification_report: Dict[str, Any],
        citation_report: Dict[str, Any],
        retry_metadata: Dict[str, Any],
        self_critique: Optional[Dict[str, Any]],
    ) -> None:
        metrics_event = {
            "document_id": context.get("document_id"),
            "timestamp": datetime.utcnow().isoformat(),
            "quality": quality_scores.get("overall_quality", 0.0),
            "quality_scores": quality_scores,
            "confidence": confidence.get("confidence_score", 0.0),
            "verification_passed": verification_report.get("overall_passed", False),
            "retry_attempt": retry_metadata.get("attempt", 1),
            "needs_retry": retry_metadata.get("needs_retry", False),
            "failure_reasons": retry_metadata.get("triggers", []),
            "citation_failures": [failure.get("claim") for failure in citation_report.get("failed", [])],
            "missing_quotes": citation_report.get("missing_quotes", []),
            "self_critique_performed": bool(self_critique),
        }

        metrics_logger = logging.getLogger("validation.metrics")
        metrics_logger.info("QUALITY_METRICS %s", json.dumps(metrics_event))

    def _score_content(self, content: Dict[str, Any], required_fields: list) -> float:
        """Score content based on required fields."""
        if not content:
            return 0.0

        filled_fields = sum(1 for field in required_fields if content.get(field))
        return filled_fields / len(required_fields) if required_fields else 0.0

    def _score_completeness(self, output: Dict[str, Any]) -> float:
        """Score overall completeness of output."""
        required_sections = [
            "research_content",
            "clinical_content",
            "marketing_content",
            "seo_content",
        ]

        filled_sections = sum(1 for section in required_sections if output.get(section))
        return filled_sections / len(required_sections)

    @staticmethod
    def _ensure_mapping(content: Any) -> Dict[str, Any]:
        if hasattr(content, "model_dump"):
            try:
                return content.model_dump()
            except Exception:
                pass
        if isinstance(content, dict):
            return content
        return content or {}

    def _calculate_confidence_scores(self, output: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores."""
        return {
            "confidence_score": 0.75,  # Default confidence
            "reliability": 0.8,
            "consistency": 0.85
        }

    def _generate_recommendations(self, quality_scores: Dict[str, float]) -> list:
        """Generate recommendations based on quality scores."""
        recommendations = []

        if quality_scores.get("research_quality", 1) < 0.7:
            recommendations.append("Enhance research content extraction")

        if quality_scores.get("clinical_accuracy", 1) < 0.7:
            recommendations.append("Improve clinical synthesis")

        if quality_scores.get("marketing_effectiveness", 1) < 0.7:
            recommendations.append("Strengthen marketing content")

        if quality_scores.get("seo_readiness", 1) < 0.7:
            recommendations.append("Expand SEO coverage")

        if quality_scores.get("fact_checking", 1) < 0.7:
            recommendations.append("Increase fact-checking rigor")

        return recommendations

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate the validation output."""
        required_keys = {
            "quality_scores",
            "confidence_scoring",
            "verification_report",
            "citation_report",
            "retry_metadata",
        }
        return required_keys <= set(output.keys())

    async def cleanup(self):
        """Clean up resources."""
        logger.info(f"Cleaning up {self.name} agent")
