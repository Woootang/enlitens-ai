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
from tools.web.scrape_url import ScrapeUrlRequest, scrape_url
from tools.web.web_search_ddg import WebSearchRequest, ddg_text_search

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
        self.minimum_field_requirements = {
            "educational_content": {
                "explanations": {"min": 5, "critical": True},
                "examples": {"min": 5, "critical": False},
                "analogies": {"min": 5, "critical": False},
                "definitions": {"min": 5, "critical": False},
                "processes": {"min": 5, "critical": False},
                "comparisons": {"min": 5, "critical": False},
                "visual_aids": {"min": 4, "critical": False},
                "learning_objectives": {"min": 5, "critical": True},
            },
            "marketing_content": {
                "headlines": {"min": 3, "critical": True},
                "taglines": {"min": 3, "critical": False},
                "value_propositions": {"min": 3, "critical": False},
                "benefits": {"min": 3, "critical": False},
                "pain_points": {"min": 3, "critical": False},
            },
            "seo_content": {
                "primary_keywords": {"min": 5, "critical": True},
                "secondary_keywords": {"min": 5, "critical": False},
                "long_tail_keywords": {"min": 4, "critical": False},
                "meta_descriptions": {"min": 3, "critical": True},
                "title_tags": {"min": 3, "critical": False},
                "content_topics": {"min": 4, "critical": False},
            },
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

            quality_scores = self._calculate_quality_scores(complete_output)
            verification_report = self.chain_of_verification.run(complete_output)
            document_text = context.get("document_text") or complete_output.get("full_document_text", "")
            citation_report = self._verify_citations(complete_output, document_text)

            quality_scores["fact_checking"] = self._score_fact_checking(citation_report)
            quality_scores["verification_chain"] = 1.0 if verification_report["overall_passed"] else 0.0
            quality_scores["overall_quality"] = self._compute_overall_quality(quality_scores)
            confidence_scoring = self._calculate_confidence_scores(complete_output)
            warnings = self._generate_warnings(quality_scores, verification_report, citation_report)
            warnings.extend(self._detect_content_gaps(complete_output))
            review_checklist = self._build_review_checklist(warnings)

            self_critique: Optional[Dict[str, Any]] = None
            if self._needs_self_critique(quality_scores, verification_report, citation_report):
                self_critique = await self._run_self_critique(
                    complete_output,
                    quality_scores,
                    verification_report,
                    citation_report,
                )

            critical_warnings = any(warning["severity"] == "critical" for warning in warnings)
            validation_passed = (
                quality_scores.get("overall_quality", 0) >= 0.65
                and verification_report["overall_passed"]
                and not citation_report["failed"]
                and not critical_warnings
            )

            retry_metadata = self._build_retry_metadata(
                context,
                quality_scores,
                verification_report,
                citation_report,
                bool(self_critique),
                validation_passed,
                warnings,
            )

            result = {
                "quality_scores": quality_scores,
                "confidence_scoring": confidence_scoring,
                "verification_report": verification_report,
                "citation_report": citation_report,
                "self_critique": self_critique,
                "retry_metadata": retry_metadata,
                "warnings": warnings,
                "review_checklist": review_checklist,
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
                warnings,
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
                "warnings": [
                    {"section": "system", "message": str(e), "severity": "critical"}
                ],
                "review_checklist": [{"item": "Review pipeline exception details.", "severity": "critical"}],
                "final_validation": {"passed": False, "recommendations": []},
            }

    def _calculate_quality_scores(self, output: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality scores for different content types."""
        scores: Dict[str, float] = {}

        educational = self._ensure_mapping(output.get("educational_content"))
        scores["educational_coverage"] = self._score_content(
            educational,
            [
                "explanations",
                "examples",
                "analogies",
                "definitions",
                "processes",
                "comparisons",
                "visual_aids",
                "learning_objectives",
            ],
        )
        scores["educational_depth"] = self._score_density(
            educational,
            [
                "explanations",
                "examples",
                "analogies",
                "processes",
                "comparisons",
                "learning_objectives",
            ],
        )
        scores["aha_alignment"] = self._score_prediction_error_alignment(educational)
        scores["educational_minimums"] = self._score_minimum_viable(
            educational,
            self.minimum_field_requirements.get("educational_content", {}),
        )

        # Check research content
        research = self._ensure_mapping(output.get("research_content"))
        scores["research_quality"] = self._score_content(research, ["findings", "methodologies"])

        # Check clinical content
        clinical = self._ensure_mapping(output.get("clinical_content"))
        scores["clinical_accuracy"] = self._score_content(clinical, ["interventions", "protocols"])

        # Check marketing content
        marketing = self._ensure_mapping(output.get("marketing_content"))
        scores["marketing_effectiveness"] = self._score_content(marketing, ["headlines", "value_propositions"])
        scores["marketing_minimums"] = self._score_minimum_viable(
            marketing,
            self.minimum_field_requirements.get("marketing_content", {}),
        )

        # Check SEO content coverage
        seo = self._ensure_mapping(output.get("seo_content"))
        scores["seo_readiness"] = self._score_content(seo, ["meta_descriptions", "primary_keywords"])
        scores["seo_minimums"] = self._score_minimum_viable(
            seo,
            self.minimum_field_requirements.get("seo_content", {}),
        )

        # Check founder voice (if available)
        scores["founder_voice_authenticity"] = self._score_voice_alignment(output)

        rebellion = self._ensure_mapping(output.get("rebellion_framework") or output.get("rebellion_result"))
        if rebellion:
            scores["rebellion_alignment"] = self._score_content(
                rebellion,
                [
                    "narrative_deconstruction",
                    "sensory_profiling",
                    "executive_function",
                    "social_processing",
                    "strengths_synthesis",
                    "rebellion_themes",
                    "aha_moments",
                ],
            )

        # Check completeness
        scores["completeness"] = self._score_completeness(output)

        # Localization and persona alignment heuristics
        scores["localization"] = self._score_local_relevance(output)

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
        """Match citations against source text and optionally live web sources."""
        blog = self._ensure_mapping(output.get("blog_content"))
        statistics: List[Dict[str, Any]] = blog.get("statistics", []) if isinstance(blog, dict) else []

        if not isinstance(statistics, list):
            return {"verified": 0, "failed": [], "missing_quotes": [], "total": 0}

        verified = 0
        missing_quotes: List[str] = []
        unresolved: List[Dict[str, Any]] = []

        for stat in statistics:
            citation = stat.get("citation") if isinstance(stat, dict) else None
            if not citation:
                missing_quotes.append(stat.get("claim", ""))
                continue
            quote = citation.get("quote", "") or ""
            if not quote.strip():
                missing_quotes.append(stat.get("claim", ""))
                continue

            claim = stat.get("claim")
            if quote and quote in source_text:
                verified += 1
                continue

            if quote and source_text:
                matcher = SequenceMatcher(None, quote.lower(), source_text.lower())
                if matcher.quick_ratio() >= 0.8:
                    verified += 1
                    continue

            unresolved.append({"claim": claim, "quote": quote})

        resolved = self._check_quotes_on_web(unresolved)
        verified += len(resolved)

        failures = [item for item in unresolved if item not in resolved]

        return {
            "verified": verified,
            "failed": failures,
            "missing_quotes": missing_quotes,
            "total": len(statistics),
        }


    def _check_quotes_on_web(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        verified: List[Dict[str, Any]] = []
        for candidate in candidates[:3]:
            quote = (candidate.get("quote") or "").strip()
            if len(quote) < 25:
                continue
            query = f'"{quote[:120]}"'
            try:
                results = ddg_text_search(WebSearchRequest(query=query, max_results=2))
            except Exception as exc:
                logger.debug("Web search failed for citation quote: %s", exc)
                continue

            for res in results:
                try:
                    scraped = scrape_url(ScrapeUrlRequest(url=res.url))
                except Exception as scrape_exc:
                    logger.debug("Scrape failed for %s: %s", res.url, scrape_exc)
                    continue
                if not scraped:
                    continue
                body = scraped.text.lower()
                if quote.lower() in body:
                    verified.append(candidate)
                    break
        return verified

    def _build_retry_metadata(
        self,
        context: Dict[str, Any],
        quality_scores: Dict[str, float],
        verification_report: Dict[str, Any],
        citation_report: Dict[str, Any],
        self_critique_performed: bool,
        validation_passed: bool,
        warnings: List[Dict[str, Any]],
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
        critical_flags = [w for w in warnings if w.get("severity") == "critical"]
        if critical_flags:
            triggers.extend({f"critical:{w.get('section')}" for w in critical_flags})

        attempt = int(context.get("retry_attempt", 1) or 1)
        needs_retry = bool(critical_flags) or not validation_passed

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
        warnings: List[Dict[str, Any]],
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
            "warnings": warnings,
            "review_checklist": self._build_review_checklist(warnings),
            "compliance_message": "Educational content only. This is not medical advice.",
        }

        metrics_logger = logging.getLogger("validation.metrics")
        metrics_logger.info("QUALITY_METRICS %s", json.dumps(metrics_event))

    def _score_content(self, content: Dict[str, Any], required_fields: list) -> float:
        """Score content based on required fields."""
        if not content:
            return 0.0

        filled_fields = sum(1 for field in required_fields if content.get(field))
        return filled_fields / len(required_fields) if required_fields else 0.0

    def _score_minimum_viable(
        self,
        content: Dict[str, Any],
        requirements: Dict[str, Dict[str, Any]],
    ) -> float:
        if not content or not requirements:
            return 0.0
        hits = 0
        total = len(requirements)
        for field, metadata in requirements.items():
            min_items = metadata.get("min", 1)
            values = content.get(field)
            if isinstance(values, list):
                filled = len([item for item in values if item])
            elif values:
                filled = 1
            else:
                filled = 0
            if filled >= min_items:
                hits += 1
        return hits / total if total else 0.0

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

        if quality_scores.get("educational_coverage", 1) < 0.7:
            recommendations.append("Regenerate educational content with richer coverage.")

        if quality_scores.get("educational_depth", 1) < 0.6:
            recommendations.append("Add more detailed, example-rich educational explainer items.")

        if quality_scores.get("aha_alignment", 1) < 0.5:
            recommendations.append("Dial up prediction-error framing and explicit 'aha' statements.")

        if quality_scores.get("educational_minimums", 1) < 0.75:
            recommendations.append("Each educational list needs at least five populated entries.")

        if quality_scores.get("research_quality", 1) < 0.7:
            recommendations.append("Enhance research content extraction")

        if quality_scores.get("clinical_accuracy", 1) < 0.7:
            recommendations.append("Improve clinical synthesis")

        if quality_scores.get("marketing_effectiveness", 1) < 0.7:
            recommendations.append("Strengthen marketing content")

        if quality_scores.get("marketing_minimums", 1) < 0.8:
            recommendations.append("Provide at least three headlines/taglines/value props per marketing batch.")

        if quality_scores.get("seo_readiness", 1) < 0.7:
            recommendations.append("Expand SEO coverage")

        if quality_scores.get("seo_minimums", 1) < 0.8:
            recommendations.append("Fill SEO keyword/meta lists to minimum viable counts.")

        if quality_scores.get("localization", 1) < 0.6:
            recommendations.append("Inject St. Louis-specific language and data into creative outputs.")

        if quality_scores.get("founder_voice_authenticity", 1) < 0.6:
            recommendations.append("Reinforce Liz's voice guardrails and key phrases.")

        if quality_scores.get("fact_checking", 1) < 0.7:
            recommendations.append("Increase fact-checking rigor")

        return recommendations

    def _score_density(self, content: Dict[str, Any], fields: List[str]) -> float:
        lengths: List[int] = []
        for field in fields:
            values = content.get(field) or []
            if isinstance(values, list):
                lengths.extend(len(str(item)) for item in values if item)
        if not lengths:
            return 0.0
        average = sum(lengths) / len(lengths)
        return min(1.0, average / 180)  # 180 chars ≈ 3 sentences

    def _score_prediction_error_alignment(self, educational: Dict[str, Any]) -> float:
        aha_keywords = [
            "aha",
            "surprise",
            "unexpected",
            "did you know",
            "counterintuitive",
            "shift",
            "reframe",
            "pattern",
            "insight",
            "prediction",
        ]
        total = 0
        hits = 0
        for field in (
            "explanations",
            "examples",
            "analogies",
            "comparisons",
            "learning_objectives",
            "aha_moments",
        ):
            values = educational.get(field) or []
            if isinstance(values, list):
                for item in values:
                    total += 1
                    text = str(item).lower()
                    if any(keyword in text for keyword in aha_keywords):
                        hits += 1
        return hits / total if total else 0.0

    def _score_voice_alignment(self, output: Dict[str, Any]) -> float:
        key_phrases = [phrase.lower() for phrase in getattr(self, "founder_phrases", [])]
        # fallback to hard-coded phrases if not set externally
        if not key_phrases:
            key_phrases = [
                "your brain isn't broken",
                "traditional therapy missed",
                "neuroscience shows",
                "real therapy for real people",
                "brain-based",
                "fuck",
            ]

        voice_sections = []
        for section in (
            "marketing_content",
            "blog_content",
            "website_copy",
            "social_media_content",
        ):
            content = self._ensure_mapping(output.get(section))
            for value in content.values():
                if isinstance(value, list):
                    voice_sections.extend(str(item).lower() for item in value if item)
        if not voice_sections:
            return 0.0
        hits = sum(
            1
            for snippet in voice_sections
            if any(phrase in snippet for phrase in key_phrases)
        )
        return min(1.0, hits / len(voice_sections))

    def _score_local_relevance(self, output: Dict[str, Any]) -> float:
        locale_keywords = [
            "st. louis",
            "saint louis",
            "st louis",
            "missouri",
            "314",
            "north city",
            "south county",
            "the loop",
            "webster groves",
            "kirkwood",
        ]
        strings: List[str] = []
        for section in (
            "marketing_content",
            "seo_content",
            "blog_content",
            "social_media_content",
            "content_creation_ideas",
        ):
            content = self._ensure_mapping(output.get(section))
            for value in content.values():
                if isinstance(value, list):
                    strings.extend(str(item).lower() for item in value if item)
        if not strings:
            return 0.0
        hits = sum(
            1 for snippet in strings if any(keyword in snippet for keyword in locale_keywords)
        )
        return min(1.0, hits / len(strings))

    @staticmethod
    def _score_fact_checking(citation_report: Dict[str, Any]) -> float:
        total = citation_report.get("total") or 0
        failed = len(citation_report.get("failed") or [])
        missing = len(citation_report.get("missing_quotes") or [])
        if total == 0:
            return 0.75  # neutral default when no stats provided
        successful = total - failed - missing
        return max(0.0, successful / total)

    def _generate_warnings(
        self,
        quality_scores: Dict[str, float],
        verification_report: Dict[str, Any],
        citation_report: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        warnings: List[Dict[str, Any]] = []

        def add(section: str, message: str, severity: str) -> None:
            warnings.append({"section": section, "message": message, "severity": severity})

        edu_cov = quality_scores.get("educational_coverage", 0.0)
        if edu_cov < 0.6:
            add("educational_content", "Educational payload missing multiple sections.", "critical")
        elif edu_cov < 0.8:
            add("educational_content", "Educational coverage is partial; consider a regen.", "warning")

        if quality_scores.get("educational_depth", 0.0) < 0.6:
            add("educational_content", "Educational items read shallow—expand explanations/examples.", "warning")

        if quality_scores.get("aha_alignment", 0.0) < 0.4:
            add("educational_content", "Few entries trigger prediction-error 'aha' framing.", "warning")

        if quality_scores.get("marketing_effectiveness", 0.0) < 0.6:
            add("marketing_content", "Marketing set lacks coverage in required fields.", "warning")

        if quality_scores.get("seo_readiness", 0.0) < 0.6:
            add("seo_content", "SEO lists are sparse; ensure keywords/meta sections are filled.", "warning")

        if quality_scores.get("localization", 0.0) < 0.4:
            add("localization", "Outputs rarely reference St. Louis context.", "warning")

        if not verification_report.get("overall_passed", False):
            add("verification", "Chain-of-verification failed; see verification_report for details.", "critical")

        failed_citations = len(citation_report.get("failed") or [])
        if failed_citations:
            add("citations", f"{failed_citations} citations could not be verified.", "critical")

        missing_quotes = len(citation_report.get("missing_quotes") or [])
        if missing_quotes:
            add("citations", f"{missing_quotes} statistics are missing direct quotes.", "warning")

        return warnings

    def _detect_content_gaps(self, output: Dict[str, Any]) -> List[Dict[str, Any]]:
        warnings: List[Dict[str, Any]] = []
        for section, requirements in self.minimum_field_requirements.items():
            content = self._ensure_mapping(output.get(section))
            if not content:
                warnings.append({
                    "section": section,
                    "message": "Section missing entirely.",
                    "severity": "critical",
                })
                continue
            for field, metadata in requirements.items():
                min_items = metadata.get("min", 1)
                critical = metadata.get("critical", False)
                values = content.get(field) or []
                if isinstance(values, list):
                    count = len([item for item in values if item])
                else:
                    count = 1 if values else 0
                if count == 0:
                    warnings.append({
                        "section": section,
                        "message": f"{field} missing.",
                        "severity": "critical" if critical else "warning",
                    })
                elif count < min_items:
                    warnings.append({
                        "section": section,
                        "message": f"{field} below minimum ({count}/{min_items}).",
                        "severity": "warning",
                    })
        return warnings

    def _build_review_checklist(self, warnings: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        if not warnings:
            return [
                {
                    "item": "Spot-check educational sections for Liz's voice and local context before publishing.",
                    "severity": "info",
                }
            ]
        checklist: List[Dict[str, str]] = []
        for warning in warnings:
            message = warning.get("message") or "Resolve validation warning."
            severity = warning.get("severity") or "warning"
            checklist.append({"item": message, "severity": severity})
        return checklist

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
