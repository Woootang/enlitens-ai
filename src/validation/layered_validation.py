"""Layered validation pipeline for Enlitens documents."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when numpy missing
    NUMPY_AVAILABLE = False

    class _FallbackLinalg:
        @staticmethod
        def norm(vec):
            return sum(x * x for x in vec) ** 0.5

    class _FallbackNumpy:
        linalg = _FallbackLinalg()

        @staticmethod
        def mean(values):
            values = list(values)
            return sum(values) / len(values) if values else 0.0

        @staticmethod
        def array(values, dtype=float):
            return [float(v) for v in values]

        @staticmethod
        def zeros(shape):
            rows, cols = shape
            return [[0.0 for _ in range(cols)] for _ in range(rows)]

        @staticmethod
        def pad(vec, padding):
            left, right = padding
            return [0.0] * left + list(vec) + [0.0] * right

        @staticmethod
        def vstack(vectors):
            return list(vectors)

        @staticmethod
        def asarray(values):
            return list(values)

        @staticmethod
        def dot(vec_a, vec_b):
            return sum(a * b for a, b in zip(vec_a, vec_b))

    np = _FallbackNumpy()  # type: ignore

try:  # pragma: no cover - optional dependency
    from FlagEmbedding import BGEM3FlagModel  # type: ignore
except Exception:  # pragma: no cover - graceful fallback
    BGEM3FlagModel = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - fallback when sklearn missing
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ValidationLayerResult:
    """Stores the outcome of a single validation layer."""

    name: str
    passed: bool
    score: float
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayeredValidationResult:
    """Aggregate validation outcome across layers."""

    layers: List[ValidationLayerResult]
    metrics: Dict[str, float]
    flagged_claims: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def passed(self) -> bool:
        return all(layer.passed for layer in self.layers)

    def summary(self) -> str:
        components = [
            f"{layer.name}: {'PASS' if layer.passed else 'FAIL'} (score={layer.score:.2f})"
            for layer in self.layers
        ]
        return " | ".join(components)

    def to_quality_payload(self) -> Dict[str, Any]:
        payload = {
            "precision_at_3": self.metrics.get("precision_at_3"),
            "recall_at_3": self.metrics.get("recall_at_3"),
            "faithfulness": self.metrics.get("faithfulness"),
            "hallucination_rate": self.metrics.get("hallucination_rate"),
            "critical_claims_flagged": len(self.flagged_claims),
            "layer_scores": {layer.name: layer.score for layer in self.layers},
            "layer_failures": [layer.name for layer in self.layers if not layer.passed],
            "evaluated_at": self.timestamp.isoformat(),
        }
        return payload


@dataclass
class LLMJudgeDecision:
    """Represents the decision returned by the LLM judge."""

    verdict: str
    confidence: float
    rationale: str

    @property
    def is_supported(self) -> bool:
        return self.verdict.lower() in {"supported", "mostly_supported"}


class LLMJudge:
    """Minimal interface for routing flagged claims to GPT-4o/Claude."""

    def __init__(self, model_preference: Optional[str] = None):
        self.model_preference = model_preference
        self._client = self._initialise_client(model_preference)

    def _initialise_client(self, model_preference: Optional[str]):  # pragma: no cover - I/O heavy
        """Initialise OpenAI or Anthropic client when credentials are present."""
        api_key = None
        if not model_preference:
            # Prefer OpenAI when available
            api_key = _get_env("OPENAI_API_KEY")
            model_preference = "gpt-4o-mini" if api_key else None

        if model_preference and model_preference.lower().startswith("gpt"):
            api_key = api_key or _get_env("OPENAI_API_KEY")
            if api_key:
                try:
                    from openai import OpenAI  # type: ignore

                    client = OpenAI(api_key=api_key)
                    logger.info("LLM judge configured for OpenAI model %s", model_preference)
                    return ("openai", client, model_preference)
                except Exception as exc:  # pragma: no cover - optional path
                    logger.warning("Failed to initialise OpenAI judge: %s", exc)

        if model_preference and "claude" in model_preference.lower():
            api_key = api_key or _get_env("ANTHROPIC_API_KEY")
            if api_key:
                try:
                    import anthropic  # type: ignore

                    client = anthropic.Anthropic(api_key=api_key)
                    logger.info("LLM judge configured for Anthropic model %s", model_preference)
                    return ("anthropic", client, model_preference)
                except Exception as exc:  # pragma: no cover - optional path
                    logger.warning("Failed to initialise Anthropic judge: %s", exc)

        if not api_key:
            logger.info("LLM judge running in heuristic mode - no API key detected")
        return None

    def evaluate_claim(self, claim: str, source_text: str, *, strict: bool = False) -> LLMJudgeDecision:
        """Evaluate a claim using GPT-4o/Claude when available, otherwise heuristics."""
        if not claim.strip():
            return LLMJudgeDecision("unsupported", 0.0, "Empty claim")

        if self._client:
            provider, client, model = self._client
            prompt = self._build_prompt(claim, source_text, strict=strict)
            try:  # pragma: no cover - network call
                if provider == "openai":
                    response = client.responses.create(
                        model=model,
                        input=prompt,
                        temperature=0.2 if strict else 0.4,
                    )
                    content = response.output[0].content[0].text  # type: ignore[attr-defined]
                else:
                    response = client.messages.create(
                        model=model,
                        max_tokens=300,
                        temperature=0.2 if strict else 0.4,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    content = response.content[0].text  # type: ignore[attr-defined]
                return self._parse_verdict(content)
            except Exception as exc:
                logger.warning("LLM judge call failed (%s). Falling back to heuristics.", exc)

        # Fallback: lexical heuristic comparing token overlap
        overlap = _token_overlap(claim, source_text)
        if overlap > 0.6:
            return LLMJudgeDecision("supported", overlap, "High lexical overlap with source text")
        if overlap > 0.4:
            return LLMJudgeDecision("mostly_supported", overlap, "Moderate lexical overlap")
        return LLMJudgeDecision("unsupported", overlap, "Low overlap; flagged for review")

    def _build_prompt(self, claim: str, source_text: str, *, strict: bool) -> str:
        instruction = (
            "You are a meticulous scientific fact checker."
            " Label the claim as supported, mostly_supported, unsupported, or contradictory"
            " compared to the provided source text."
        )
        if strict:
            instruction += " Be conservative and flag uncertainties as unsupported."
        return (
            f"{instruction}\n\nClaim:\n{claim}\n\nSource text:\n{source_text}\n\n"
            "Respond with JSON: {\"verdict\": <label>, \"confidence\": <0-1>, \"rationale\": <text>}"
        )

    def _parse_verdict(self, response_text: str) -> LLMJudgeDecision:
        try:
            data = json.loads(_extract_json_blob(response_text))
            return LLMJudgeDecision(
                verdict=data.get("verdict", "unsupported"),
                confidence=float(data.get("confidence", 0.0)),
                rationale=data.get("rationale", ""),
            )
        except Exception as exc:
            logger.warning("Unable to parse LLM verdict (%s); using heuristic fallback", exc)
            return LLMJudgeDecision("unsupported", 0.0, "Unparseable LLM response")


class StructuralValidator:
    """Checks structural integrity of generated documents."""

    def validate(self, document: Any) -> ValidationLayerResult:
        issues: List[str] = []
        score_components: List[float] = []

        metadata = getattr(document, "source_metadata", None)
        archival = getattr(document, "archival_content", None)
        synthesis = getattr(document, "ai_synthesis", None)

        if not metadata or not getattr(metadata, "title", "").strip():
            issues.append("Missing source title")
        else:
            score_components.append(1.0)

        if not archival or len(getattr(archival, "full_document_text_markdown", "")) < 500:
            issues.append("Insufficient archival content")
            score_components.append(0.0)
        else:
            score_components.append(1.0)

        if not synthesis or not getattr(synthesis, "enlitens_takeaway", "").strip():
            issues.append("Missing Enlitens takeaway")
        else:
            score_components.append(1.0)

        key_findings = getattr(synthesis, "key_findings", []) if synthesis else []
        if len(key_findings) < 3:
            issues.append("Fewer than three key findings detected")
            score_components.append(0.5 if key_findings else 0.0)
        else:
            score_components.append(1.0)

        contraindications = getattr(synthesis, "contraindications", []) if synthesis else []
        if not contraindications:
            issues.append("Contraindications missing")
            score_components.append(0.2)
        else:
            score_components.append(1.0)

        score = sum(score_components) / max(len(score_components), 1)
        return ValidationLayerResult(
            name="Structural Rules",
            passed=score >= 0.7,
            score=score,
            issues=issues,
            details={"checked_components": len(score_components)},
        )


class SemanticSimilarityValidator:
    """Uses BGE-M3 (or TF-IDF fallback) to score claim faithfulness."""

    def __init__(self, similarity_threshold: float = 0.68):
        self.similarity_threshold = similarity_threshold
        self._embedder = self._load_embedder()

    def _load_embedder(self):  # pragma: no cover - runtime dependent
        if BGEM3FlagModel is not None:
            try:
                model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
                logger.info("Loaded BGE-M3 model for semantic validation")
                return ("bge", model)
            except Exception as exc:
                logger.warning("Failed to load BGE-M3 model: %s", exc)

        if TfidfVectorizer is not None and cosine_similarity is not None:
            logger.info("Falling back to TF-IDF similarity for semantic validation")
            return ("tfidf", TfidfVectorizer())

        logger.warning("No embedding backend available; semantic validation will degrade")
        return None

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768))

        if self._embedder is None:
            # simple bag-of-words fallback
            vectors = [
                np.array([hash(token) % 1000 for token in text.lower().split()], dtype=float)
                for text in texts
            ]
            max_len = max((len(v) for v in vectors), default=1)
            padded = [np.pad(v, (0, max_len - len(v))) for v in vectors]
            return np.vstack(padded)

        backend, model = self._embedder
        if backend == "bge":  # pragma: no cover - requires model download
            encoded = model.encode(texts, batch_size=min(8, len(texts)))
            dense_vectors = encoded.get("dense_vecs") if isinstance(encoded, dict) else encoded
            return np.asarray(dense_vectors)

        if backend == "tfidf":
            matrix = model.fit_transform(texts)
            return matrix.toarray()

        raise RuntimeError("Unknown embedding backend")

    def score_claims(self, claims: Sequence[str], source_text: str) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        if not claims:
            return results

        embeddings = self._embed([source_text, *claims])
        if not embeddings:
            return [(claim, 0.0) for claim in claims]

        if NUMPY_AVAILABLE:
            source_vec = embeddings[0:1]
            claim_vecs = embeddings[1:]
            claim_empty = getattr(claim_vecs, "size", 0) == 0
        else:
            source_vec = embeddings[0]
            claim_vecs = embeddings[1:]
            claim_empty = len(claim_vecs) == 0

        if claim_empty:
            return [(claim, 0.0) for claim in claims]

        if NUMPY_AVAILABLE and cosine_similarity and getattr(claim_vecs, "ndim", 0) == 2:
            similarities = cosine_similarity(claim_vecs, source_vec).flatten()
        else:
            # manual cosine similarity
            similarities = []
            for vec in claim_vecs:
                src_vector = source_vec.flatten() if NUMPY_AVAILABLE else source_vec
                vec_values = vec.flatten() if NUMPY_AVAILABLE and hasattr(vec, "flatten") else vec
                denom = (_vector_norm(vec_values) * _vector_norm(src_vector))
                if denom == 0:
                    similarities.append(0.0)
                else:
                    similarities.append(float(_vector_dot(vec_values, src_vector) / denom))
            similarities = np.asarray(similarities) if NUMPY_AVAILABLE else similarities

        for claim, score in zip(claims, similarities):
            results.append((claim, float(score)))
        return results

    def validate(self, claims: Sequence[str], source_text: str) -> ValidationLayerResult:
        scored_claims = self.score_claims(claims, source_text)
        below_threshold = [
            {"claim": claim, "similarity": score}
            for claim, score in scored_claims
            if score < self.similarity_threshold
        ]
        avg_score = float(np.mean([score for _, score in scored_claims])) if scored_claims else 0.0
        issues = [
            f"Claim below similarity threshold ({item['similarity']:.2f}): {item['claim']}"
            for item in below_threshold
        ]
        detail_payload = {
            "scored_claims": [
                {"claim": claim, "similarity": score} for claim, score in scored_claims
            ],
            "below_threshold": below_threshold,
        }

        return ValidationLayerResult(
            name="Semantic Similarity (BGE-M3)",
            passed=len(below_threshold) == 0,
            score=avg_score,
            issues=issues,
            details=detail_payload,
        )


class SelfConsistencyVoter:
    """Runs multiple judge passes for critical claims."""

    def __init__(self, judge: LLMJudge, votes: int = 3):
        self.judge = judge
        self.votes = max(1, votes)

    def run(self, claim: str, source_text: str) -> Dict[str, Any]:
        verdicts: List[LLMJudgeDecision] = []
        for iteration in range(self.votes):
            strict = iteration % 2 == 1
            verdicts.append(self.judge.evaluate_claim(claim, source_text, strict=strict))

        supported_votes = sum(1 for verdict in verdicts if verdict.is_supported)
        support_ratio = supported_votes / self.votes
        aggregate_confidence = float(np.mean([v.confidence for v in verdicts])) if verdicts else 0.0
        dominant_verdict = (
            "supported" if support_ratio > 0.66 else "unsupported" if support_ratio < 0.34 else "ambiguous"
        )
        return {
            "claim": claim,
            "support_ratio": support_ratio,
            "aggregate_confidence": aggregate_confidence,
            "dominant_verdict": dominant_verdict,
            "votes": [verdict.__dict__ for verdict in verdicts],
        }


class LayeredValidationPipeline:
    """Coordinates structural → semantic → LLM-as-judge validation."""

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.68,
        enable_self_consistency: bool = True,
        llm_model: Optional[str] = None,
    ):
        self.structural_validator = StructuralValidator()
        self.semantic_validator = SemanticSimilarityValidator(similarity_threshold)
        self.judge = LLMJudge(llm_model)
        self.enable_self_consistency = enable_self_consistency
        self.self_consistency = SelfConsistencyVoter(self.judge) if enable_self_consistency else None

    def validate_document(self, document: Any) -> LayeredValidationResult:
        layers: List[ValidationLayerResult] = []
        flagged_claims: List[Dict[str, Any]] = []

        structural_result = self.structural_validator.validate(document)
        layers.append(structural_result)

        synthesis = getattr(document, "ai_synthesis", None)
        source_text = getattr(getattr(document, "archival_content", None), "full_document_text_markdown", "")
        claims = _extract_claim_texts(getattr(synthesis, "key_findings", []))

        semantic_result = self.semantic_validator.validate(claims, source_text)
        layers.append(semantic_result)

        below_threshold = semantic_result.details.get("below_threshold", [])

        for flagged in below_threshold:
            claim_text = flagged.get("claim", "")
            similarity = flagged.get("similarity", 0.0)
            decision = self.judge.evaluate_claim(claim_text, source_text)
            flagged_entry = {
                "claim": claim_text,
                "similarity": similarity,
                "judge_verdict": decision.verdict,
                "judge_confidence": decision.confidence,
                "judge_rationale": decision.rationale,
            }
            if self.enable_self_consistency and self.self_consistency:
                consistency = self.self_consistency.run(claim_text, source_text)
                flagged_entry["self_consistency"] = consistency
            flagged_claims.append(flagged_entry)

        hallucination_rate = (
            len(flagged_claims) / max(len(claims), 1)
            if claims
            else 0.0
        )
        faithfulness = 1.0 - hallucination_rate if hallucination_rate <= 1 else 0.0

        metrics = {
            "precision_at_3": _precision_at_k(claims, flagged_claims, k=3),
            "recall_at_3": _recall_at_k(claims, flagged_claims, k=3),
            "faithfulness": faithfulness,
            "hallucination_rate": hallucination_rate,
        }

        return LayeredValidationResult(layers=layers, metrics=metrics, flagged_claims=flagged_claims)


def _vector_norm(vec: Sequence[float]) -> float:
    if NUMPY_AVAILABLE:
        return float(np.linalg.norm(vec))
    return sum(float(x) * float(x) for x in vec) ** 0.5


def _vector_dot(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if NUMPY_AVAILABLE:
        return float(np.dot(vec_a, vec_b))
    return sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))


def _precision_at_k(claims: Sequence[str], flagged: Sequence[Dict[str, Any]], k: int) -> float:
    if not claims:
        return 1.0
    top_k = claims[:k]
    flagged_claims = {entry.get("claim") for entry in flagged}
    if not top_k:
        return 1.0
    hits = sum(1 for claim in top_k if claim not in flagged_claims)
    return hits / len(top_k)


def _recall_at_k(claims: Sequence[str], flagged: Sequence[Dict[str, Any]], k: int) -> float:
    if not claims:
        return 1.0
    flagged_claims = {entry.get("claim") for entry in flagged}
    protected_claims = [claim for claim in claims if claim not in flagged_claims]
    if not protected_claims:
        return 0.0
    return len(protected_claims[:k]) / min(len(claims), k)


def _extract_claim_texts(key_findings: Optional[Sequence[Any]]) -> List[str]:
    claims: List[str] = []
    if not key_findings:
        return claims

    for finding in key_findings:
        text: Optional[str] = None
        if isinstance(finding, str):
            text = finding
        elif isinstance(finding, dict):
            text = (
                finding.get("finding_text")
                or finding.get("text")
                or finding.get("summary")
            )
        else:
            text = getattr(finding, "finding_text", None) or getattr(finding, "text", None)

        if not text:
            text = str(finding)

        normalised = text.strip()
        if normalised:
            claims.append(normalised)

    return claims


def _token_overlap(text_a: str, text_b: str) -> float:
    tokens_a = _normalise_tokens(text_a)
    tokens_b = _normalise_tokens(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    shared = tokens_a & tokens_b
    return len(shared) / float(min(len(tokens_a), len(tokens_b)))


def _normalise_tokens(text: str) -> set:
    cleaned = [
        token.strip(".,;:!?()[]{}\"'`").lower()
        for token in text.split()
    ]
    return {token for token in cleaned if token}


def _extract_json_blob(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    raise ValueError("JSON object not found")


def _get_env(name: str) -> Optional[str]:  # pragma: no cover - environment specific
    import os

    value = os.environ.get(name)
    return value if value else None

