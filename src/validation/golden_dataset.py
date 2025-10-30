"""Golden dataset utilities for regression testing."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - fallback without numpy
    class _FallbackNumpy:
        @staticmethod
        def mean(values):
            values = list(values)
            return sum(values) / len(values) if values else 0.0

    np = _FallbackNumpy()  # type: ignore

from .layered_validation import SemanticSimilarityValidator


@dataclass
class GoldenRecord:
    document_id: str
    source_text: str
    expected_key_findings: List[str]
    generated_key_findings: List[str]
    critical_claims: List[str]

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "GoldenRecord":
        return cls(
            document_id=payload["document_id"],
            source_text=payload["source_text"],
            expected_key_findings=list(payload.get("expected_key_findings", [])),
            generated_key_findings=list(payload.get("generated_key_findings", [])),
            critical_claims=list(payload.get("critical_claims", [])),
        )


class GoldenDataset:
    """Loads and manages golden dataset records."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.records: List[GoldenRecord] = self._load()

    def _load(self) -> List[GoldenRecord]:
        records: List[GoldenRecord] = []
        with self.dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                records.append(GoldenRecord.from_json(payload))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterable[GoldenRecord]:
        return iter(self.records)


class GoldenDatasetEvaluator:
    """Computes regression metrics for the golden dataset."""

    def __init__(
        self,
        records: Sequence[GoldenRecord],
        *,
        similarity_threshold: float = 0.68,
    ):
        self.records = list(records)
        self.similarity_validator = SemanticSimilarityValidator(similarity_threshold)

    def evaluate(self) -> Dict[str, float]:
        if not self.records:
            return {
                "precision_at_3": 1.0,
                "recall_at_3": 1.0,
                "faithfulness": 1.0,
                "hallucination_rate": 0.0,
            }

        precision_hits = 0
        precision_total = 0
        recall_hits = 0
        recall_total = 0
        similarity_scores: List[float] = []
        hallucinated_claims = 0
        total_claims = 0

        for record in self.records:
            expected = record.expected_key_findings
            generated = record.generated_key_findings
            total_claims += len(generated)
            raw_scores = self.similarity_validator.score_claims(generated, record.source_text)
            adjusted_scores: List[float] = []
            matches = {}
            for claim, score in raw_scores:
                adjusted = score
                if adjusted < self.similarity_validator.similarity_threshold and (
                    claim.lower() in record.source_text.lower() or claim in expected
                ):
                    adjusted = 1.0
                adjusted_scores.append(adjusted)
                if adjusted >= self.similarity_validator.similarity_threshold:
                    matches[claim] = adjusted

            similarity_scores.extend(adjusted_scores)

            precision_hits += sum(1 for claim in generated[:3] if claim in matches)
            precision_total += min(len(generated), 3)

            recall_hits += sum(1 for claim in expected[:3] if claim in matches)
            recall_total += min(len(expected), 3)

            hallucinated_claims += len(generated) - len(matches)

        precision = precision_hits / precision_total if precision_total else 1.0
        recall = recall_hits / recall_total if recall_total else 1.0
        faithfulness = float(np.mean(similarity_scores)) if similarity_scores else 1.0
        hallucination_rate = hallucinated_claims / total_claims if total_claims else 0.0

        return {
            "precision_at_3": float(precision),
            "recall_at_3": float(recall),
            "faithfulness": float(faithfulness),
            "hallucination_rate": float(hallucination_rate),
        }

