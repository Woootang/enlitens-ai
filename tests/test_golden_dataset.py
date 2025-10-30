"""Regression tests for the golden dataset quality harness."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.golden_dataset import GoldenDataset, GoldenDatasetEvaluator


def test_golden_dataset_size():
    dataset_path = Path(__file__).resolve().parents[1] / "golden_dataset" / "golden_documents.jsonl"
    dataset = GoldenDataset(dataset_path)
    assert 50 <= len(dataset) <= 100, "Golden dataset must contain between 50 and 100 documents"


def test_golden_dataset_metrics_are_perfect():
    dataset_path = Path(__file__).resolve().parents[1] / "golden_dataset" / "golden_documents.jsonl"
    dataset = GoldenDataset(dataset_path)
    evaluator = GoldenDatasetEvaluator(dataset.records)
    metrics = evaluator.evaluate()

    assert metrics["precision_at_3"] == 1.0
    assert metrics["recall_at_3"] == 1.0
    # Faithfulness should also be high because generated == expected.
    assert metrics["faithfulness"] > 0.5
    assert metrics["hallucination_rate"] == 0.0

