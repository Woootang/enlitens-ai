"""Regression tests for the golden dataset quality harness."""
import json
from pathlib import Path

from golden_dataset import build_week1_golden_set
from src.validation.golden_dataset import GoldenDataset, GoldenDatasetEvaluator


DATASET_DIR = Path(__file__).resolve().parents[1] / "golden_dataset"


def test_golden_dataset_size():
    dataset = GoldenDataset(DATASET_DIR / "golden_documents.jsonl")
    assert 50 <= len(dataset) <= 100, "Golden dataset must contain between 50 and 100 documents"


def test_golden_dataset_metrics_are_perfect():
    dataset = GoldenDataset(DATASET_DIR / "golden_documents.jsonl")
    evaluator = GoldenDatasetEvaluator(dataset.records)
    metrics = evaluator.evaluate()

    assert metrics["precision_at_3"] == 1.0
    assert metrics["recall_at_3"] == 1.0
    # Faithfulness should also be high because generated == expected.
    assert metrics["faithfulness"] > 0.5
    assert metrics["hallucination_rate"] == 0.0


def test_week1_golden_cases_fixture_is_up_to_date(tmp_path):
    expected_path = DATASET_DIR / "golden_cases_v2.jsonl"
    regen_path = tmp_path / "golden_cases_v2.jsonl"

    build_week1_golden_set.generate_dataset(regen_path)

    expected = [json.loads(line) for line in expected_path.read_text(encoding="utf-8").splitlines()]
    regenerated = [json.loads(line) for line in regen_path.read_text(encoding="utf-8").splitlines()]

    assert regenerated == expected, "Regenerate golden_cases_v2.jsonl via build_week1_golden_set.py"

