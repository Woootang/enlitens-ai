import asyncio
from pathlib import Path

import pytest

from src.testing.golden_regression import GoldenRegressionRunner, PipelineCaseResult


class _StubCaseRunner:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.counter = 0

    async def __call__(self, case):
        doc_id = f"stub-{self.counter}"
        self.counter += 1
        text = (
            "We burn the broken map, detoxify the sea, and hand clients autonomy blueprints. "
            "Systems change or we call out the bullshit loudly."
        )
        log_path = self.tmp_path / f"{doc_id}.log"
        log_path.write_text(
            "2025-01-01 12:00:00 - ValidationAgent - INFO - Document {} completed\n".format(doc_id),
            encoding="utf-8",
        )
        return PipelineCaseResult(
            document_id=doc_id,
            final_text=text,
            log_path=log_path,
            raw_output={"agent_outputs": {"founder_voice_result": {"narrative": text}}},
        )


def test_golden_regression_runner_generates_summary(tmp_path):
    dataset_path = Path("golden_dataset/golden_cases_v2.jsonl")
    if not dataset_path.exists():
        pytest.skip("Golden dataset not present; run build script first")

    runner = GoldenRegressionRunner(
        dataset_path=dataset_path,
        case_runner=_StubCaseRunner(tmp_path),
        max_cases=2,
        voice_threshold=0.3,
    )

    report = asyncio.run(runner.run())
    summary = report.summary()

    assert summary["total_cases"] == 2
    assert summary["voice_pass_rate"] == pytest.approx(1.0)
    assert summary["agent_error_count"] == 0

    serialised = report.to_dict()
    assert serialised["summary"]["total_cases"] == 2
    assert len(serialised["cases"]) == 2

