import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.optional_context_loader import analyze_optional_context, read_optional_text_file


def test_read_optional_text_file_missing_logs_debug(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    missing_path = tmp_path / "intakes.txt"

    with caplog.at_level("DEBUG"):
        content = read_optional_text_file(missing_path, description="client intake insights")

    assert content is None
    assert any(
        "Optional client intake insights file not found" in record.message for record in caplog.records
    )


def test_analyze_optional_context_with_analyzer(tmp_path: Path) -> None:
    path = tmp_path / "intakes.txt"
    path.write_text("sample client content", encoding="utf-8")

    def analyzer(payload):
        assert payload == ["sample client content"]
        return {"processed": True}

    result = analyze_optional_context(
        path,
        description="client intake insights",
        analyzer=analyzer,
    )

    assert result == {"processed": True}


def test_analyze_optional_context_without_analyzer_returns_raw(tmp_path: Path) -> None:
    path = tmp_path / "transcripts.txt"
    path.write_text("founder transcript sample", encoding="utf-8")

    result = analyze_optional_context(
        path,
        description="founder transcript insights",
        analyzer=None,
        fallback_slice=10,
    )

    assert result == {"raw_content": "founder tr"}
