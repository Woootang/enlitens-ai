import textwrap
from pathlib import Path

from src.monitoring.diagnostics import analyse_log, render_summary


def _write_log(tmp_path: Path) -> Path:
    content = textwrap.dedent(
        """
        2025-01-10 12:00:00 - supervisor_agent - INFO - Stage 1/3 ▶ Planning task list
        2025-01-10 12:00:01 - supervisor_agent - INFO - Document DOC-1 started
        2025-01-10 12:00:02 - extraction_agent - ERROR - ExtractionAgent encountered parse failure
        2025-01-10 12:00:03 - extraction_agent - INFO - Document DOC-1 completed
        2025-01-10 12:00:04 - synthesis_agent - WARNING - Document DOC-1 requires tone adjustment
        2025-01-10 12:00:05 - supervisor_agent - INFO - Stage 2/3 ✓ Delegation complete
        2025-01-10 12:00:06 - validation_agent - INFO - Document DOC-1 completed
        2025-01-10 12:00:07 - supervisor_agent - INFO - Document DOC-2 started
        2025-01-10 12:00:08 - validation_agent - ERROR - ValidationAgent failed for Document DOC-2
        2025-01-10 12:00:09 - supervisor_agent - INFO - Document DOC-2 failed
        """
    ).strip()
    log_path = tmp_path / "test.log"
    log_path.write_text(content)
    return log_path


def test_analyse_log_counts(tmp_path):
    log_path = _write_log(tmp_path)
    summary = analyse_log(log_path)

    assert summary.levels["INFO"] == 7
    assert summary.levels["ERROR"] == 2
    assert summary.levels["WARNING"] == 1

    assert summary.agent_stats["SupervisorAgent"].info == 5
    assert summary.agent_stats["ExtractionAgent"].errors == 1
    assert summary.agent_stats["ValidationAgent"].errors == 1

    assert summary.documents["DOC-1"].started == 1
    assert summary.documents["DOC-1"].completed == 2
    assert summary.documents["DOC-2"].failed == 1

    assert summary.stage_events[0].stage == 1
    assert summary.stage_events[0].total == 3


def test_render_summary_text(tmp_path):
    log_path = _write_log(tmp_path)
    summary = analyse_log(log_path)
    output = render_summary(summary)
    assert "Log levels" in output
    assert "SupervisorAgent" in output
    assert "DOC-2" in output
