import json
from datetime import datetime

from src.monitoring.error_telemetry import TelemetryRecorder, TelemetrySeverity


def test_record_event_persists_and_aggregates(tmp_path):
    recorder = TelemetryRecorder(log_dir=tmp_path)

    first_ts = datetime(2024, 1, 2, 10, 30, 0)
    second_ts = datetime(2024, 1, 2, 11, 45, 0)
    recorder.record_event(
        doc_id="doc-1",
        agent="agent-a",
        severity=TelemetrySeverity.MAJOR,
        impact="Primary path failed",
        details={"error": "boom"},
        timestamp=first_ts,
    )
    recorder.record_event(
        doc_id="doc-2",
        agent="agent-b",
        severity=TelemetrySeverity.MINOR,
        impact="Fallback executed",
        details="fallback succeeded",
        timestamp=second_ts,
    )

    log_file = tmp_path / "2024-01-02.jsonl"
    assert log_file.exists()
    with log_file.open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle]

    assert lines[0]["doc_id"] == "doc-1"
    assert lines[1]["impact"] == "Fallback executed"

    summary = recorder.get_summary_snapshot()
    assert summary["counts_by_severity"] == {
        TelemetrySeverity.MAJOR: 1,
        TelemetrySeverity.MINOR: 1,
    }
    assert summary["affected_agents"]["agent-a"]["count"] == 1
    assert summary["affected_agents"]["agent-b"]["first_seen"].startswith("2024-01-02T11:45:00")

    summary_path = recorder.flush_summary(suffix="run")
    assert summary_path.exists()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["summary"]["counts_by_severity"][TelemetrySeverity.MAJOR] == 1
    assert data["summary"]["first_occurrence"][TelemetrySeverity.MINOR].startswith("2024-01-02T11:45:00")
