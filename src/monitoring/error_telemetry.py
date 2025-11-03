"""Utility for recording structured telemetry about warnings and errors."""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class TelemetrySeverity:
    """Enumeration of supported telemetry severities."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"

    @classmethod
    def all(cls) -> set[str]:
        return {cls.CRITICAL, cls.MAJOR, cls.MINOR}


@dataclass
class _AgentSummary:
    count: int = 0
    first_seen: Optional[str] = None


@dataclass
class TelemetryRecorder:
    """Recorder that persists structured error telemetry and maintains summaries."""

    log_dir: Path = field(default_factory=lambda: Path("logs") / "errors")

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._summary: Dict[str, Any] = {
            "counts_by_severity": defaultdict(int),
            "affected_agents": defaultdict(lambda: _AgentSummary()),
            "first_occurrence": {},
        }

    def record_event(
        self,
        doc_id: Optional[str],
        agent: str,
        severity: str,
        impact: str,
        details: Any,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Persist a telemetry event and update in-memory aggregates."""

        if severity not in TelemetrySeverity.all():
            raise ValueError(f"Unsupported telemetry severity: {severity}")

        ts = timestamp or datetime.utcnow()
        iso_timestamp = ts.isoformat() + "Z"
        event: Dict[str, Any] = {
            "timestamp": iso_timestamp,
            "doc_id": doc_id,
            "agent": agent,
            "severity": severity,
            "impact": impact,
            "details": details,
        }

        log_path = self.log_dir / f"{ts.date().isoformat()}.jsonl"

        with self._lock:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

            counts = self._summary["counts_by_severity"]
            counts[severity] += 1

            agents: Dict[str, _AgentSummary] = self._summary["affected_agents"]
            agent_summary = agents[agent]
            agent_summary.count += 1
            if not agent_summary.first_seen:
                agent_summary.first_seen = iso_timestamp

            first_occurrence: Dict[str, str] = self._summary["first_occurrence"]
            if severity not in first_occurrence:
                first_occurrence[severity] = iso_timestamp

    def get_summary_snapshot(self) -> Dict[str, Any]:
        """Return a deep copy of the current summary suitable for serialization."""
        with self._lock:
            snapshot = {
                "counts_by_severity": dict(self._summary["counts_by_severity"]),
                "affected_agents": {
                    agent: {"count": summary.count, "first_seen": summary.first_seen}
                    for agent, summary in self._summary["affected_agents"].items()
                },
                "first_occurrence": dict(self._summary["first_occurrence"]),
            }
        return snapshot

    def flush_summary(self, suffix: Optional[str] = None) -> Path:
        """Persist the aggregated summary to disk and return the output path."""

        timestamp = datetime.utcnow()
        filename_suffix = f"-{suffix}" if suffix else ""
        summary_path = self.log_dir / f"{timestamp.date().isoformat()}{filename_suffix}-summary.json"
        summary_payload = {
            "generated_at": timestamp.isoformat() + "Z",
            "summary": self.get_summary_snapshot(),
        }
        with self._lock:
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(summary_payload, handle, ensure_ascii=False, indent=2)
        return summary_path


telemetry_recorder = TelemetryRecorder()


def log_with_telemetry(
    logger_method: Callable[..., None],
    message: str,
    *args: Any,
    agent: str,
    severity: str,
    impact: str,
    doc_id: Optional[str] = None,
    details: Optional[Any] = None,
    **kwargs: Any,
) -> None:
    """Log a message and mirror it to the telemetry recorder."""

    logger_method(message, *args, **kwargs)
    if details is None:
        formatted = message
        if args:
            try:
                formatted = message % args
            except TypeError:
                try:
                    formatted = message.format(*args)
                except Exception:
                    formatted = message
        details_payload: Any = formatted
    else:
        details_payload = details

    telemetry_recorder.record_event(
        doc_id=doc_id,
        agent=agent,
        severity=severity,
        impact=impact,
        details=details_payload,
    )
