"""Utilities for baselining multi-agent pipeline telemetry.

This module provides lightweight analytics for the log file emitted by
``process_multi_agent_corpus.py``. It enables Week 1 baseline diagnostics by
summarising log levels, per-agent failures, processing stage durations, and
document throughput. The goal is to surface failure modes quickly without
requiring a dedicated observability stack.

The code is dependency-light so it can run on the same GPU workstation that
executes the generation pipeline.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ANSI_ESCAPE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
LOG_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - "
    r"(?P<logger>[^-]+) - (?P<level>[^-]+) - (?P<message>.*)$"
)
STAGE_RE = re.compile(
    r"Stage (?P<current>\d+)/(?:of )?(?P<total>\d+)",  # matches "Stage 2/6" or "Stage 2 of 6"
)
DOC_EVENT_RE = re.compile(
    r"Document (?P<doc_id>[^\s]+) (?P<status>started|completed|failed)", re.IGNORECASE
)
AGENT_HINT_RE = re.compile(r"(?P<agent>[A-Z][A-Za-z]+Agent)")


@dataclass
class AgentStats:
    """Aggregated statistics for a single agent logger."""

    name: str
    warnings: int = 0
    errors: int = 0
    info: int = 0

    def update(self, level: str) -> None:
        level_upper = level.upper()
        if level_upper == "ERROR" or level_upper == "CRITICAL":
            self.errors += 1
        elif level_upper == "WARNING":
            self.warnings += 1
        else:
            self.info += 1


@dataclass
class StageEvent:
    """Represents a logged processing stage transition."""

    timestamp: datetime
    stage: int
    total: int
    message: str


@dataclass
class DocumentStatus:
    """Tracks start/completion/failure counts for a document."""

    started: int = 0
    completed: int = 0
    failed: int = 0


@dataclass
class LogSummary:
    """Structured summary derived from the log file."""

    path: Path
    levels: Counter = field(default_factory=Counter)
    agent_stats: Dict[str, AgentStats] = field(default_factory=dict)
    stage_events: List[StageEvent] = field(default_factory=list)
    documents: Dict[str, DocumentStatus] = field(default_factory=lambda: defaultdict(DocumentStatus))

    def to_dict(self) -> Dict[str, object]:
        return {
            "log_path": str(self.path),
            "levels": dict(self.levels),
            "agents": {
                name: {
                    "info": stats.info,
                    "warnings": stats.warnings,
                    "errors": stats.errors,
                }
                for name, stats in sorted(self.agent_stats.items())
            },
            "stage_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "stage": event.stage,
                    "total": event.total,
                    "message": event.message,
                }
                for event in self.stage_events
            ],
            "documents": {
                doc_id: {
                    "started": status.started,
                    "completed": status.completed,
                    "failed": status.failed,
                }
                for doc_id, status in sorted(self.documents.items())
            },
        }

    def agent_failures(self) -> Dict[str, int]:
        return {name: stats.errors for name, stats in self.agent_stats.items() if stats.errors}


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes so regex parsing works with colored logs."""

    return ANSI_ESCAPE.sub("", text)


def parse_stage(message: str, timestamp: datetime) -> Optional[StageEvent]:
    match = STAGE_RE.search(message)
    if not match:
        return None
    current = match.group("current")
    total = match.group("total")
    try:
        return StageEvent(timestamp=timestamp, stage=int(current), total=int(total), message=message.strip())
    except ValueError:
        return None


def parse_agent_name(logger: str, message: str) -> Optional[str]:
    """Infer the agent name from the logger or message."""

    candidates: List[str] = []
    if "agent" in logger.lower():
        candidates.append(logger.strip())
    agent_hint = AGENT_HINT_RE.search(message)
    if agent_hint:
        candidates.append(agent_hint.group("agent"))
    if candidates:
        # Normalise snake_case names into TitleCase for readability
        raw = candidates[0]
        if "_" in raw:
            return "".join(part.capitalize() for part in raw.split("_"))
        return raw
    return None


def parse_document_event(message: str) -> Optional[Dict[str, str]]:
    match = DOC_EVENT_RE.search(message)
    if not match:
        return None
    return {"document_id": match.group("doc_id"), "status": match.group("status").lower()}


def analyse_log(path: Path) -> LogSummary:
    summary = LogSummary(path=path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            clean_line = strip_ansi(raw_line).strip()
            if not clean_line:
                continue
            match = LOG_LINE_RE.match(clean_line)
            if not match:
                continue
            timestamp = datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S")
            logger_name = match.group("logger").strip()
            level = match.group("level").strip()
            message = match.group("message").strip()

            summary.levels[level] += 1

            agent_name = parse_agent_name(logger_name, message)
            if agent_name:
                stats = summary.agent_stats.setdefault(agent_name, AgentStats(name=agent_name))
                stats.update(level)

            stage_event = parse_stage(message, timestamp)
            if stage_event:
                summary.stage_events.append(stage_event)

            doc_event = parse_document_event(message)
            if doc_event:
                record = summary.documents[doc_event["document_id"]]
                status = doc_event["status"]
                if status == "started":
                    record.started += 1
                elif status == "completed":
                    record.completed += 1
                elif status == "failed":
                    record.failed += 1

    return summary


def render_summary(summary: LogSummary, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(summary.to_dict(), indent=2)

    lines: List[str] = []
    lines.append(f"Log path: {summary.path}")
    lines.append("Log levels:")
    for level, count in summary.levels.most_common():
        lines.append(f"  - {level}: {count}")

    if summary.agent_stats:
        lines.append("\nAgent diagnostics:")
        for name, stats in sorted(summary.agent_stats.items()):
            lines.append(
                f"  - {name}: {stats.errors} errors, {stats.warnings} warnings, {stats.info} info"
            )
    else:
        lines.append("\nAgent diagnostics: none found in log")

    if summary.stage_events:
        lines.append("\nStage transitions:")
        for event in summary.stage_events[-10:]:  # show latest 10 for brevity
            lines.append(
                f"  - {event.timestamp:%Y-%m-%d %H:%M:%S} :: stage {event.stage}/{event.total} :: {event.message}"
            )

    if summary.documents:
        lines.append("\nDocument throughput:")
        for doc_id, status in sorted(summary.documents.items()):
            lines.append(
                f"  - {doc_id}: started={status.started}, completed={status.completed}, failed={status.failed}"
            )

    failures = summary.agent_failures()
    if failures:
        lines.append("\nAgents requiring attention:")
        for name, count in sorted(failures.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"  - {name}: {count} errors")
    else:
        lines.append("\nAgents requiring attention: none")

    return "\n".join(lines)


def _default_log_path() -> Path:
    return Path("logs") / "enlitens_complete_processing.log"


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise Enlitens pipeline telemetry")
    parser.add_argument("--log", type=Path, default=_default_log_path(), help="Path to log file")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of human summary")
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = analyse_log(args.log)
    print(render_summary(summary, as_json=args.json))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
