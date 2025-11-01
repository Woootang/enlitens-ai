"""Golden dataset regression harness for the Enlitens multi-agent system.

This module wires together the generated golden dataset with the diagnostic
utilities introduced earlier in the project.  It provides a light-weight
runner that can execute each scenario through the multi-agent pipeline,
evaluate tone alignment via the :mod:`voice_audit` toolkit, and analyse log
files with :mod:`diagnostics`.

The runner is intentionally flexible: in production it can drive the real
``SupervisorAgent`` pipeline, while in tests a stub ``case_runner`` may be
supplied so the aggregation logic can be exercised without large model
dependencies.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence

from src.monitoring.diagnostics import LogSummary, analyse_log
from src.monitoring.voice_audit import VoiceAuditResult, VoiceAuditor

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class PipelineCaseResult:
    """Represents the outcome of running a golden case through the pipeline."""

    document_id: str
    final_text: str
    log_path: Optional[Path]
    raw_output: Dict[str, Any]


@dataclass
class CaseEvaluation:
    """Combined evaluation artefacts for a single golden dataset entry."""

    case: Dict[str, Any]
    pipeline: PipelineCaseResult
    voice: VoiceAuditResult
    diagnostics: Optional[LogSummary]

    def to_dict(self) -> Dict[str, Any]:
        diagnostics_dict = self.diagnostics.to_dict() if self.diagnostics else None
        return {
            "case": self.case,
            "pipeline": {
                "document_id": self.pipeline.document_id,
                "log_path": str(self.pipeline.log_path) if self.pipeline.log_path else None,
            },
            "voice": {
                "similarity": self.voice.similarity,
                "verdict": self.voice.verdict,
            },
            "diagnostics": diagnostics_dict,
        }


@dataclass
class RegressionReport:
    """Holds all case evaluations and produces summary statistics."""

    cases: List[CaseEvaluation]

    def summary(self) -> Dict[str, Any]:
        total_cases = len(self.cases)
        voice_passes = sum(1 for case in self.cases if case.voice.verdict == "pass")
        average_similarity = (
            sum(case.voice.similarity for case in self.cases) / total_cases
            if total_cases
            else 0.0
        )
        agent_error_count = 0
        for case in self.cases:
            if case.diagnostics:
                agent_error_count += sum(
                    stats.errors for stats in case.diagnostics.agent_stats.values()
                )

        return {
            "total_cases": total_cases,
            "voice_pass_rate": voice_passes / total_cases if total_cases else 0.0,
            "average_voice_similarity": average_similarity,
            "agent_error_count": agent_error_count,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "cases": [case.to_dict() for case in self.cases],
        }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _flatten_text(value: Any) -> Iterable[str]:
    """Yield string fragments from arbitrarily nested structures."""

    if value is None:
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for element in value.values():
            yield from _flatten_text(element)
        return
    if isinstance(value, (list, tuple, set)):
        for element in value:
            yield from _flatten_text(element)
        return


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "case"


def _load_voice_reference(path: Path) -> List[str]:
    """Extract reference paragraphs from the canonical voice document."""

    content = path.read_text(encoding="utf-8")
    paragraphs: List[str] = []
    current: List[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if line.startswith("#") or line.startswith("---"):
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


# ---------------------------------------------------------------------------
# Regression runner
# ---------------------------------------------------------------------------


CaseRunner = Callable[[Dict[str, Any]], Awaitable[PipelineCaseResult]]


class GoldenRegressionRunner:
    """Execute the golden dataset through the pipeline and aggregate scores."""

    def __init__(
        self,
        dataset_path: Path | str = Path("golden_dataset/golden_cases_v2.jsonl"),
        voice_reference_path: Path | str = Path("docs/enlitens_voice_reference.md"),
        *,
        case_runner: Optional[CaseRunner] = None,
        voice_auditor: Optional[VoiceAuditor] = None,
        max_cases: Optional[int] = None,
        voice_threshold: float = 0.72,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.voice_reference_path = Path(voice_reference_path)
        self.max_cases = max_cases
        self.voice_threshold = voice_threshold

        references = _load_voice_reference(self.voice_reference_path)
        if not references:
            raise ValueError("Voice reference document produced no samples")

        self.voice_auditor = voice_auditor or VoiceAuditor(references)
        self.case_runner = case_runner or self._default_case_runner
        self._supervisor = None

    async def run(self) -> RegressionReport:
        cases = self._load_cases()
        evaluations: List[CaseEvaluation] = []

        try:
            for case in cases:
                pipeline_result = await self.case_runner(case)
                voice_result = self.voice_auditor.score(
                    pipeline_result.document_id,
                    pipeline_result.final_text,
                    threshold=self.voice_threshold,
                )
                diagnostics = None
                if pipeline_result.log_path:
                    try:
                        diagnostics = analyse_log(pipeline_result.log_path)
                    except FileNotFoundError:
                        LOGGER.warning(
                            "Log file for case %s missing at %s",
                            pipeline_result.document_id,
                            pipeline_result.log_path,
                        )
                evaluations.append(
                    CaseEvaluation(
                        case=case,
                        pipeline=pipeline_result,
                        voice=voice_result,
                        diagnostics=diagnostics,
                    )
                )
        finally:
            if self._supervisor is not None:
                try:
                    await self._supervisor.cleanup()
                except Exception as exc:  # pragma: no cover - defensive logging
                    LOGGER.warning("Supervisor cleanup failed: %s", exc)
                self._supervisor = None

        return RegressionReport(evaluations)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_cases(self) -> List[Dict[str, Any]]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Golden dataset not found: {self.dataset_path}")
        cases: List[Dict[str, Any]] = []
        with self.dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                cases.append(json.loads(line))
        if self.max_cases is not None:
            cases = cases[: self.max_cases]
        return cases

    async def _default_case_runner(self, case: Dict[str, Any]) -> PipelineCaseResult:
        """Run a golden case through the real supervisor pipeline."""

        from src.agents.supervisor_agent import SupervisorAgent

        if self._supervisor is None:
            self._supervisor = SupervisorAgent()
            init_ok = await self._supervisor.initialize()
            if not init_ok:
                raise RuntimeError("Failed to initialize SupervisorAgent for regression run")

        document_id = _slugify(case.get("title", "golden-case"))
        payload = {
            "document_id": document_id,
            "document_text": case.get("source_excerpt", ""),
            "doc_type": case.get("doc_type", "regression"),
        }
        result = await self._supervisor.process_document(payload)

        final_text = "\n\n".join(
            fragment.strip()
            for fragment in _flatten_text(result.get("agent_outputs", {}))
            if fragment and fragment.strip()
        )
        if not final_text:
            # Fall back to document text so the voice audit still runs.
            final_text = case.get("source_excerpt", "")

        return PipelineCaseResult(
            document_id=result.get("document_id", document_id),
            final_text=final_text,
            log_path=None,
            raw_output=result,
        )


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


async def _cli_async(args: argparse.Namespace) -> None:
    runner = GoldenRegressionRunner(
        dataset_path=args.dataset,
        voice_reference_path=args.voice_reference,
        max_cases=args.max_cases,
        voice_threshold=args.voice_threshold,
    )
    report = await runner.run()
    json.dump(report.to_dict(), fp=args.output, indent=2, ensure_ascii=False)
    args.output.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run golden regression evaluation")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("golden_dataset/golden_cases_v2.jsonl"),
        help="Path to the generated golden dataset",
    )
    parser.add_argument(
        "--voice-reference",
        type=Path,
        default=Path("docs/enlitens_voice_reference.md"),
        help="Canonical Enlitens voice reference document",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit the number of cases to evaluate (useful for smoke tests)",
    )
    parser.add_argument(
        "--voice-threshold",
        type=float,
        default=0.72,
        help="Similarity threshold for voice audit pass/fail",
    )
    parser.add_argument(
        "--output",
        type=argparse.FileType("w", encoding="utf-8"),
        default="-",
        help="Where to write the JSON quality report (default: stdout)",
    )

    args = parser.parse_args(argv)
    asyncio.run(_cli_async(args))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    raise SystemExit(main())

