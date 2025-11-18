#!/usr/bin/env python3
"""
Helper script to pilot the multi-agent pipeline on a single document while
tracking Gemini / Deep Research usage deltas.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict

import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.usage_tracker import get_usage_summary  # noqa: E402


def run_pipeline(pdf_path: Path, output_path: Path, stl_report: Path | None) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "process_multi_agent_corpus.py"),
        "--input-dir",
        str(pdf_path.parent),
        "--output-file",
        str(output_path),
    ]
    if stl_report:
        cmd.extend(["--st-louis-report", str(stl_report)])

    subprocess.run(cmd, check=True)


def diff_usage(before: Dict, after: Dict) -> Dict[str, Dict[str, int]]:
    summary = {}
    before_tools = before.get("tools", {})
    after_tools = after.get("tools", {})
    for tool, after_payload in after_tools.items():
        prev = before_tools.get(tool, {})
        summary[tool] = {
            "before": prev.get("count", 0),
            "after": after_payload.get("count", 0),
            "delta": after_payload.get("count", 0) - prev.get("count", 0),
            "limit": after_payload.get("limit"),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single-document pipeline pilot with usage tracking.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF to process.")
    parser.add_argument("--stl-report", help="Optional path to the St. Louis report PDF.")
    parser.add_argument("--output", default="pilot_output.json", help="Where to write the resulting JSON.")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="enlitens-pilot-"))
    tmp_pdf = tmp_dir / pdf_path.name
    shutil.copy(pdf_path, tmp_pdf)

    output_path = Path(args.output).resolve()

    before_usage = get_usage_summary()
    start = time.time()
    try:
        run_pipeline(tmp_pdf, output_path, Path(args.stl_report).resolve() if args.stl_report else None)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    after_usage = get_usage_summary()
    delta = diff_usage(before_usage, after_usage)

    print("✅ Pilot completed")
    print(f"   Output file: {output_path}")
    print(f"   Duration: {time.time() - start:.2f}s")
    print("   Usage deltas (requests today):")
    print(json.dumps(delta, indent=2))


if __name__ == "__main__":
    main()

