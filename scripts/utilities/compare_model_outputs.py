#!/usr/bin/env python3
"""
Compare MedGemma vs Llama JSON outputs.

This tool aligns output files produced by `run_local_comparison.py`
and generates a simple report highlighting textual differences and
optional BERTScore similarity against the source text.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from bert_score import score as bert_score  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    bert_score = None  # type: ignore

logger = logging.getLogger(__name__)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def section_diffs(a: Dict, b: Dict, keys: List[str]) -> Dict[str, Dict[str, int]]:
    results: Dict[str, Dict[str, int]] = {}
    for key in keys:
        text_a = a.get("extraction", {}).get(key, "") or ""
        text_b = b.get("extraction", {}).get(key, "") or ""
        if not text_a and not text_b:
            continue
        diff = {
            "len_a": len(text_a),
            "len_b": len(text_b),
            "char_diff": len(text_a) - len(text_b),
        }
        results[key] = diff
    return results


def compute_bertscore_pair(
    source_text: str,
    candidate: str,
) -> Optional[Tuple[float, float, float]]:
    if not bert_score:
        return None
    try:
        P, R, F1 = bert_score(
            [candidate],
            [source_text],
            lang="en",
            verbose=False,
        )
        return (float(P[0]), float(R[0]), float(F1[0]))
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("BERTScore failed: %s", exc)
        return None


def compare_pair(med_file: Path, llama_file: Path) -> Dict:
    med_payload = load_json(med_file)
    llama_payload = load_json(llama_file)

    sections = ["background", "methods", "findings", "limitations", "statistics", "conclusions"]
    diffs = section_diffs(med_payload, llama_payload, sections)

    bert_scores: Dict[str, Dict[str, float]] = {}
    source_text = med_payload.get("docling", {}).get("verbatim_text", "")
    for key in sections:
        med_text = med_payload.get("extraction", {}).get(key, "")
        llama_text = llama_payload.get("extraction", {}).get(key, "")
        med_score = compute_bertscore_pair(source_text, med_text) if med_text else None
        llama_score = compute_bertscore_pair(source_text, llama_text) if llama_text else None
        bert_scores[key] = {
            "medgemma_f1": med_score[2] if med_score else None,
            "llama_f1": llama_score[2] if llama_score else None,
        }

    return {
        "document_id": med_payload.get("document_id"),
        "sections": diffs,
        "bertscore": bert_scores,
        "medgemma_file": str(med_file),
        "llama_file": str(llama_file),
    }


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MedGemma vs Llama outputs.")
    parser.add_argument("--medgemma-dir", required=True, help="Directory with MedGemma JSON outputs.")
    parser.add_argument("--llama-dir", required=True, help="Directory with Llama JSON outputs.")
    parser.add_argument("--report", default="comparison_report.json", help="Where to store the JSON report.")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)
    med_dir = Path(args.medgemma_dir)
    llama_dir = Path(args.llama_dir)

    report: List[Dict] = []
    missing: List[str] = []

    for med_file in sorted(med_dir.glob("*.json")):
        document_id = med_file.stem.split("_")[0]
        llama_file = llama_dir / f"{document_id}_llama.json"
        if not llama_file.exists():
            missing.append(document_id)
            continue
        report.append(compare_pair(med_file, llama_file))

    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("📊 Wrote comparison report for %d documents -> %s", len(report), args.report)
    if missing:
        logger.warning("⚠️ Missing llama outputs for %s", ", ".join(missing[:10]))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))

