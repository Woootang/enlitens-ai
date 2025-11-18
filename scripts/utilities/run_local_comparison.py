#!/usr/bin/env python3
"""
Local MedGemma ↔ Llama comparison pipeline.

This script orchestrates the complete PDF-processing workflow for a given
model: Docling extraction (cached), local LLM extraction, external
enrichment, Gemini JSON validation, and final JSON persistence.  Run it
twice (once per model) and use `scripts/compare_model_outputs.py` to
inspect the differences.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List

from process_pdfs.cache_utils import get_or_create_extraction
from process_pdfs.enrichment import build_enrichment_payload
from process_pdfs.extraction import extract_scientific_content
from src.integrations.gemini_cli_json_assembler import GeminiJSONAssembler
from src.utils.llm_client import LLMClient
from src.utils.local_model_manager import LocalModelManager

logger = logging.getLogger(__name__)


STATUS_FILE = Path("logs/local_status.json")


def update_status(payload: Dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_pdfs(input_path: Path) -> Iterable[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        yield input_path
    elif input_path.is_dir():
        for pdf in sorted(input_path.glob("*.pdf")):
            yield pdf
    else:
        raise FileNotFoundError(f"No PDFs found at {input_path}")


def load_model_client(manager: LocalModelManager, model_key: str) -> LLMClient:
    config = manager.get(model_key)
    model_name = config.get("model_name")
    base_url = config.get("base_url") or "http://localhost:8000/v1"
    logger.info("🔌 Using model=%s (%s)", model_key, model_name)
    return LLMClient(base_url=base_url, model_name=model_name)


def finalize_with_gemini(document_id: str, draft_payload: Dict) -> Dict:
    assembler = GeminiJSONAssembler()
    if not assembler.available:
        logger.warning("Gemini CLI not available; returning draft payload for %s", document_id)
        return draft_payload
    payload = {
        "document_id": document_id,
        "metadata": draft_payload.get("docling", {}).get("metadata"),
        "agent_outputs": {
            "extraction": draft_payload.get("extraction"),
            "enrichment": draft_payload.get("enrichment"),
        },
        "quality": {
            "model": draft_payload.get("model"),
            "source_pdf": draft_payload.get("source_pdf"),
        },
        "context_snippet": draft_payload.get("docling", {}).get("verbatim_text", "")[:2000],
    }
    validated = assembler.assemble_entry(payload)
    if validated:
        logger.info("🤝 Gemini produced consolidated JSON for %s", document_id)
        return {
            **draft_payload,
            "gemini_validated": True,
            "knowledge_entry": validated,
        }
    logger.warning("Gemini assembler returned no payload; falling back to draft for %s", document_id)
    return {**draft_payload, "gemini_validated": False}


def process_document(
    pdf_path: Path,
    model_key: str,
    manager: LocalModelManager,
    output_dir: Path,
    force_extraction: bool = False,
) -> Path:
    start = time.time()
    update_status(
        {
            "model": model_key,
            "document": pdf_path.name,
            "stage": "docling",
            "timestamp": time.time(),
        }
    )
    docling_payload = get_or_create_extraction(pdf_path, cache_root=Path("cache/docling_outputs"), force=force_extraction)
    llm_client = load_model_client(manager, model_key)

    update_status(
        {
            "model": model_key,
            "document": docling_payload["paper_id"],
            "stage": "llm_extraction",
            "timestamp": time.time(),
        }
    )
    extraction = extract_scientific_content(
        document_text=docling_payload["verbatim_text"],
        llm_client=llm_client,
    )
    update_status(
        {
            "model": model_key,
            "document": docling_payload["paper_id"],
            "stage": "enrichment",
            "timestamp": time.time(),
        }
    )
    enrichment = build_enrichment_payload(docling_payload.get("metadata", {}), extraction)

    draft = {
        "document_id": docling_payload["paper_id"],
        "model": model_key,
        "source_pdf": str(pdf_path),
        "docling": docling_payload,
        "extraction": extraction,
        "enrichment": enrichment,
        "processing_seconds": time.time() - start,
    }

    update_status(
        {
            "model": model_key,
            "document": docling_payload["paper_id"],
            "stage": "gemini_validation",
            "timestamp": time.time(),
            "enrichment_counts": {
                "wikipedia": len(enrichment.get("wikipedia", {})),
                "citations": len(enrichment.get("citations", {})),
            },
        }
    )
    final_payload = finalize_with_gemini(docling_payload["paper_id"], draft)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{docling_payload['paper_id']}_{model_key}.json"
    output_file.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    total_time = time.time() - start
    update_status(
        {
            "model": model_key,
            "document": docling_payload["paper_id"],
            "stage": "complete",
            "timestamp": time.time(),
            "duration_seconds": total_time,
            "gemini_validated": final_payload.get("gemini_validated", False),
        }
    )
    logger.info("✅ Saved %s (%.1fs)", output_file, total_time)
    return output_file


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local PDF pipeline for a specific model.")
    parser.add_argument("--input", required=True, help="Path to a PDF file or directory of PDFs.")
    parser.add_argument("--model", required=True, choices=["medgemma", "llama"], help="Model key to use.")
    parser.add_argument("--output-dir", default="data/local_runs", help="Directory to store outputs.")
    parser.add_argument("--force-extraction", action="store_true", help="Regenerate Docling cache even if present.")
    parser.add_argument("--auto-start", action="store_true", help="Call the model start command before processing.")
    parser.add_argument("--auto-stop", action="store_true", help="Call the model stop command after processing.")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    log_path = Path("logs/processing.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
        force=True,
    )
    args = parse_args(argv)

    manager = LocalModelManager()
    if args.auto_start:
        manager.start(args.model)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) / args.model
    processed_files: List[Path] = []

    try:
        for pdf in iter_pdfs(input_path):
            logger.info("📄 Processing %s with %s", pdf.name, args.model)
            processed_files.append(
                process_document(pdf, args.model, manager, output_dir, args.force_extraction)
            )
    finally:
        if args.auto_stop:
            manager.stop(args.model)

    logger.info("🏁 Completed %d documents. Outputs in %s", len(processed_files), output_dir)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))

