#!/usr/bin/env python3
"""
End-to-end ingestion orchestrator.

Scans the input PDF folder, executes the Docling → LLM → enrichment pipeline,
and persists the combined record to the JSONL knowledge ledger. Successfully
processed PDFs are moved to the processed directory, while failures are shunted
to a retry folder for manual inspection.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.neo4j_publisher import Neo4jPublisher  # noqa: E402
from src.pipeline.document_pipeline import process_pdf_document  # noqa: E402
from src.persistence.postgres_store import PostgresStore  # noqa: E402
from src.persistence.vector_mirror import VectorMirror  # noqa: E402
from src.utils.jsonl_store import append_jsonl_record  # noqa: E402
from src.utils.llm_client import LLMClient  # noqa: E402
from src.utils.local_model_manager import LocalModelManager  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = Path("enlitens_corpus/input_pdfs")
DEFAULT_PROCESSED_DIR = Path("enlitens_corpus/processed")
DEFAULT_FAILED_DIR = Path("enlitens_corpus/failed")
DEFAULT_LEDGER = Path("data/knowledge_base/enliten_knowledge_base.jsonl")
DEFAULT_LEDGER_MIRROR = Path("data/knowledge_base/enliten_knowledge_base.jsonl.bak")


def iter_pdfs(root: Path) -> Iterable[Path]:
    if root.is_file() and root.suffix.lower() == ".pdf":
        yield root
        return
    if not root.exists():
        raise FileNotFoundError(f"Input path {root} does not exist.")
    for path in sorted(root.glob("*.pdf")):
        if path.is_file():
            yield path


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_destination(pdf_path: Path, base_dir: Path) -> Path:
    ensure_directory(base_dir)
    dated_dir = base_dir / datetime.utcnow().strftime("%Y-%m-%d")
    ensure_directory(dated_dir)
    candidate = dated_dir / pdf_path.name
    counter = 1
    while candidate.exists():
        candidate = dated_dir / f"{pdf_path.stem}-{counter}{pdf_path.suffix}"
        counter += 1
    return candidate


def move_file(pdf_path: Path, destination_root: Path) -> Path:
    destination = _resolve_destination(pdf_path, destination_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.replace(destination)
    return destination


def load_llm_client(manager: LocalModelManager, model_key: str) -> LLMClient:
    config = manager.get(model_key)
    model_name = config.get("model_name")
    base_url = config.get("base_url") or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    logger.info("🔌 Using model=%s (%s)", model_key, model_name)
    return LLMClient(base_url=base_url, model_name=model_name)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Enlitens ingestion pipeline across pending PDFs.")
    parser.add_argument("--model", default="llama", help="Model key defined in config/local_models.yaml.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing PDFs to ingest.")
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR), help="Directory for processed PDFs.")
    parser.add_argument("--failed-dir", default=str(DEFAULT_FAILED_DIR), help="Directory for failed PDFs.")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER), help="JSONL ledger path.")
    parser.add_argument(
        "--ledger-mirror",
        default=str(DEFAULT_LEDGER_MIRROR),
        help="Mirror copy of the JSONL ledger for redundancy.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of PDFs to process (0 = all).")
    parser.add_argument("--force-extraction", action="store_true", help="Recompute Docling cache even if present.")
    parser.add_argument("--skip-gemini", action="store_true", help="Skip Gemini CLI validation.")
    parser.add_argument("--auto-start", action="store_true", help="Start the local model automatically.")
    parser.add_argument("--auto-stop", action="store_true", help="Stop the local model after the run completes.")
    return parser.parse_args(argv)


def configure_logging() -> None:
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


def ingest_batch(args: argparse.Namespace) -> Tuple[int, int]:
    input_dir = Path(args.input_dir)
    processed_dir = Path(args.processed_dir)
    failed_dir = Path(args.failed_dir)
    ledger_path = Path(args.ledger)
    mirror_path = Path(args.ledger_mirror) if args.ledger_mirror else None

    ensure_directory(DEFAULT_INPUT_DIR)
    ensure_directory(processed_dir)
    ensure_directory(failed_dir)
    ensure_directory(ledger_path.parent)

    manager = LocalModelManager()
    postgres_store = PostgresStore()
    vector_mirror = VectorMirror()
    graph_publisher = Neo4jPublisher()

    if args.auto_start:
        manager.start(args.model)

    try:
        llm_client = load_llm_client(manager, args.model)
        success_count = 0
        failure_count = 0

        for idx, pdf_path in enumerate(iter_pdfs(input_dir), start=1):
            if args.limit and idx > args.limit:
                logger.info("Reached limit of %d documents; stopping early.", args.limit)
                break

            logger.info("📄 [%d] Processing %s", idx, pdf_path.name)
            try:
                record = process_pdf_document(
                    pdf_path,
                    llm_client=llm_client,
                    model_key=args.model,
                    force_extraction=args.force_extraction,
                    run_gemini=not args.skip_gemini,
                )
                append_jsonl_record(
                    record,
                    ledger_path=ledger_path,
                    mirror_path=mirror_path,
                )
                if postgres_store.available:
                    try:
                        postgres_store.upsert_record(record)
                    except Exception as db_exc:  # pragma: no cover - defensive
                        logger.error(
                            "⚠️ Postgres persistence failed for %s: %s",
                            record.get("document_id"),
                            db_exc,
                        )
                vector_mirror.mirror(record)
                new_location = move_file(pdf_path, processed_dir)
                logger.info("✅ Stored %s and moved PDF to %s", record["document_id"], new_location)
                graph_publisher.publish_document(record)
                success_count += 1
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("❌ Failed to ingest %s: %s", pdf_path.name, exc)
                moved_path = move_file(pdf_path, failed_dir)
                logger.info("Moved problematic PDF to %s for review.", moved_path)
                failure_count += 1
        return success_count, failure_count
    finally:
        if args.auto_stop:
            manager.stop(args.model)
        graph_publisher.close()


def main(argv: List[str]) -> int:
    configure_logging()
    args = parse_args(argv)
    successes, failures = ingest_batch(args)
    logger.info("🏁 Ingestion complete. %d succeeded, %d failed.", successes, failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

