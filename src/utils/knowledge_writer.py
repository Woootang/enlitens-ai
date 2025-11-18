"""
Utilities for persisting science-only knowledge base records.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.models.enlitens_schemas import ScienceExtractionRecord

logger = logging.getLogger(__name__)


class ScienceKnowledgeWriter:
    """Append-only writer for the science-first knowledge base."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or "data/knowledge_base")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.entries_path = self.base_dir / "science_entries.jsonl"
        self.manifest_path = self.base_dir / "science_manifest.json"
        self.text_dir = self.base_dir / "text"
        self.text_dir.mkdir(parents=True, exist_ok=True)

    def append_record(self, record: ScienceExtractionRecord) -> int:
        """Append a science record if content changed. Returns the revision number."""
        manifest = self._load_manifest()
        doc_id = record.metadata.document_id
        manifest_entry = manifest.get(doc_id)

        if (
            manifest_entry
            and manifest_entry.get("text_sha256") == record.metadata.text_sha256
            and manifest_entry.get("source_sha256") == record.metadata.source_sha256
        ):
            record.metadata.revision = manifest_entry.get("revision", 1)
            logger.info(
                "🗂️ Science KB already up to date for %s (rev %s)",
                doc_id,
                record.metadata.revision,
            )
            return record.metadata.revision

        revision = (manifest_entry or {}).get("revision", 0) + 1
        record.metadata.revision = revision

        with self.entries_path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json(exclude_none=True))
            handle.write("\n")

        manifest[doc_id] = {
            "title": record.metadata.title,
            "revision": revision,
            "ingested_at": record.metadata.ingestion_timestamp.isoformat(),
            "source_filename": record.metadata.source_filename,
            "source_sha256": record.metadata.source_sha256,
            "text_sha256": record.metadata.text_sha256,
            "keywords": record.metadata.keywords,
            "pipeline_mode": record.metadata.pipeline_mode,
            "source_text_path": record.source_text_path,
        }
        self._write_manifest(manifest)
        logger.info("📚 Wrote science KB record %s (rev %s)", doc_id, revision)
        return revision

    def write_text_blob(self, document_id: str, text: str) -> Optional[Path]:
        """Persist the extracted markdown for reuse by downstream agents."""
        if not text:
            return None
        text_path = self.text_dir / f"{document_id}.md"
        text_path.write_text(text, encoding="utf-8")
        return text_path

    def read_manifest(self) -> Dict[str, Any]:
        """Return the manifest without modifying it."""
        return self._load_manifest()

    def load_records(self, limit: Optional[int] = None) -> List[ScienceExtractionRecord]:
        """Stream science records back into memory (best-effort)."""
        if not self.entries_path.exists():
            return []
        records: List[ScienceExtractionRecord] = []
        with self.entries_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    records.append(ScienceExtractionRecord.model_validate(payload))
                except Exception as exc:
                    logger.warning("⚠️ Failed to read record at line %d: %s", idx + 1, exc)
                if limit and len(records) >= limit:
                    break
        return records

    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError as exc:
            logger.warning("⚠️ Manifest JSON corrupted (%s); starting fresh", exc)
            return {}

    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        tmp_path = self.manifest_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        tmp_path.replace(self.manifest_path)


