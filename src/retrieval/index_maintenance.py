"""Index maintenance helpers for scheduled refreshes and integrity checks."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional, Sequence

from src.models.enlitens_schemas import EnlitensKnowledgeEntry

from .embedding_ingestion import EmbeddingIngestionPipeline, IngestionStats, IntegrityReport

MaintenanceSchedule = Literal["nightly", "weekly"]


@dataclass
class RefreshReport:
    """Summary of an index refresh operation."""

    schedule: MaintenanceSchedule
    started_at: datetime
    completed_at: datetime
    documents_processed: int
    total_chunks: int
    ingest_stats: List[IngestionStats]


class IndexMaintenance:
    """Coordinate scheduled index refreshes and integrity checks."""

    def __init__(self, pipeline: Optional[EmbeddingIngestionPipeline] = None) -> None:
        self.pipeline = pipeline or EmbeddingIngestionPipeline()

    def refresh(
        self,
        entries: Sequence[EnlitensKnowledgeEntry],
        schedule: MaintenanceSchedule = "nightly",
        rebuild: bool = False,
    ) -> RefreshReport:
        start_time = datetime.utcnow()
        stats = [
            self.pipeline.ingest_entry_with_rebuild(entry)
            if rebuild
            else self.pipeline.ingest_entry(entry)
            for entry in entries
        ]
        end_time = datetime.utcnow()

        return RefreshReport(
            schedule=schedule,
            started_at=start_time,
            completed_at=end_time,
            documents_processed=len(stats),
            total_chunks=sum(stat.chunks_ingested for stat in stats),
            ingest_stats=stats,
        )

    def run_integrity_check(
        self,
        entries: Sequence[EnlitensKnowledgeEntry],
    ) -> IntegrityReport:
        return self.pipeline.run_integrity_check(entries)

    @staticmethod
    def schedule_window(schedule: MaintenanceSchedule, reference: Optional[datetime] = None) -> Dict[str, datetime]:
        reference = reference or datetime.utcnow()
        if schedule == "weekly":
            start = reference - timedelta(days=7)
        else:
            start = reference - timedelta(days=1)
        return {"start": start, "end": reference}
