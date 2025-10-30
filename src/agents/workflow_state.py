"""Shared workflow state for the LangGraph supervisor pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Set

from pydantic import BaseModel, Field, ConfigDict


class WorkflowState(BaseModel):
    """Pydantic model holding the shared workflow state across nodes."""

    document_id: str
    document_text: str
    doc_type: Optional[str] = None
    client_insights: Optional[Dict[str, Any]] = None
    founder_insights: Optional[Dict[str, Any]] = None
    st_louis_context: Optional[Dict[str, Any]] = None

    # Shared orchestration metadata
    stage: str = "initial"
    skip_nodes: Set[str] = Field(default_factory=set)
    completed_nodes: Dict[str, str] = Field(default_factory=dict)
    attempt_counters: Dict[str, int] = Field(default_factory=dict)
    errors: Dict[str, str] = Field(default_factory=dict)
    intermediate_results: Dict[str, Any] = Field(default_factory=dict)
    cache_prefix: str = Field(default_factory=lambda: "workflow")
    cache_chunk_id: str = Field(default_factory=lambda: "root")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Node outputs tracked explicitly
    science_result: Optional[Dict[str, Any]] = None
    context_result: Optional[Dict[str, Any]] = None
    clinical_result: Optional[Dict[str, Any]] = None
    educational_result: Optional[Dict[str, Any]] = None
    rebellion_result: Optional[Dict[str, Any]] = None
    founder_voice_result: Optional[Dict[str, Any]] = None
    marketing_result: Optional[Dict[str, Any]] = None
    validation_result: Optional[Dict[str, Any]] = None

    # Runtime bookkeeping
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    marketing_completed: bool = False
    validation_completed: bool = False

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        use_enum_values=True,
    )

    def mark_completed(self, node_name: str, status: str = "done") -> None:
        self.completed_nodes[node_name] = status

    def record_attempt(self, node_name: str) -> int:
        attempts = self.attempt_counters.get(node_name, 0) + 1
        self.attempt_counters[node_name] = attempts
        return attempts

    def as_dict(self) -> Dict[str, Any]:
        data = self.model_dump()
        # Convert sets to lists for JSON compatibility
        data["skip_nodes"] = list(self.skip_nodes)
        return data
