"""Shared workflow state for the LangGraph supervisor pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Set, Annotated

from typing_extensions import TypedDict


def _keep_last_value(existing: Any, new: Any) -> Any:
    """Reducer function that keeps only the last value written to a channel.

    This allows multiple nodes to write to the same channel (like 'stage')
    without triggering InvalidUpdateError. The last value wins.
    """
    return new


def _merge_intermediate_results(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer function that merges intermediate results dictionaries.

    This allows multiple agents to contribute to intermediate_results
    without triggering InvalidUpdateError. New values override existing ones.
    """
    if existing is None:
        return new or {}
    if new is None:
        return existing
    return {**existing, **new}


def _merge_completed_nodes(existing: Dict[str, str], new: Dict[str, str]) -> Dict[str, str]:
    """Reducer function that merges completed nodes dictionaries.

    This allows multiple agents to mark themselves as completed
    without triggering InvalidUpdateError. New values override existing ones.
    """
    if existing is None:
        return new or {}
    if new is None:
        return existing
    return {**existing, **new}


def _merge_attempt_counters(existing: Dict[str, int], new: Dict[str, int]) -> Dict[str, int]:
    """Reducer function that merges attempt counters dictionaries."""
    if existing is None:
        return new or {}
    if new is None:
        return existing
    return {**existing, **new}


def _merge_errors(existing: Dict[str, str], new: Dict[str, str]) -> Dict[str, str]:
    """Reducer function that merges errors dictionaries."""
    if existing is None:
        return new or {}
    if new is None:
        return existing
    return {**existing, **new}


def _merge_sets(existing: Set[str], new: Set[str]) -> Set[str]:
    """Reducer function that merges sets."""
    if existing is None:
        return new or set()
    if new is None:
        return existing
    return existing | new


def _merge_metadata(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer that merges metadata dictionaries across node writes."""
    if existing is None:
        return new or {}
    if new is None:
        return existing
    merged = dict(existing)
    merged.update(new)
    return merged


class WorkflowState(TypedDict, total=False):
    """TypedDict holding the shared workflow state across nodes.

    Note: Changed from Pydantic BaseModel to TypedDict to support LangGraph's
    Annotated reducer pattern for handling multiple writes per step.

    The 'stage' field uses Annotated with a reducer to allow multiple agents
    to update it in the same step without raising InvalidUpdateError.
    """

    # Required fields
    document_id: str
    document_text: str

    # Optional input context
    doc_type: Optional[str]
    client_insights: Optional[Dict[str, Any]]
    founder_insights: Optional[Dict[str, Any]]
    st_louis_context: Optional[Dict[str, Any]]
    raw_client_context: Optional[str]
    raw_founder_context: Optional[str]

    # Shared orchestration metadata - use Annotated to allow multiple updates
    stage: Annotated[str, _keep_last_value]
    skip_nodes: Annotated[Set[str], _merge_sets]
    completed_nodes: Annotated[Dict[str, str], _merge_completed_nodes]
    attempt_counters: Annotated[Dict[str, int], _merge_attempt_counters]
    errors: Annotated[Dict[str, str], _merge_errors]
    intermediate_results: Annotated[Dict[str, Any], _merge_intermediate_results]
    cache_prefix: str
    cache_chunk_id: str
    metadata: Annotated[Dict[str, Any], _merge_metadata]

    # Node outputs tracked explicitly
    science_result: Optional[Dict[str, Any]]
    context_result: Optional[Dict[str, Any]]
    client_profile_result: Optional[Dict[str, Any]]
    clinical_result: Optional[Dict[str, Any]]
    educational_result: Optional[Dict[str, Any]]
    rebellion_result: Optional[Dict[str, Any]]
    founder_voice_result: Optional[Dict[str, Any]]
    marketing_result: Optional[Dict[str, Any]]
    validation_result: Optional[Dict[str, Any]]

    # Runtime bookkeeping
    start_timestamp: Optional[datetime]
    end_timestamp: Optional[datetime]
    marketing_completed: bool
    validation_completed: bool


def create_initial_state(
    document_id: str,
    document_text: str,
    doc_type: Optional[str] = None,
    client_insights: Optional[Dict[str, Any]] = None,
    founder_insights: Optional[Dict[str, Any]] = None,
    st_louis_context: Optional[Dict[str, Any]] = None,
    raw_client_context: Optional[str] = None,
    raw_founder_context: Optional[str] = None,
    cache_prefix: Optional[str] = None,
    cache_chunk_id: Optional[str] = None,
) -> WorkflowState:
    """Create a properly initialized workflow state."""
    return WorkflowState(
        document_id=document_id,
        document_text=document_text,
        doc_type=doc_type,
        client_insights=client_insights,
        founder_insights=founder_insights,
        st_louis_context=st_louis_context,
        raw_client_context=raw_client_context,
        raw_founder_context=raw_founder_context,
        stage="initial",
        skip_nodes=set(),
        completed_nodes={},
        attempt_counters={},
        errors={},
        intermediate_results={},
        cache_prefix=cache_prefix or document_id,
        cache_chunk_id=cache_chunk_id or f"{document_id}:root",
        metadata={},
        science_result=None,
        context_result=None,
        client_profile_result=None,
        clinical_result=None,
        educational_result=None,
        rebellion_result=None,
        founder_voice_result=None,
        marketing_result=None,
        validation_result=None,
        start_timestamp=None,
        end_timestamp=None,
        marketing_completed=False,
        validation_completed=False,
    )


def mark_completed(state: WorkflowState, node_name: str, status: str = "done") -> None:
    """Mark a node as completed in the workflow state."""
    state["completed_nodes"][node_name] = status


def record_attempt(state: WorkflowState, node_name: str) -> int:
    """Record an attempt for a node and return the total count."""
    attempts = state["attempt_counters"].get(node_name, 0) + 1
    state["attempt_counters"][node_name] = attempts
    return attempts


def as_dict(state: WorkflowState) -> Dict[str, Any]:
    """Convert state to a plain dict, with sets converted to lists for JSON compatibility."""
    data = dict(state)
    data["skip_nodes"] = list(data.get("skip_nodes", set()))
    return data
