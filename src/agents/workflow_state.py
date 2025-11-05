"""Shared workflow state for the LangGraph supervisor pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Set, Annotated, List, Tuple

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


class LocalityMatchRecord(TypedDict, total=False):
    """Aggregated locality match enriched with reference metadata."""

    name: str
    locality: Dict[str, Any]
    document_mentions: int
    intake_mentions: int
    transcript_mentions: int
    health_report_mentions: int
    signal_strength: float


class LocalityGapRecord(TypedDict, total=False):
    """Locality signal without a supporting reference entry."""

    name: str
    document_mentions: int
    intake_mentions: int
    transcript_mentions: int
    health_report_mentions: int
    signal_strength: float


class LocalityRecords(TypedDict, total=False):
    """Structured locality context used by downstream agents."""

    matches: List[LocalityMatchRecord]
    gaps: List[LocalityGapRecord]


class ThemeSourceBreakdown(TypedDict, total=False):
    """Per-source contribution for a detected theme."""

    weighted_frequency: float
    frequency: float


class DominantThemeRecord(TypedDict, total=False):
    """Theme enriched with weights and locality tags."""

    theme: str
    total_weight: float
    source_breakdown: Dict[str, ThemeSourceBreakdown]
    locality_tags: List[Tuple[str, float]]


class ThemeGapRecord(TypedDict, total=False):
    """Theme that is missing signals from some context assets."""

    theme: str
    present_sources: List[str]
    missing_sources: List[str]
    total_weight: float
    locality_tags: List[Tuple[str, float]]


class SocioeconomicContrastFlag(TypedDict, total=False):
    """Theme with divergent socioeconomic indicators across localities."""

    theme: str
    indicators: List[str]
    estimated_income_gap: Optional[float]
    locality_profiles: List[Dict[str, Any]]


class WorkflowState(TypedDict, total=False):
    """TypedDict holding the shared workflow state across nodes.

    Note: Changed from Pydantic BaseModel to TypedDict to support LangGraph's
    Annotated reducer pattern for handling multiple writes per step.

    The 'stage' field uses Annotated with a reducer to allow multiple agents
    to update it in the same step without raising InvalidUpdateError.

    Locality/theme analytics fields expose the following schema:

    * ``locality_records`` holds ``{"matches": [...], "gaps": [...]}`` where
      ``matches`` provide :class:`LocalityMatchRecord` payloads enriched with
      reference atlas metadata and ``gaps`` provide :class:`LocalityGapRecord`
      entries highlighting unreferenced but high-signal localities.
    * ``dominant_themes`` is a list of :class:`DominantThemeRecord` objects and
      surfaces weighted signals by context source plus locality tags.
    * ``theme_gaps`` mirrors ``dominant_themes`` but only includes themes that
      are missing at least one source contribution. Each element is a
      :class:`ThemeGapRecord`.
    * ``socioeconomic_contrast_flags`` collects
      :class:`SocioeconomicContrastFlag` records that detail the socioeconomic
      tension observed between the highlighted localities for a theme.
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
    document_passages: Optional[List[Dict[str, Any]]]
    intake_registry: Optional[Dict[str, Any]]
    transcript_registry: Optional[Dict[str, Any]]
    regional_atlas: Optional[Dict[str, Any]]
    health_report_summary: Optional[Dict[str, Any]]
    document_locality_matches: Optional[List[Tuple[str, int]]]
    locality_records: Optional[LocalityRecords]
    dominant_themes: Optional[List[DominantThemeRecord]]
    theme_gaps: Optional[List[ThemeGapRecord]]
    socioeconomic_contrast_flags: Optional[List[SocioeconomicContrastFlag]]

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
    document_passages: Optional[List[Dict[str, Any]]] = None,
    intake_registry: Optional[Dict[str, Any]] = None,
    transcript_registry: Optional[Dict[str, Any]] = None,
    regional_atlas: Optional[Dict[str, Any]] = None,
    health_report_summary: Optional[Dict[str, Any]] = None,
    document_locality_matches: Optional[List[Tuple[str, int]]] = None,
    locality_records: Optional[LocalityRecords] = None,
    dominant_themes: Optional[List[DominantThemeRecord]] = None,
    theme_gaps: Optional[List[ThemeGapRecord]] = None,
    socioeconomic_contrast_flags: Optional[List[SocioeconomicContrastFlag]] = None,
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
        document_passages=document_passages,
        intake_registry=intake_registry,
        transcript_registry=transcript_registry,
        regional_atlas=regional_atlas,
        health_report_summary=health_report_summary,
        document_locality_matches=document_locality_matches,
        locality_records=locality_records,
        dominant_themes=dominant_themes,
        theme_gaps=theme_gaps,
        socioeconomic_contrast_flags=socioeconomic_contrast_flags,
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

    def _normalize_pairs(
        entries: Optional[List[Any]], value_key: str
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if not isinstance(entries, list):
            return normalized
        for entry in entries:
            if isinstance(entry, dict):
                normalized.append(dict(entry))
                continue
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                name, score = entry
                normalized.append({"name": name, value_key: score})
        return normalized

    if "document_locality_matches" in data:
        data["document_locality_matches"] = _normalize_pairs(
            data.get("document_locality_matches"), "mentions"
        )

    if "locality_records" in data and isinstance(data["locality_records"], dict):
        serialized_records: Dict[str, List[Dict[str, Any]]] = {}
        for bucket, entries in data["locality_records"].items():
            if not isinstance(entries, list):
                continue
            serialized_records[bucket] = [
                dict(entry) for entry in entries if isinstance(entry, dict)
            ]
        data["locality_records"] = serialized_records

    def _normalize_theme_entries(entries: Optional[List[Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if not isinstance(entries, list):
            return normalized
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            item = dict(entry)
            item["locality_tags"] = _normalize_pairs(
                item.get("locality_tags"), "weight"
            )
            normalized.append(item)
        return normalized

    if "dominant_themes" in data:
        data["dominant_themes"] = _normalize_theme_entries(data.get("dominant_themes"))

    if "theme_gaps" in data:
        data["theme_gaps"] = _normalize_theme_entries(data.get("theme_gaps"))

    if "socioeconomic_contrast_flags" in data and isinstance(
        data["socioeconomic_contrast_flags"], list
    ):
        normalized_flags: List[Dict[str, Any]] = []
        for entry in data["socioeconomic_contrast_flags"]:
            if isinstance(entry, dict):
                normalized_flags.append(dict(entry))
        data["socioeconomic_contrast_flags"] = normalized_flags

    return data
