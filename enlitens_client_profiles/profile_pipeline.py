"""End-to-end pipeline entry point for generating and persisting client profiles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

from .config import ProfilePipelineConfig
from .data_ingestion import IngestionBundle, load_ingestion_bundle
from .orchestrator import PersonaOrchestrator
from .schema import ClientProfileDocument
from .similarity import SIMILARITY_THRESHOLD, SimilarityIndex
from .telemetry import ClientProfileTelemetry


@dataclass
class PipelineResult:
    generated: List[ClientProfileDocument]
    output_dir: Path
    manifest_path: Path


def profile_manifest_path(config: ProfilePipelineConfig) -> Path:
    return config.output_dir / "profiles_manifest.json"


def load_manifest(path: Path) -> List[str]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, profile_ids: Iterable[str]) -> None:
    path.write_text(json.dumps(list(profile_ids), indent=2), encoding="utf-8")


def save_profile(document: ClientProfileDocument, directory: Path) -> Path:
    filename = f"{document.meta.profile_id}.json"
    target = directory / filename
    target.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    return target


def run_profile_pipeline(
    config: ProfilePipelineConfig,
    *,
    desired_profiles: int,
    telemetry: ClientProfileTelemetry,
    allow_duplicates: bool = False,
) -> PipelineResult:
    bundle = load_ingestion_bundle(config)

    orchestrator = PersonaOrchestrator(config)
    manifest_file = profile_manifest_path(config)
    existing_ids = set(load_manifest(manifest_file))

    index = SimilarityIndex(config.cache_dir / "similarity_index.json")
    existing_documents: List[ClientProfileDocument] = []
    for profile_id in existing_ids:
        profile_path = config.output_dir / f"{profile_id}.json"
        if not profile_path.exists():
            continue
        try:
            document = ClientProfileDocument.model_validate_json(profile_path.read_text(encoding="utf-8"))
            existing_documents.append(document)
        except Exception:
            continue
    index.register_existing_if_needed(existing_documents)

    generated_documents: List[ClientProfileDocument] = []
    telemetry.log_event("profiles_pipeline_started", {"existing_profiles": len(existing_ids)})

    orchestrator.prepare_context(bundle)

    for _ in range(desired_profiles):
        document, knowledge_context, foundation, research = orchestrator.assemble_persona(bundle)

        if document is None:
            telemetry.log_event(
                "persona_research_missing",
                {"reason": "no_results", "queries": research.queries, "gaps": foundation.gaps},
            )
            continue

        if document.meta.profile_id in existing_ids:
            telemetry.log_event("profile_duplicate_skipped", {"profile_id": document.meta.profile_id})
            continue
        similarity_report = index.evaluate(document)
        if not allow_duplicates and similarity_report.exceeds_threshold():
            conflict_dir = config.cache_dir / "conflicts"
            conflict_dir.mkdir(parents=True, exist_ok=True)
            conflict_path = conflict_dir / f"{document.meta.profile_id}.json"
            conflict_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
            telemetry.log_event(
                "profile_similarity_flagged",
                {
                    "profile_id": document.meta.profile_id,
                    "match_profile_id": similarity_report.profile_id,
                    "cosine": round(similarity_report.cosine, 4),
                    "jaccard": round(similarity_report.jaccard, 4),
                    "threshold": SIMILARITY_THRESHOLD,
                },
            )
            continue
        elif allow_duplicates and similarity_report.exceeds_threshold():
            telemetry.log_event(
                "profile_similarity_allowed",
                {
                    "profile_id": document.meta.profile_id,
                    "match_profile_id": similarity_report.profile_id,
                    "cosine": round(similarity_report.cosine, 4),
                    "jaccard": round(similarity_report.jaccard, 4),
                    "threshold": SIMILARITY_THRESHOLD,
                },
            )
        save_profile(document, config.output_dir)
        existing_ids.add(document.meta.profile_id)
        generated_documents.append(document)
        index.register(document)
        telemetry.log_profile_created(document)
        telemetry.log_event(
            "persona_agents_completed",
            {
                "profile_id": document.meta.profile_id,
                "research_queries": research.queries,
                "foundation_gaps": foundation.gaps,
                "analytics_available": bool(bundle.analytics),
                "top_keywords": [keyword for keyword, _ in knowledge_context.top_keywords[:10]],
                "brand_site_pages": len(knowledge_context.site_documents),
                "brand_mentions": len(knowledge_context.brand_mentions),
            },
        )

    save_manifest(manifest_file, existing_ids)
    telemetry.log_event(
        "profiles_pipeline_completed",
        {
            "generated": len(generated_documents),
            "total_profiles": len(existing_ids),
            "output_dir": str(config.output_dir),
        },
    )

    return PipelineResult(
        generated=generated_documents,
        output_dir=config.output_dir,
        manifest_path=manifest_file,
    )

