"""Foundation builder agent that shapes persona scaffolding from the knowledge graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .data_ingestion import IngestionBundle
from .knowledge_keeper import KnowledgeGraphContext

logger = logging.getLogger(__name__)


SUSPECT_FAMILY_KEYWORDS = {"mom", "dad", "parent", "kid", "son", "daughter", "partner", "husband", "wife"}
SUSPECT_JOB_KEYWORDS = {"manager", "teacher", "engineer", "nurse", "therapist", "consultant", "director", "analyst"}


@dataclass(slots=True)
class PersonaFoundation:
    demographics: Dict[str, str]
    family_clues: List[str]
    occupation_clues: List[str]
    locality_hypotheses: List[str]
    search_signals: List[str]
    gaps: List[str] = field(default_factory=list)


class FoundationBuilderAgent:
    """Derive a structured scaffold for downstream research and writing."""

    def build(self, bundle: IngestionBundle, context: KnowledgeGraphContext) -> PersonaFoundation:
        demographics = {
            "age_range": "[needs_research]",
            "gender": "[needs_research]",
            "pronouns": "[needs_research]",
            "orientation": "[needs_research]",
            "ethnicity": "[needs_research]",
            "family_status": "[needs_research]",
            "occupation": "[needs_research]",
            "education": "[needs_research]",
            "locality": "[needs_research]",
        }

        family_clues = self._collect_keyword_sentences(bundle, SUSPECT_FAMILY_KEYWORDS)
        occupation_clues = self._collect_keyword_sentences(bundle, SUSPECT_JOB_KEYWORDS)

        locality_hypotheses = [
            f"{name} ({count} intake refs)"
            for name, count in sorted(context.locality_counts.items(), key=lambda item: -item[1])[:6]
        ]

        search_signals: List[str] = []
        if context.analytics_summary:
            for line in context.analytics_summary.splitlines():
                clean = line.strip("- ").strip()
                if clean:
                    search_signals.append(clean)

        gaps = [
            "Age range inferred from intake tone (needs verification)",
            "Confirm family system details (household members, caregiving roles)",
            "Validate commute + locality from research",
            "Enrich occupation context with market data",
        ]
        if not search_signals:
            gaps.append("Surface current search demand / GA insights")

        return PersonaFoundation(
            demographics=demographics,
            family_clues=family_clues[:6],
            occupation_clues=occupation_clues[:6],
            locality_hypotheses=locality_hypotheses,
            search_signals=search_signals[:15],
            gaps=gaps,
        )

    def _collect_keyword_sentences(self, bundle: IngestionBundle, keywords: Sequence[str]) -> List[str]:
        matches: List[str] = []
        lowered_keywords = [kw.lower() for kw in keywords]
        for record in bundle.intakes:
            text_lower = record.raw_text.lower()
            if any(kw in text_lower for kw in lowered_keywords):
                matches.append(record.raw_text.strip())
        return matches


