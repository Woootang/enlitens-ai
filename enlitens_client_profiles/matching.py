"""Matching and context utilities for persona-aware workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .schema import ClientProfileDocument
from .similarity import build_corpus, embed_text, _jaccard_similarity


def load_persona_library(directory: Path) -> List[ClientProfileDocument]:
    """Load all persona JSON files from a directory."""

    personas: List[ClientProfileDocument] = []
    if not directory.exists():
        return personas

    for path in sorted(directory.glob("*.json")):
        try:
            raw = path.read_text(encoding="utf-8")
            personas.append(ClientProfileDocument.model_validate_json(raw))
        except Exception:
            continue
    return personas


def score_persona_match(
    persona: ClientProfileDocument,
    *,
    narrative_text: str,
    attribute_tags: Sequence[str],
) -> float:
    """Compute a blended similarity score between a persona and candidate signals."""

    persona_vector = embed_text(build_corpus(persona))
    candidate_vector = embed_text(narrative_text)
    cosine = float(np.dot(persona_vector, candidate_vector))
    jaccard = _jaccard_similarity(attribute_tags, persona.attribute_set())
    # Weighted blend emphasises linguistic alignment while respecting attribute tags
    return 0.7 * cosine + 0.3 * jaccard


def match_personas(
    personas: Iterable[ClientProfileDocument],
    *,
    narrative_text: str,
    attribute_tags: Sequence[str],
    top_k: int = 3,
) -> List[Tuple[ClientProfileDocument, float]]:
    """Return top-k persona matches given intake narrative and attribute hints."""

    scores: List[Tuple[ClientProfileDocument, float]] = []
    for persona in personas:
        score = score_persona_match(persona, narrative_text=narrative_text, attribute_tags=attribute_tags)
        scores.append((persona, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]


def build_ai_context(persona: ClientProfileDocument) -> Dict[str, str]:
    """Create a compact context payload suitable for AI prompts and assessments."""

    return {
        "persona_name": persona.meta.persona_name,
        "persona_tagline": persona.meta.persona_tagline or "",
        "attribute_tags": ", ".join(persona.meta.attribute_tags),
        "demographics": f"{persona.demographics.age_range or ''} · {persona.demographics.locality or ''} · {persona.demographics.occupation or ''}",
        "neurodivergence": ", ".join(persona.neurodivergence_profile.identities),
        "nervous_system_pattern": persona.clinical_challenges.nervous_system_pattern,
        "goals": "; ".join(persona.goals_motivations.therapy_goals[:3]),
        "pain_points": "; ".join((persona.pain_points_barriers.internal + persona.pain_points_barriers.systemic)[:3]),
        "strengths": "; ".join(persona.adaptive_strengths.strengths[:3]),
        "narrative": persona.narrative.liz_voice,
        "quotes": " | ".join([persona.quotes.struggle, persona.quotes.hope] + persona.quotes.additional[:1]),
    }


