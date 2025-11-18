"""
Topic Alignment Utilities
-------------------------

Derives a bridge between highly specific research focus areas and Enlitens'
client personas.  The goal is to avoid false negatives when a paper speaks to
mechanisms (e.g. inflammation, dopaminergic signalling) that impact our
neurodivergent clients even if the paper never mentions "autism" or "ADHD"
explicitly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


# Maps research topic keywords to persona themes we care about.
TOPIC_BRIDGES: Dict[str, Sequence[str]] = {
    # Neurobiology
    "dopamine": ("adhd self-regulation", "motivational scaffolding"),
    "prefrontal": ("executive function load", "adhd burnout"),
    "executive function": ("executive function load", "adhd burnout"),
    "frontostriatal": ("adhd self-regulation",),
    "neuroplastic": ("brain-based interventions", "therapy that teaches new wiring"),
    "neuroinflammation": ("autistic burnout", "sensory overwhelm"),
    "inflammation": ("adhd burnout", "trauma-driven allostatic load"),
    "cortisol": ("cptsd hypervigilance", "stress pile-up"),
    "allostatic": ("chronic stress adaptation",),
    "epigenetic": ("intergenerational trauma", "environment-driven brain shifts"),
    "oxidative": ("sensory overwhelm recovery",),
    "microglia": ("brain-body load",),
    # Social context
    "loneliness": ("masking exhaustion", "social burnout"),
    "social exclusion": ("autistic masking", "rejection sensitivity dysphoria"),
    "belonging": ("community co-regulation",),
    "stigma": ("systemic trauma", "diagnostic gaslighting"),
    # Sleep / circadian
    "circadian": ("sleep dysregulation", "adhd nervous system"),
    "melatonin": ("sleep dysregulation",),
    # Sensory + nervous system
    "sensory": ("sensory modulation",),
    "autonomic": ("nervous system regulation", "polyvagal work"),
    "parasympathetic": ("nervous system regulation",),
    # Comorbid traits commonly seen in our personas
    "anxiety": ("anxiety loops", "emotional regulation"),
    "depression": ("shutdown response",),
    "ptsd": ("complex trauma", "hypervigilance"),
    "trauma": ("complex trauma", "hypervigilance"),
    "masking": ("masking exhaustion",),
    "burnout": ("autistic burnout", "adhd burnout"),
    # Hormonal / endocrine crossovers
    "hpa axis": ("stress hormone dysregulation",),
    "estradiol": ("cycle-aware planning",),
}

DIRECT_DIAGNOSES = {
    "adhd",
    "attention-deficit",
    "autism",
    "autistic",
    "asd",
    "asperger",
    "dyslexia",
    "anxiety",
    "ptsd",
    "trauma",
    "neurodivergent",
}

TOKEN_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)


def _normalise_terms(terms: Iterable[str]) -> Set[str]:
    normalised: Set[str] = set()
    for term in terms:
        if not term:
            continue
        lowered = term.lower()
        pieces = TOKEN_SPLIT_RE.split(lowered)
        pieces = [piece for piece in pieces if piece]
        if not pieces:
            continue
        normalised.update(pieces)
        normalised.add(lowered.strip())
    return normalised


def _flatten_entities(entities: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for bucket in entities.values():
        if isinstance(bucket, list):
            for item in bucket:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("label") or item.get("name")
                    if text:
                        values.append(str(text))
                elif isinstance(item, str):
                    values.append(item)
        elif isinstance(bucket, dict):
            text = bucket.get("text")
            if text:
                values.append(str(text))
    return values


@dataclass(slots=True)
class TopicAlignment:
    primary_topics: List[str]
    related_persona_themes: List[str]
    alignment_note: str
    alignment_confidence: str  # "direct", "adjacent", "weak"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_topics": self.primary_topics,
            "related_persona_themes": self.related_persona_themes,
            "alignment_note": self.alignment_note,
            "alignment_confidence": self.alignment_confidence,
        }


class TopicAlignmentBuilder:
    """
    Derive a contextual mapping between a research paper's topics and Enlitens'
    persona library.
    """

    def build(
        self,
        *,
        paper_profile: Optional[Dict[str, Any]],
        entities: Optional[Dict[str, Any]],
    ) -> Optional[TopicAlignment]:
        if not paper_profile and not entities:
            return None

        top_terms = paper_profile.get("top_terms", []) if paper_profile else []
        lead_sentences = paper_profile.get("lead_sentences", "") if paper_profile else ""
        entity_terms = _flatten_entities(entities or {})

        vocabulary = _normalise_terms(top_terms + entity_terms + [lead_sentences])
        if not vocabulary:
            return None

        primary_topics = sorted({term for term in vocabulary if len(term) > 2})[:30]

        related_themes: Set[str] = set()
        direct_hits = False
        for token in vocabulary:
            if token in DIRECT_DIAGNOSES:
                direct_hits = True
                related_themes.add(token)
            for keyword, bridges in TOPIC_BRIDGES.items():
                if keyword in token:
                    related_themes.update(bridges)

        if not related_themes and not direct_hits:
            # Provide a default explanation referencing nervous system adaptation.
            related_themes = {
                "nervous system regulation",
                "brain-based accommodation planning",
            }

        alignment_confidence = "direct" if direct_hits else ("adjacent" if related_themes else "weak")

        alignment_note = (
            "This paper focuses on "
            f"{', '.join(primary_topics[:6]) or 'specialised mechanisms'}."
        )
        if direct_hits:
            alignment_note += (
                " It explicitly references neurodivergent diagnoses that overlap with our personas, so match them directly."
            )
        else:
            alignment_note += (
                " Even though it may not say 'autism' or 'ADHD' outright, these mechanisms drive lived issues like "
                f"{', '.join(sorted(related_themes))}. Build personas and briefs that make those links explicit."
            )

        return TopicAlignment(
            primary_topics=primary_topics[:15],
            related_persona_themes=sorted(related_themes)[:15],
            alignment_note=alignment_note,
            alignment_confidence=alignment_confidence,
        )


__all__ = ["TopicAlignment", "TopicAlignmentBuilder"]


