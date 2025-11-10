"""Similarity and deduplication utilities for persona documents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .schema import ClientProfileDocument

WORD_RE = re.compile(r"[a-z0-9']+")
SIMILARITY_THRESHOLD = 0.41


def _tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def _jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


def build_corpus(document: ClientProfileDocument) -> str:
    """Generate a corpus string representing the persona."""

    sections: List[str] = []
    sections.extend(document.goals_motivations.therapy_goals)
    sections.extend(document.goals_motivations.life_goals)
    sections.append(document.narrative.liz_voice)
    sections.extend(document.marketing_copy.website_about)
    sections.extend(document.marketing_copy.landing_page_intro)
    sections.extend(document.marketing_copy.email_nurture)
    sections.extend(document.marketing_copy.social_snippets)
    sections.extend(document.adaptive_strengths.strengths)
    sections.extend(document.pain_points_barriers.internal)
    sections.extend(document.pain_points_barriers.systemic)
    sections.extend(document.pain_points_barriers.access)
    sections.extend(document.support_system.supportive_allies)
    sections.extend(document.support_system.gaps)
    sections.append(document.local_environment.home_environment or "")
    sections.append(document.local_environment.work_environment or "")
    sections.append(document.local_environment.commute or "")
    sections.extend(document.clinical_challenges.presenting_issues)
    sections.append(document.clinical_challenges.nervous_system_pattern)
    sections.extend(document.sensory_profile.sensitivities)
    sections.extend(document.sensory_profile.seeking_behaviors)
    sections.extend(document.sensory_profile.regulation_methods)
    sections.append(document.quotes.struggle)
    sections.append(document.quotes.hope)
    sections.extend(document.quotes.additional)

    return "\n".join(filter(None, sections))


@dataclass
class SimilarityReport:
    profile_id: Optional[str]
    cosine: float
    jaccard: float

    def exceeds_threshold(self) -> bool:
        return max(self.cosine, self.jaccard) >= SIMILARITY_THRESHOLD


class SimilarityIndex:
    """Persistent index of persona vectors to enforce uniqueness."""

    _embedding_model: Optional[SentenceTransformer] = None
    _embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.vectors: Dict[str, List[float]] = {}
        self.attribute_sets: Dict[str, List[str]] = {}
        self._load()

    @classmethod
    def _load_model(cls) -> SentenceTransformer:
        if cls._embedding_model is None:
            cls._embedding_model = SentenceTransformer(cls._embedding_model_name, device="cpu")
        return cls._embedding_model

    @classmethod
    def _embed(cls, text: str) -> np.ndarray:
        model = cls._load_model()
        embedding = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return embedding[0]

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            self.vectors = {k: list(map(float, v)) for k, v in payload.get("vectors", {}).items()}
            self.attribute_sets = {k: list(v) for k, v in payload.get("attributes", {}).items()}
        except Exception:
            self.vectors = {}
            self.attribute_sets = {}

    def _persist(self) -> None:
        payload = {
            "vectors": self.vectors,
            "attributes": self.attribute_sets,
        }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def has(self, profile_id: str) -> bool:
        return profile_id in self.vectors

    def register(self, document: ClientProfileDocument, persist: bool = True) -> None:
        corpus = build_corpus(document)
        vector = self._embed(corpus)
        self.vectors[document.meta.profile_id] = vector.tolist()
        self.attribute_sets[document.meta.profile_id] = document.attribute_set()
        if persist:
            self._persist()

    def evaluate(self, document: ClientProfileDocument) -> SimilarityReport:
        if not self.vectors:
            return SimilarityReport(profile_id=None, cosine=0.0, jaccard=0.0)

        candidate_vector = self._embed(build_corpus(document))
        candidate_attributes = document.attribute_set()

        best_cosine = 0.0
        best_jaccard = 0.0
        best_profile: Optional[str] = None

        for profile_id, vector in self.vectors.items():
            stored = np.asarray(vector, dtype=np.float32)
            cosine = float(np.dot(candidate_vector, stored))
            jaccard = _jaccard_similarity(candidate_attributes, self.attribute_sets.get(profile_id, []))
            score = max(cosine, jaccard)
            if score > max(best_cosine, best_jaccard):
                best_cosine = cosine
                best_jaccard = jaccard
                best_profile = profile_id

        return SimilarityReport(profile_id=best_profile, cosine=best_cosine, jaccard=best_jaccard)

    def register_existing_if_needed(self, documents: Iterable[ClientProfileDocument]) -> None:
        updated = False
        for document in documents:
            if not self.has(document.meta.profile_id):
                self.register(document, persist=False)
                updated = True
        if updated:
            self._persist()


def embed_text(text: str) -> np.ndarray:
    """Return a normalised embedding vector for arbitrary text."""
    return SimilarityIndex._embed(text)


