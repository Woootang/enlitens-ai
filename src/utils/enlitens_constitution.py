"""Utilities for loading and applying the Enlitens constitutional principles."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
CONSTITUTION_PATH = BASE_DIR / "docs" / "enlitens_constitution_v2.yaml"


@dataclass(frozen=True)
class Principle:
    """Representation of a single constitutional principle."""

    principle_id: str
    title: str
    description: str
    directives: Sequence[str]
    exemplar_language: Sequence[str]

    def prompt_block(self, include_exemplars: bool = True) -> str:
        body_lines = [f"{self.principle_id} – {self.title}: {self.description}"]
        if self.directives:
            body_lines.append("Directives:")
            body_lines.extend([f"  • {item}" for item in self.directives])
        if include_exemplars and self.exemplar_language:
            body_lines.append("Language Examples:")
            body_lines.extend([f"  • {item}" for item in self.exemplar_language])
        return "\n".join(body_lines)


@lru_cache(maxsize=1)
def _load_constitution() -> Dict[str, Any]:
    if not CONSTITUTION_PATH.exists():
        raise FileNotFoundError(f"Enlitens constitution not found at {CONSTITUTION_PATH}")
    with CONSTITUTION_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class EnlitensConstitution:
    """Helper for retrieving principles and enforcing language policies."""

    PATHOLOGY_TERMS = {
        "disorder": "pattern",
        "Disorder": "Pattern",
        "disorders": "patterns",
        "Disorders": "Patterns",
        "deficit": "difference",
        "Deficit": "Difference",
        "deficits": "differences",
        "Deficits": "Differences",
        "pathology": "legacy pathology narrative",
        "Pathology": "Legacy pathology narrative",
    }
    LEGACY_REFERENCES = {
        "DSM": "DSM (legacy checklist manual)",
        "DSM-5": "DSM-5 (legacy checklist manual)",
        "ADOS": "ADOS (legacy observational battery under critique)",
        "ADOS-2": "ADOS-2 (legacy observational battery under critique)",
    }
    CONTEXT_KEYWORDS = {"context", "environment", "system", "water", "conditions", "ecosystem"}
    STRENGTH_KEYWORDS = {"strength", "asset", "capacity", "talent", "creativity", "resilience", "superpower"}
    SYSTEM_KEYWORDS = {"system", "policy", "structure", "ableism", "racism", "capitalism", "workplace", "school"}
    TRAUMA_KEYWORDS = {"safety", "nervous", "polyvagal", "regulation", "survival"}
    AUTONOMY_KEYWORDS = {"independent", "autonomy", "blueprint", "launch", "self-advocacy", "self advocacy", "graduate"}
    BOLD_MARKERS = {"bullshit", "torch", "rebellion", "burn", "bold", "refuse"}

    def __init__(self) -> None:
        data = _load_constitution()
        self.version: str = str(data.get("version", ""))
        self.updated: str = str(data.get("updated", ""))
        self._principles: Dict[str, Principle] = {}
        for raw in data.get("principles", []):
            principle = Principle(
                principle_id=raw.get("id"),
                title=raw.get("title", ""),
                description=raw.get("description", ""),
                directives=tuple(raw.get("directives", []) or []),
                exemplar_language=tuple(raw.get("exemplar_language", []) or []),
            )
            self._principles[principle.principle_id] = principle

    # ------------------------------------------------------------------
    # Principle helpers
    # ------------------------------------------------------------------
    def get(self, principle_id: str) -> Principle:
        try:
            return self._principles[principle_id]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise KeyError(f"Unknown principle id: {principle_id}") from exc

    def iter(self, principle_ids: Optional[Iterable[str]] = None) -> Iterable[Principle]:
        if principle_ids is None:
            return self._principles.values()
        return (self._principles[pid] for pid in principle_ids if pid in self._principles)

    def render_prompt_section(
        self,
        principle_ids: Optional[Iterable[str]] = None,
        *,
        include_exemplars: bool = True,
        header: str = "ENLITENS CONSTITUTION",
    ) -> str:
        blocks = [header]
        for principle in self.iter(principle_ids):
            blocks.append(principle.prompt_block(include_exemplars=include_exemplars))
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Language enforcement helpers
    # ------------------------------------------------------------------
    def sanitize_language(self, text: str) -> str:
        if not text:
            return text
        sanitized = text
        for term, replacement in self.PATHOLOGY_TERMS.items():
            sanitized = re.sub(rf"\b{re.escape(term)}\b", replacement, sanitized)
        for term, replacement in self.LEGACY_REFERENCES.items():
            sanitized = sanitized.replace(term, replacement)
        return sanitized

    def sanitize_list(self, values: Optional[Sequence[str]]) -> List[str]:
        cleaned: List[str] = []
        if not values:
            return cleaned
        for item in values:
            cleaned_item = self.sanitize_language(item)
            if cleaned_item:
                cleaned.append(cleaned_item)
        return cleaned

    def sanitize_mapping(self, mapping: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in (mapping or {}).items():
            if isinstance(value, list):
                sanitized[key] = self.sanitize_list(value)
            elif isinstance(value, str):
                sanitized[key] = self.sanitize_language(value)
            else:
                sanitized[key] = value
        return sanitized

    def contains_pathology(self, text: str) -> bool:
        lowered = text.lower()
        return any(re.search(rf"\b{re.escape(term.lower())}\b", lowered) for term in self.PATHOLOGY_TERMS)

    def contains_legacy_reference(self, text: str) -> bool:
        return any(token in text for token in self.LEGACY_REFERENCES)

    def ensure_keyword_presence(self, text: str, keywords: Iterable[str]) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in keywords)

    def as_json(self, principle_ids: Optional[Iterable[str]] = None) -> str:
        export = [
            {
                "id": principle.principle_id,
                "title": principle.title,
                "description": principle.description,
                "directives": list(principle.directives),
                "exemplar_language": list(principle.exemplar_language),
            }
            for principle in self.iter(principle_ids)
        ]
        return json.dumps(export, ensure_ascii=False, indent=2)


__all__ = ["EnlitensConstitution", "Principle", "CONSTITUTION_PATH"]
