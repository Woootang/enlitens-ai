"""Lightweight OpenAlex client for scholarly updates."""

from __future__ import annotations

from typing import List, Optional

import backoff
import httpx
from pydantic import BaseModel

OPENALEX_BASE = "https://api.openalex.org"


class OpenAlexWork(BaseModel):
    id: str
    title: str
    publication_year: Optional[int] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    concepts: List[str] = []
    primary_location: Optional[dict] = None


def _compose_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    if not inverted_index:
        return None
    tokens = sorted(
        ((word, positions) for word, positions in inverted_index.items()),
        key=lambda item: item[1][0],
    )
    ordered = sorted(
        [(pos, word) for word, positions in tokens for pos in positions],
        key=lambda item: item[0],
    )
    return " ".join(word for _, word in ordered)


@backoff.on_exception(backoff.expo, httpx.RequestError, max_time=60)
def search_openalex(query: str, *, per_page: int = 5) -> List[OpenAlexWork]:
    params = {
        "search": query,
        "per-page": per_page,
        "sort": "publication_year:desc",
    }
    response = httpx.get(f"{OPENALEX_BASE}/works", params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    works: List[OpenAlexWork] = []
    for item in payload.get("results", []):
        data = {
            "id": item.get("id", ""),
            "title": item.get("title", ""),
            "publication_year": item.get("publication_year"),
            "doi": (item.get("ids", {}) or {}).get("doi"),
            "abstract": _compose_abstract(item.get("abstract_inverted_index")),
            "primary_location": item.get("primary_location"),
        }
        concepts = [c.get("display_name", "") for c in item.get("concepts", [])][:5]
        data["concepts"] = [c for c in concepts if c]
        works.append(OpenAlexWork(**data))
    return works
