"""Deep research agent that enriches persona scaffolds with external data."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.parse import urlparse

import requests

from .config import ProfilePipelineConfig
from .foundation_builder import PersonaFoundation

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SearchResult:
    query: str
    title: str
    url: str
    snippet: str
    source: str


@dataclass(slots=True)
class ResearchCache:
    generated_at: datetime
    queries: List[str] = field(default_factory=list)
    results: List[SearchResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "queries": self.queries,
            "results": [
                {
                    "query": result.query,
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "source": result.source,
                }
                for result in self.results
            ],
            "notes": self.notes,
            "missing": self.missing,
        }

    def narrative_block(self) -> str:
        lines: List[str] = []
        for note in self.notes:
            lines.append(f"- {note}")
        for result in self.results[:15]:
            lines.append(f"* {result.query}: {result.title} → {result.url}")
        return "\n".join(lines)


class DeepResearchAgent:
    """Query open web/search APIs to enrich persona foundations."""

    def __init__(self, config: ProfilePipelineConfig) -> None:
        self.config = config
        self.serper_key = os.environ.get("SERPER_API_KEY")
        self.firecrawl_key = os.environ.get("FIRECRAWL_API_KEY")
        self.brave_key = os.environ.get("BRAVE_API_KEY")
        self.output_dir = config.cache_dir / "research"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, foundation: PersonaFoundation) -> ResearchCache:
        cache = ResearchCache(generated_at=datetime.utcnow())

        queries = self._compose_queries(foundation)
        cache.queries = queries

        for query in queries:
            results = self._run_single_query(query)
            if not results:
                cache.missing.append(query)
                continue
            cache.results.extend(results)

        if foundation.gaps:
            cache.notes.extend(foundation.gaps)

        if cache.results:
            self._persist(cache)
        else:
            logger.info("Deep research produced no results; nothing persisted.")
        return cache

    # Compose queries
    def _compose_queries(self, foundation: PersonaFoundation) -> List[str]:
        queries: List[str] = []
        seen: set[str] = set()

        def _normalise(text: str, *, fallback: str = "") -> str:
            cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return cleaned or fallback

        def _add(query: str) -> None:
            clean = query.strip()
            if not clean or clean in seen:
                return
            seen.add(clean)
            queries.append(clean)

        localities = [item.split("(")[0].strip() for item in foundation.locality_hypotheses[:4] if item]
        if not localities:
            localities = ["St. Louis", "Clayton", "Kirkwood"]

        for locality in localities:
            _add(f"{locality} neurodivergent community meetups 2025")
            _add(f"{locality} public library sensory friendly events")
            _add(f"{locality} school district autism accommodations programs")
            _add(f"{locality} coworking spaces inclusive neurodivergent adults")
            _add(f"{locality} affordable recreation for neurodivergent families")

        for clue in foundation.family_clues[:3]:
            snippet = _normalise(clue[:120], fallback="caregiver")
            _add(f"{localities[0]} resources for {snippet} caregiver support")

        for clue in foundation.occupation_clues[:3]:
            occupation = _normalise(clue, fallback="professionals")
            _add(f"st louis {occupation} workplace inclusion initiatives")

        for signal in foundation.search_signals[:6]:
            focus = _normalise(signal, fallback="neurodivergent support")
            _add(f"{focus} st louis community data 2025")

        if not queries:
            _add("st louis neurodivergent community resources 2025")

        return queries[:12]

    def _run_single_query(self, query: str) -> List[SearchResult]:
        results: List[SearchResult] = []
        if self.brave_key:
            results = self._query_brave(query)
        if not results and self.serper_key:
            results = self._query_serper(query)

        if results:
            return self._filter_results(results)
        logger.debug("No search API keys available or no results for query '%s'", query)
        return []

    def _query_serper(self, query: str) -> List[SearchResult]:
        try:
            response = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": self.serper_key or "", "Content-Type": "application/json"},
                json={"q": query, "gl": "us", "hl": "en", "num": 8},
                timeout=15,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Serper query failed (%s): %s", query, exc)
            return []

        payload = response.json()
        organic = payload.get("organic", []) or []
        results: List[SearchResult] = []
        for entry in organic[:8]:
            results.append(
                SearchResult(
                    query=query,
                    title=entry.get("title", ""),
                    url=entry.get("link", ""),
                    snippet=entry.get("snippet", ""),
                    source="serper",
                )
            )
        return results

    def _query_brave(self, query: str) -> List[SearchResult]:
        response: Optional[requests.Response] = None
        for attempt in range(3):
            try:
                response = requests.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={"X-Subscription-Token": self.brave_key or ""},
                    params={"q": query, "count": 8, "country": "us", "safesearch": "off"},
                    timeout=15,
                )
                response.raise_for_status()
                break
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status == 429 and attempt < 2:
                    wait_time = 1.2 + attempt
                    logger.warning("Brave rate limit hit (%s); retrying after %.1fs", query, wait_time)
                    time.sleep(wait_time)
                    continue
                logger.warning("Brave query failed (%s): %s", query, exc)
                return []
            except Exception as exc:
                logger.warning("Brave query failed (%s): %s", query, exc)
                return []

        payload = response.json()
        web = payload.get("web", {}).get("results", []) or []
        results: List[SearchResult] = []
        for entry in web[:8]:
            results.append(
                SearchResult(
                    query=query,
                    title=entry.get("title", ""),
                    url=entry.get("url", ""),
                    snippet=entry.get("description", ""),
                    source="brave",
                )
            )
        time.sleep(1.05)
        return results

    def _filter_results(self, results: Sequence[SearchResult]) -> List[SearchResult]:
        filtered: List[SearchResult] = []
        seen_domains: set[str] = set()
        mental_health_markers = ("therapy", "counsel", "mentalhealth", "psychi", "psych", "behavioralhealth")
        allowed_health_domains = {"enlitens.com"}

        for result in results:
            domain = urlparse(result.url).netloc.lower()
            if domain in seen_domains:
                continue
            if any(marker in domain for marker in mental_health_markers) and domain not in allowed_health_domains:
                continue
            seen_domains.add(domain)
            filtered.append(result)
        return filtered

    def _persist(self, cache: ResearchCache) -> None:
        timestamp = cache.generated_at.strftime("%Y%m%d_%H%M%S")
        target = self.output_dir / f"research_{timestamp}.json"
        try:
            target.write_text(json.dumps(cache.to_json(), indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to persist research cache: %s", exc)


