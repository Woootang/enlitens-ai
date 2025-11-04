"""External research orchestrator that delegates to MCP-compatible connectors.

The orchestrator is intentionally lightweight so it can run inside the same
event loop as the LangGraph supervisor. Connectors may integrate with:

* Model Context Protocol tool hosts
* HTTP knowledge APIs (SerpAPI, Bing, civic open-data portals)
* Static fixtures for offline development

Each connector returns ``ResearchHit`` objects that include verification status
and metadata. The orchestrator deduplicates results before handing them back to
agents.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class ResearchQuery:
    """A structured research request issued by an agent."""

    topic: str
    focus: str
    location: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    max_results: int = 3
    require_verification: bool = True


@dataclass
class ResearchHit:
    """Structured representation of an external research snippet."""

    title: str
    snippet: str
    url: str
    publisher: Optional[str] = None
    published_at: Optional[str] = None
    summary: Optional[str] = None
    verification_status: str = "verified"
    score: float = 0.0


class ResearchConnector:
    """Abstract base for external research connectors."""

    name: str = "connector"

    async def search(self, query: ResearchQuery) -> List[ResearchHit]:
        raise NotImplementedError


class NullConnector(ResearchConnector):
    """Fallback connector that returns no results."""

    name = "null"

    async def search(self, query: ResearchQuery) -> List[ResearchHit]:
        logger.debug("Null connector invoked for query: %s", query.topic)
        return []


class StaticConnector(ResearchConnector):
    """Connector backed by static fixtures for deterministic tests."""

    name = "static"

    def __init__(self, fixtures: Sequence[Dict[str, Any]]):
        self._fixtures = [dict(item) for item in fixtures]

    async def search(self, query: ResearchQuery) -> List[ResearchHit]:
        hits: List[ResearchHit] = []
        for item in self._fixtures:
            hits.append(
                ResearchHit(
                    title=item.get("title", "Static research insight"),
                    snippet=item.get("snippet", item.get("summary", "")),
                    url=item.get("url", "https://example.com/static"),
                    publisher=item.get("publisher"),
                    published_at=item.get("published_at"),
                    summary=item.get("summary"),
                    verification_status=item.get("verification_status", "needs_review"),
                    score=float(item.get("score", 0.1)),
                )
            )
        return hits[: query.max_results]


class HTTPConnector(ResearchConnector):
    """Generic HTTP connector for RESTful knowledge APIs."""

    name = "http"

    def __init__(
        self,
        *,
        endpoint: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        payload_template: Optional[Dict[str, Any]] = None,
        timeout: float = 12.0,
    ) -> None:
        self.endpoint = endpoint
        self.method = method.upper()
        self.headers = headers or {}
        self.payload_template = payload_template or {}
        self.timeout = timeout

    async def search(self, query: ResearchQuery) -> List[ResearchHit]:
        try:
            import httpx  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            logger.warning("httpx not installed; HTTP connector disabled: %s", exc)
            return []

        request_payload = json.loads(json.dumps(self.payload_template))  # deep copy
        if isinstance(request_payload, dict):
            request_payload.setdefault("query", query.topic)
            request_payload.setdefault("location", query.location)
            request_payload.setdefault("tags", query.tags)
            request_payload.setdefault("limit", query.max_results)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                if self.method == "POST":
                    response = await client.post(
                        self.endpoint,
                        json=request_payload,
                        headers=self.headers,
                    )
                else:
                    params = {"query": query.topic, "limit": query.max_results}
                    response = await client.get(
                        self.endpoint,
                        params=params,
                        headers=self.headers,
                    )
                response.raise_for_status()
            except Exception as exc:  # pragma: no cover - IO guarded
                logger.warning("HTTP connector request failed: %s", exc)
                return []

        data = response.json()
        records: Iterable[Dict[str, Any]]
        if isinstance(data, dict):
            records = data.get("results") or data.get("data") or []
        else:
            records = data

        hits: List[ResearchHit] = []
        for record in records:
            title = record.get("title") or record.get("name")
            url = record.get("url") or record.get("link")
            snippet = record.get("snippet") or record.get("summary") or record.get("description")
            if not title or not url or not snippet:
                continue
            hits.append(
                ResearchHit(
                    title=str(title),
                    snippet=str(snippet),
                    url=str(url),
                    publisher=record.get("publisher") or record.get("source"),
                    published_at=record.get("published_at") or record.get("date"),
                    summary=str(record.get("summary") or snippet),
                    verification_status=str(record.get("verification_status") or "needs_review"),
                    score=float(record.get("score" or 0.5)),
                )
            )
        return hits[: query.max_results]


class MCPConnector(ResearchConnector):
    """Shell-based MCP connector that invokes a tool client command."""

    name = "mcp"

    def __init__(self, *, command: str, args: Optional[List[str]] = None, timeout: float = 20.0) -> None:
        self.command = command
        self.args = args or []
        self.timeout = timeout

    async def search(self, query: ResearchQuery) -> List[ResearchHit]:
        try:
            process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                json.dumps({
                    "query": query.topic,
                    "location": query.location,
                    "tags": query.tags,
                    "limit": query.max_results,
                }),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
        except Exception as exc:  # pragma: no cover - IO guarded
            logger.warning("MCP connector failed: %s", exc)
            return []

        if process.returncode != 0:
            logger.warning("MCP connector exited with %s: %s", process.returncode, stderr.decode("utf-8", "ignore"))
            return []

        try:
            payload = json.loads(stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("MCP connector returned invalid JSON: %s", exc)
            return []

        records = payload.get("results") if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            return []

        hits: List[ResearchHit] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            title = record.get("title")
            snippet = record.get("snippet") or record.get("summary")
            url = record.get("url")
            if not title or not snippet or not url:
                continue
            hits.append(
                ResearchHit(
                    title=str(title),
                    snippet=str(snippet),
                    url=str(url),
                    publisher=record.get("publisher") or record.get("source"),
                    published_at=record.get("published_at"),
                    summary=str(record.get("summary") or snippet),
                    verification_status=str(record.get("verification_status") or "needs_review"),
                    score=float(record.get("score" or 0.4)),
                )
            )
        return hits[: query.max_results]


class ExternalResearchOrchestrator:
    """Coordinates multiple research connectors and deduplicates hits."""

    def __init__(self, connectors: Sequence[ResearchConnector], *, max_total_results: int = 12) -> None:
        self.connectors = list(connectors) if connectors else [NullConnector()]
        self.max_total_results = max_total_results

    @classmethod
    def from_settings(cls) -> "ExternalResearchOrchestrator":
        raw_config = os.getenv("ENLITENS_RESEARCH_CONNECTORS", "")
        connectors: List[ResearchConnector] = []

        if raw_config:
            try:
                data = json.loads(raw_config)
            except json.JSONDecodeError as exc:
                logger.warning("Invalid ENLITENS_RESEARCH_CONNECTORS JSON: %s", exc)
                data = []

            if isinstance(data, list):
                for entry in data:
                    if not isinstance(entry, dict):
                        continue
                    connector_type = (entry.get("type") or "").lower()
                    try:
                        if connector_type == "http":
                            connectors.append(
                                HTTPConnector(
                                    endpoint=entry["endpoint"],
                                    method=entry.get("method", "GET"),
                                    headers=entry.get("headers"),
                                    payload_template=entry.get("payload"),
                                    timeout=float(entry.get("timeout", 12.0)),
                                )
                            )
                        elif connector_type == "mcp":
                            connectors.append(
                                MCPConnector(
                                    command=entry["command"],
                                    args=entry.get("args"),
                                    timeout=float(entry.get("timeout", 20.0)),
                                )
                            )
                        elif connector_type == "static":
                            connectors.append(StaticConnector(entry.get("fixtures", [])))
                    except KeyError as exc:
                        logger.warning("Connector config missing key: %s", exc)
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning("Failed to initialise connector: %s", exc)

        if not connectors:
            connectors = [NullConnector()]

        return cls(connectors=connectors)

    async def gather(self, queries: Sequence[ResearchQuery]) -> List[ResearchHit]:
        if not queries:
            return []

        tasks = []
        for connector in self.connectors:
            for query in queries:
                tasks.append(self._run_connector(connector, query))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        hits: List[ResearchHit] = []
        for result in results:
            if isinstance(result, Exception):
                logger.debug("Connector task raised: %s", result)
                continue
            hits.extend(result)

        deduped: Dict[str, ResearchHit] = {}
        for hit in hits:
            key = hit.url.strip().lower()
            existing = deduped.get(key)
            if existing is None or hit.score > existing.score:
                deduped[key] = hit

        ordered_hits = sorted(deduped.values(), key=lambda item: item.score, reverse=True)
        return ordered_hits[: self.max_total_results]

    async def _run_connector(self, connector: ResearchConnector, query: ResearchQuery) -> List[ResearchHit]:
        try:
            return await connector.search(query)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Connector %s failed: %s", connector.name, exc)
            return []

