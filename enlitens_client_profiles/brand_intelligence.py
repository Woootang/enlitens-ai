"""Brand intelligence agent for crawling Enlitens assets and collecting mentions."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree as ET

import httpx

try:  # pragma: no cover - optional dependency at runtime
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - neo4j driver optional
    GraphDatabase = None  # type: ignore

from .config import ProfilePipelineConfig
from .site_crawler import SiteCrawler

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SiteDocument:
    url: str
    title: str
    headings: List[str]
    summary: str


@dataclass(slots=True)
class BrandMention:
    source: str
    title: str
    url: str
    snippet: str


@dataclass(slots=True)
class BrandIntelSnapshot:
    generated_at: str
    site_documents: List[SiteDocument] = field(default_factory=list)
    brand_mentions: List[BrandMention] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)

    def site_markdown(self, limit: int = 8) -> str:
        lines: List[str] = []
        for doc in self.site_documents[:limit]:
            lines.append(f"### {doc.title or doc.url}\n{doc.summary[:1200]}")
        return "\n\n".join(lines)

    def mentions_markdown(self, limit: int = 6) -> str:
        lines: List[str] = []
        for mention in self.brand_mentions[:limit]:
            lines.append(f"* {mention.title} → {mention.url}\n  {mention.snippet}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "site_documents": [
                {
                    "url": doc.url,
                    "title": doc.title,
                    "headings": doc.headings,
                    "summary": doc.summary,
                }
                for doc in self.site_documents
            ],
            "brand_mentions": [
                {
                    "source": mention.source,
                    "title": mention.title,
                    "url": mention.url,
                    "snippet": mention.snippet,
                }
                for mention in self.brand_mentions
            ],
            "search_queries": list(self.search_queries),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "BrandIntelSnapshot":
        site_documents = [
            SiteDocument(
                url=str(entry.get("url", "")),
                title=str(entry.get("title", "")),
                headings=list(entry.get("headings", []) or []),
                summary=str(entry.get("summary", "")),
            )
            for entry in payload.get("site_documents", [])
        ]
        brand_mentions = [
            BrandMention(
                source=str(entry.get("source", "")),
                title=str(entry.get("title", "")),
                url=str(entry.get("url", "")),
                snippet=str(entry.get("snippet", "")),
            )
            for entry in payload.get("brand_mentions", [])
        ]
        return cls(
            generated_at=str(payload.get("generated_at", datetime.utcnow().isoformat())),
            site_documents=site_documents,
            brand_mentions=brand_mentions,
            search_queries=list(payload.get("search_queries", []) or []),
        )


class BrandIntelligenceAgent:
    """Collect Enlitens site content and public mentions into a structured snapshot."""

    BRAND_KEYWORDS = [
        "\"Enlitens\"",
        "\"Enlitens Counseling\"",
        "\"Liz Wooten\" Enlitens",
        "\"Liz Wooten LPC\"",
    ]

    REVIEW_BLOCKLIST = {
        "google.com/maps",
        "maps.google",
        "facebook.com",
        "yelp.com",
        "healthgrades.com",
        "zocdoc.com",
    }

    def __init__(self, config: ProfilePipelineConfig) -> None:
        self.config = config
        self.base_url = config.enlitens_site_root.rstrip("/") + "/"
        self.brave_key = config.brave_api_key or os.environ.get("BRAVE_API_KEY")

    def collect(self, *, force_refresh: bool = False) -> BrandIntelSnapshot:
        if not force_refresh and self.config.brand_snapshot_path.exists():
            try:
                cached = json.loads(self.config.brand_snapshot_path.read_text(encoding="utf-8"))
                snapshot = BrandIntelSnapshot.from_dict(cached)
                if snapshot.site_documents and snapshot.brand_mentions:
                    return snapshot
            except Exception:  # pragma: no cover - cache fallback
                logger.debug("Failed to load cached brand snapshot, rebuilding")

        urls = self._fetch_sitemap_urls()
        logger.info("Brand intelligence: discovered %d sitemap URLs", len(urls))
        site_documents = self._crawl_site(urls)
        mentions, queries = self._search_brand_mentions()

        snapshot = BrandIntelSnapshot(
            generated_at=datetime.utcnow().isoformat(),
            site_documents=site_documents,
            brand_mentions=mentions,
            search_queries=queries,
        )

        self._persist_snapshot(snapshot)
        self._persist_to_neo4j(snapshot)
        return snapshot

    # --- Sitemap helpers -----------------------------------------------------------------

    def _fetch_sitemap_urls(self) -> List[str]:
        candidates = ["sitemap_index.xml", "sitemap.xml"]
        collected: List[str] = []
        for candidate in candidates:
            target = urljoin(self.base_url, candidate)
            try:
                response = httpx.get(target, timeout=20.0)
                response.raise_for_status()
            except Exception:
                continue

            try:
                root = ET.fromstring(response.text)
            except ET.ParseError:
                continue

            if root.tag.endswith("sitemapindex"):
                for child in root.findall("{*}sitemap/{*}loc"):
                    sub_url = child.text.strip()
                    collected.extend(self._fetch_sitemap_leaf(sub_url))
                break
            elif root.tag.endswith("urlset"):
                for child in root.findall("{*}url/{*}loc"):
                    loc = child.text.strip()
                    if loc:
                        collected.append(loc)
                break

        return sorted(set(collected))

    def _fetch_sitemap_leaf(self, url: str) -> List[str]:
        urls: List[str] = []
        try:
            response = httpx.get(url, timeout=20.0)
            response.raise_for_status()
            root = ET.fromstring(response.text)
        except Exception:
            return urls

        if not root.tag.endswith("urlset"):
            return urls

        for child in root.findall("{*}url/{*}loc"):
            loc = child.text.strip()
            if loc:
                urls.append(loc)
        return urls

    # --- Crawling ------------------------------------------------------------------------

    def _crawl_site(self, urls: Sequence[str]) -> List[SiteDocument]:
        if not urls:
            return []

        crawler = SiteCrawler(self.base_url, cache_dir=self.config.site_cache_dir)
        pages = crawler.crawl(urls=urls)
        documents: List[SiteDocument] = []
        for url, payload in pages.items():
            documents.append(
                SiteDocument(
                    url=url,
                    title=payload.get("title", ""),
                    headings=payload.get("headings", []),
                    summary=payload.get("summary", ""),
                )
            )
        documents.sort(key=lambda entry: entry.url)
        return documents

    # --- External mentions ---------------------------------------------------------------

    def _search_brand_mentions(self) -> tuple[List[BrandMention], List[str]]:
        if not self.brave_key:
            logger.warning("Brave API key not configured; skipping external brand search")
            return [], []

        mentions: List[BrandMention] = []
        seen_domains: set[str] = set()
        queries: List[str] = []

        for keyword in self.BRAND_KEYWORDS:
            query = f"{keyword} -review -glassdoor"
            try:
                response = httpx.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={"X-Subscription-Token": self.brave_key},
                    params={"q": query, "count": 8, "country": "us", "safesearch": "off"},
                    timeout=15.0,
                )
                response.raise_for_status()
            except Exception as exc:  # pragma: no cover - remote failure
                logger.warning("Brand search failed (%s): %s", query, exc)
                continue

            queries.append(query)
            payload = response.json()
            for result in payload.get("web", {}).get("results", []) or []:
                url = result.get("url", "")
                domain = urlparse(url).netloc.lower()
                if not url or any(block in domain for block in self.REVIEW_BLOCKLIST):
                    continue
                if domain in seen_domains:
                    continue
                seen_domains.add(domain)
                mentions.append(
                    BrandMention(
                        source="brave",
                        title=result.get("title", ""),
                        url=url,
                        snippet=result.get("description", ""),
                    )
                )
            time.sleep(1.05)

        return mentions, queries

    # --- Persistence --------------------------------------------------------------------

    def _persist_snapshot(self, snapshot: BrandIntelSnapshot) -> None:
        try:
            self.config.brand_snapshot_path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - filesystem errors
            logger.warning("Failed to persist brand snapshot: %s", exc)

    def _persist_to_neo4j(self, snapshot: BrandIntelSnapshot) -> None:
        if not (self.config.neo4j_uri and self.config.neo4j_user and self.config.neo4j_password):
            return
        if GraphDatabase is None:  # pragma: no cover - dependency not installed
            logger.warning("neo4j driver unavailable; skipping brand graph persistence")
            return

        try:
            driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
            )
        except Exception as exc:  # pragma: no cover - connection failure
            logger.warning("Unable to connect to Neo4j: %s", exc)
            return

        database = self.config.neo4j_database or None

        def _run(tx, query: str, **params) -> None:
            tx.run(query, **params)

        with driver.session(database=database) as session:
            session.execute_write(
                _run,
                """
                MERGE (b:Brand {name: $brand})
                SET b.updated_at = datetime($updated_at)
                """,
                brand="Enlitens",
                updated_at=snapshot.generated_at,
            )
            for doc in snapshot.site_documents:
                session.execute_write(
                    _run,
                    """
                    MERGE (b:Brand {name: $brand})
                    MERGE (p:Page {url: $url})
                    SET p.title = $title, p.summary = $summary
                    MERGE (b)-[:HAS_PAGE]->(p)
                    """,
                    brand="Enlitens",
                    url=doc.url,
                    title=doc.title,
                    summary=doc.summary[:2000],
                )
            for mention in snapshot.brand_mentions:
                session.execute_write(
                    _run,
                    """
                    MERGE (b:Brand {name: $brand})
                    MERGE (m:Mention {url: $url})
                    SET m.title = $title, m.snippet = $snippet, m.source = $source
                    MERGE (b)-[:MENTIONED_IN]->(m)
                    """,
                    brand="Enlitens",
                    url=mention.url,
                    title=mention.title,
                    snippet=mention.snippet[:1800],
                    source=mention.source,
                )

        driver.close()


