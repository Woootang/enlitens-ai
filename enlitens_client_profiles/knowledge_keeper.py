"""Knowledge Keeper agent that assembles an enriched graph from source materials."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

from .brand_intelligence import BrandMention, SiteDocument
from .config import ProfilePipelineConfig
from .data_ingestion import IngestionBundle, KnowledgeAsset, TranscriptSnippet

logger = logging.getLogger(__name__)


_TOKEN_SPLIT = re.compile(r"[^a-zA-Z0-9']+")
_STOPWORDS = {
    "the",
    "and",
    "with",
    "that",
    "from",
    "this",
    "have",
    "they",
    "their",
    "them",
    "about",
    "there",
    "into",
    "would",
    "could",
    "should",
    "because",
    "some",
    "what",
    "your",
    "when",
    "where",
    "which",
    "while",
    "through",
    "after",
    "before",
    "again",
    "really",
    "just",
    "like",
}


@dataclass(slots=True)
class KnowledgeGraphContext:
    graph: nx.MultiDiGraph
    locality_counts: Dict[str, int]
    top_keywords: List[Tuple[str, int]]
    founder_voice_highlights: List[str]
    knowledge_assets: List[KnowledgeAsset]
    analytics_summary: str = ""
    site_documents: List[SiteDocument] = field(default_factory=list)
    brand_mentions: List[BrandMention] = field(default_factory=list)

    def sample_founder_voice(self, limit: int = 5) -> List[str]:
        return self.founder_voice_highlights[:limit]


class KnowledgeKeeperAgent:
    """Construct and persist a knowledge graph for downstream personas."""

    def __init__(self, config: ProfilePipelineConfig) -> None:
        self.config = config
        self.graph_path = config.cache_dir / "knowledge_graph.graphml"

    def _tokenise(self, text: str) -> Iterable[str]:
        for token in _TOKEN_SPLIT.split(text.lower()):
            if not token or token in _STOPWORDS or len(token) < 3:
                continue
            yield token

    def _extract_keywords(self, sentences: Iterable[str], *, limit: int = 50) -> List[Tuple[str, int]]:
        counter: Counter[str] = Counter()
        for sentence in sentences:
            counter.update(self._tokenise(sentence))
        return counter.most_common(limit)

    def _add_intake_nodes(self, graph: nx.MultiDiGraph, bundle: IngestionBundle) -> None:
        for record in bundle.intakes:
            node_id = f"intake::{record.line_number}"
            graph.add_node(
                node_id,
                type="intake",
                line=record.line_number,
                text=record.raw_text,
            )
            for keyword in set(self._tokenise(record.raw_text)):
                keyword_id = f"keyword::{keyword}"
                graph.add_node(keyword_id, type="keyword", label=keyword)
                graph.add_edge(node_id, keyword_id, relation="mentions")

    def _add_transcript_nodes(self, graph: nx.MultiDiGraph, snippets: Iterable[TranscriptSnippet]) -> List[str]:
        highlights: List[str] = []
        for snippet in snippets:
            text = snippet.raw_text.strip()
            node_id = f"transcript::{snippet.line_number}"
            graph.add_node(
                node_id,
                type="transcript",
                speaker=snippet.speaker or "unknown",
                text=text,
            )
            if len(highlights) < 50 and len(text) > 40:
                highlights.append(text)
        return highlights

    def _add_locality_nodes(self, graph: nx.MultiDiGraph, locality_counts: Dict[str, int]) -> None:
        for locality, count in locality_counts.items():
            node_id = f"locality::{locality}"
            graph.add_node(node_id, type="locality", count=int(count))
            graph.add_edge("root", node_id, relation="appears_in_intakes")

    def _add_analytics_nodes(self, graph: nx.MultiDiGraph, analytics_summary: str) -> None:
        if not analytics_summary:
            return
        graph.add_node("analytics::summary", type="analytics", text=analytics_summary)
        graph.add_edge("root", "analytics::summary", relation="analytics_snapshot")

    def _add_site_documents(self, graph: nx.MultiDiGraph, documents: Iterable[SiteDocument]) -> None:
        for doc in documents:
            node_id = f"site::{doc.url}"
            graph.add_node(
                node_id,
                type="site_page",
                url=doc.url,
                title=doc.title,
                summary=doc.summary[:2000],
            )
            graph.add_edge("root", node_id, relation="brand_site")

    def _add_brand_mentions(self, graph: nx.MultiDiGraph, mentions: Iterable[BrandMention]) -> None:
        for mention in mentions:
            node_id = f"mention::{mention.url}"
            graph.add_node(
                node_id,
                type="brand_mention",
                url=mention.url,
                title=mention.title,
                snippet=mention.snippet[:1800],
                source=mention.source,
            )
            graph.add_edge("root", node_id, relation="brand_external_reference")

    def build_graph(self, bundle: IngestionBundle) -> KnowledgeGraphContext:
        graph = nx.MultiDiGraph()
        graph.add_node("root", type="root", label="knowledge")

        self._add_intake_nodes(graph, bundle)
        founder_highlights = self._add_transcript_nodes(graph, bundle.transcripts)
        self._add_locality_nodes(graph, bundle.locality_counts)
        self._add_analytics_nodes(graph, bundle.analytics_summary_block())
        self._add_site_documents(graph, bundle.site_documents)
        self._add_brand_mentions(graph, bundle.brand_mentions)

        for asset in bundle.knowledge_assets:
            node_id = f"asset::{asset.name}"
            graph.add_node(
                node_id,
                type="knowledge_asset",
                title=asset.name,
                metadata=json.dumps(asset.metadata, ensure_ascii=False),
            )
            graph.add_edge("root", node_id, relation="knowledge_asset")

        keywords = self._extract_keywords(record.raw_text for record in bundle.intakes)

        try:
            nx.write_graphml(graph, Path(self.graph_path))
        except Exception as exc:  # pragma: no cover - filesystem issues
            logger.warning("Unable to persist knowledge graph: %s", exc)

        return KnowledgeGraphContext(
            graph=graph,
            locality_counts=bundle.locality_counts,
            top_keywords=keywords,
            founder_voice_highlights=founder_highlights,
            knowledge_assets=bundle.knowledge_assets,
            analytics_summary=bundle.analytics_summary_block(),
            site_documents=bundle.site_documents,
            brand_mentions=bundle.brand_mentions,
        )


