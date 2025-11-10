"""Ingest raw artifacts (intakes, transcripts, PDFs) for profile generation."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.extraction.enhanced_pdf_extractor import EnhancedPDFExtractor

from .brand_intelligence import (
    BrandIntelligenceAgent,
    BrandIntelSnapshot,
    BrandMention,
    SiteDocument,
)
from .analytics import AnalyticsSnapshot, build_analytics_snapshot
from .config import ProfilePipelineConfig
from .stl_geography import STL_REGION_DATA


@dataclass(slots=True)
class IntakeRecord:
    raw_text: str
    line_number: int


@dataclass(slots=True)
class TranscriptSnippet:
    speaker: Optional[str]
    raw_text: str
    line_number: int


@dataclass(slots=True)
class KnowledgeAsset:
    name: str
    content: str
    metadata: Dict[str, str]


@dataclass(slots=True)
class IngestionBundle:
    intakes: List[IntakeRecord]
    transcripts: List[TranscriptSnippet]
    health_report_markdown: str
    knowledge_assets: List[KnowledgeAsset]
    locality_counts: Dict[str, int]
    intake_sentence_pool: List[str]
    founder_voice_snippets: List[str]
    analytics: Optional[AnalyticsSnapshot] = None
    site_documents: List[SiteDocument] = field(default_factory=list)
    brand_mentions: List[BrandMention] = field(default_factory=list)
    brand_site_block: str = ""
    brand_mentions_block: str = ""

    def analytics_summary_block(self) -> str:
        if not self.analytics:
            return ""
        return self.analytics.summary_block()


def load_intakes(path: Path) -> List[IntakeRecord]:
    """
    Load intake records from a file where each intake is wrapped in {}.
    Format: {intake text here}\n\n{next intake}\n\n...
    """
    import re
    text = path.read_text(encoding="utf-8")
    
    # Extract all text within {} brackets
    pattern = r'\{([^{}]+)\}'
    matches = re.findall(pattern, text, re.DOTALL)
    
    records: List[IntakeRecord] = []
    for idx, match in enumerate(matches, start=1):
        cleaned = match.strip()
        if cleaned and len(cleaned) > 10:  # Skip very short/empty ones
            records.append(IntakeRecord(raw_text=cleaned, line_number=idx))
    
    return records


def load_transcripts(path: Path) -> List[TranscriptSnippet]:
    snippets: List[TranscriptSnippet] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            clean = line.strip()
            if not clean:
                continue
            if ":" in clean:
                speaker, remainder = clean.split(":", 1)
                snippets.append(TranscriptSnippet(speaker=speaker.strip(), raw_text=remainder.strip(), line_number=idx))
            else:
                snippets.append(TranscriptSnippet(speaker=None, raw_text=clean, line_number=idx))
    return snippets


def load_health_report(pdf_path: Path) -> str:
    extractor = EnhancedPDFExtractor()
    result = extractor.extract(str(pdf_path))
    if isinstance(result, dict):
        archival = result.get("archival_content", {})
        markdown = archival.get("full_document_text_markdown")
        if markdown:
            return markdown
    if isinstance(result, str):
        return result
    raise RuntimeError(f"Unable to extract text from {pdf_path}")


def load_knowledge_assets(directory: Path) -> List[KnowledgeAsset]:
    assets: List[KnowledgeAsset] = []
    for path in directory.iterdir():
        if path.is_dir():
            continue
        if path.suffix.lower() in {".txt", ".md", ".json"}:
            content = path.read_text(encoding="utf-8", errors="ignore")
            metadata: Dict[str, str] = {"filename": path.name}
            if path.suffix.lower() == ".json":
                try:
                    parsed = json.loads(content)
                    metadata["json_keys"] = ",".join(list(parsed.keys())[:10])
                except Exception:
                    metadata["json_keys"] = "parse_error"
            assets.append(KnowledgeAsset(name=path.stem, content=content, metadata=metadata))
    return assets


def _split_sentences(blocks: Iterable[str]) -> List[str]:
    sentences: List[str] = []
    for block in blocks:
        if not block:
            continue
        # Primitive sentence splitting to avoid heavy dependencies
        for sentence in re.split(r"(?<=[.!?])\s+", block):
            cleaned = sentence.strip()
            if len(cleaned) >= 40:
                sentences.append(cleaned)
    return sentences


def _collect_localities(records: Iterable[IntakeRecord]) -> Dict[str, int]:
    locality_counter: Counter[str] = Counter()
    all_municipalities = {name.lower(): name for name in STL_REGION_DATA.all_municipalities}

    for record in records:
        lowered = record.raw_text.lower()
        for city_lower, city_proper in all_municipalities.items():
            if city_lower in lowered:
                locality_counter[city_proper] += 1

    return dict(locality_counter)


def _collect_founder_snippets(snippets: Iterable[TranscriptSnippet], limit: int = 150) -> List[str]:
    collected: List[str] = []
    for snippet in snippets:
        if snippet.speaker:
            normalized = snippet.speaker.strip().lower()
            if normalized not in {"liz", "liz wooten", "liz w."}:
                continue
        text = snippet.raw_text.strip()
        if len(text) < 20:
            continue
        collected.append(text)
        if len(collected) >= limit:
            break
    return collected


def load_ingestion_bundle(config: ProfilePipelineConfig) -> IngestionBundle:
    intakes = load_intakes(config.intakes_path)
    transcripts = load_transcripts(config.transcripts_path)
    health_report = load_health_report(config.health_report_path)
    knowledge_assets = load_knowledge_assets(config.knowledge_base_dir)
    analytics_snapshot = build_analytics_snapshot(
        credentials_path=config.google_credentials_path,
        ga_property_id=config.ga_property_id,
        gsc_site_url=config.gsc_site_url,
        lookback_days=config.analytics_lookback_days,
    )

    brand_agent = BrandIntelligenceAgent(config)
    brand_snapshot: BrandIntelSnapshot = brand_agent.collect()

    return IngestionBundle(
        intakes=intakes,
        transcripts=transcripts,
        health_report_markdown=health_report,
        knowledge_assets=knowledge_assets,
        locality_counts=_collect_localities(intakes),
        intake_sentence_pool=_split_sentences(record.raw_text for record in intakes),
        founder_voice_snippets=_collect_founder_snippets(transcripts),
        analytics=analytics_snapshot,
        site_documents=brand_snapshot.site_documents,
        brand_mentions=brand_snapshot.brand_mentions,
        brand_site_block=brand_snapshot.site_markdown(),
        brand_mentions_block=brand_snapshot.mentions_markdown(),
    )

