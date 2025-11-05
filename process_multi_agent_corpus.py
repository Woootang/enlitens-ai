#!/usr/bin/env python3
"""
Multi-Agent Enlitens Corpus Processing System

This script orchestrates a sophisticated multi-agent system for processing research papers
and generating high-quality, neuroscience-based content for Enlitens therapy practice.

Features:
- Multi-agent architecture with specialized agents
- GPU memory optimization for 24GB VRAM systems
- Comprehensive error handling and recovery
- St. Louis regional context integration
- Founder voice (Liz Wooten) authenticity
- Quality validation and scoring
- Progress tracking and checkpointing
"""

import argparse
import asyncio
import copy
import gc
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.supervisor_agent import SupervisorAgent
from src.models.enlitens_schemas import EnlitensKnowledgeBase, EnlitensKnowledgeEntry
from src.extraction.enhanced_pdf_extractor import EnhancedPDFExtractor
from src.extraction.enhanced_extraction_tools import EnhancedExtractionTools
from src.agents.extraction_team import ExtractionTeam
from src.utils.enhanced_logging import setup_enhanced_logging, log_startup_banner
from src.retrieval.embedding_ingestion import EmbeddingIngestionPipeline
from src.utils.terminology import sanitize_structure, contains_banned_terms
from src.context import REGIONAL_CONTEXT_DATA, match_municipalities
from src.pipeline.optional_context_loader import analyze_optional_context
from src.data.locality_loader import LocalityRecord, load_locality_reference

# Keyword buckets used to derive themes from intake narratives and transcripts
INTAKE_THEME_KEYWORDS: Dict[str, str] = {
    "adhd": "ADHD & executive function",
    "attention": "ADHD & executive function",
    "executive": "ADHD & executive function",
    "focus": "ADHD & executive function",
    "autism": "Autistic identity",
    "asd": "Autistic identity",
    "mask": "Autistic identity",
    "sensory": "Sensory integration",
    "meltdown": "Sensory integration",
    "overwhelm": "Sensory integration",
    "anxiety": "Anxiety & regulation",
    "panic": "Anxiety & regulation",
    "worry": "Anxiety & regulation",
    "depression": "Mood regulation",
    "mood": "Mood regulation",
    "trauma": "Trauma & complex PTSD",
    "ptsd": "Trauma & complex PTSD",
    "grief": "Grief & adjustment",
    "loss": "Grief & adjustment",
    "burnout": "Burnout & work stress",
    "work": "Burnout & work stress",
    "job": "Burnout & work stress",
    "relationship": "Relational stress",
    "marriage": "Relational stress",
    "partner": "Relational stress",
    "family": "Relational stress",
    "child": "Family supports",
    "school": "School supports",
    "college": "School supports",
    "assessment": "Assessment navigation",
    "diagnosis": "Assessment navigation",
    "women": "Gendered diagnosis gaps",
    "female": "Gendered diagnosis gaps",
    "girl": "Gendered diagnosis gaps",
    "nonbinary": "Gender expansive experiences",
    "trans": "Gender expansive experiences",
    "lgbt": "Gender expansive experiences",
}

HEALTH_PRIORITY_TERMS: Dict[str, str] = {
    "maternal": "Maternal & perinatal health",
    "birth": "Maternal & perinatal health",
    "infant": "Maternal & perinatal health",
    "asthma": "Asthma & air quality",
    "air": "Asthma & air quality",
    "opioid": "Substance use",
    "overdose": "Substance use",
    "suicide": "Behavioral health",
    "violence": "Community violence",
    "homicide": "Community violence",
    "transport": "Transportation barriers",
    "transit": "Transportation barriers",
    "food": "Food security",
    "housing": "Housing security",
    "homeless": "Housing security",
    "lead": "Environmental health",
    "pollution": "Environmental health",
}
# Configure comprehensive logging - single log file for all processing
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = "enlitens_complete_processing.log"  # Single comprehensive log
log_file_path = Path("logs") / log_filename

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Determine remote monitoring endpoint (defaults to local monitoring server)
monitor_endpoint_env = os.getenv("ENLITENS_MONITOR_URL")
if monitor_endpoint_env is None:
    monitor_endpoint = os.getenv("ENLITENS_MONITOR_URL", "http://localhost:8765/api/log")
else:
    monitor_endpoint = monitor_endpoint_env.strip()

if monitor_endpoint and monitor_endpoint.lower() in {"", "none", "disable", "disabled", "false", "0"}:
    monitor_endpoint = None

# Setup enhanced logging with visual improvements
setup_enhanced_logging(
    log_file=str(log_file_path),
    file_level=logging.INFO,
    console_level=logging.INFO,
    remote_logging_url=monitor_endpoint
)

logger = logging.getLogger(__name__)

if monitor_endpoint:
    logger.info(f"📡 Streaming logs to monitoring server at {monitor_endpoint}")
else:
    logger.info("📝 Remote monitoring disabled; using local log file only")

# ---------------------------------------------------------------------------
# Monitoring helpers
# ---------------------------------------------------------------------------


def post_monitor_stats(payload: Dict[str, Any]) -> None:
    """Send structured progress updates to the monitoring dashboard."""

    if not monitor_endpoint:
        return

    try:
        import httpx
        from urllib.parse import urlparse

        # Parse the URL to get the scheme, netloc, and path
        parsed_url = urlparse(monitor_endpoint)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        path = parsed_url.path

        # Construct the full URL for the request
        full_url = f"{base_url}{path}"

        # Ensure the path ends with a slash if it's a directory
        if full_url.endswith('/'):
            full_url = full_url[:-1]

        # Add query parameters if any
        if parsed_url.query:
            full_url += f"?{parsed_url.query}"

        # Add fragment if any
        if parsed_url.fragment:
            full_url += f"#{parsed_url.fragment}"

        # Use httpx to send the request
        with httpx.Client() as client:
            response = client.post(
                full_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10.0, # Increased timeout for monitoring
            )
            response.raise_for_status() # Raise an exception for bad status codes

    except (httpx.RequestError, httpx.HTTPStatusError, TimeoutError, ConnectionError) as e:
        logger.warning(f"Could not send stats to monitoring server: {e}")
    except Exception as e:
        logger.warning(f"An unexpected error occurred during monitoring stats: {e}")

# Clean up old logs after logger is configured
try:
    import glob
    old_logs = glob.glob("*.log") + glob.glob("logs/*.log")
    for old_log in old_logs:
        if old_log not in [log_filename, f"logs/{log_filename}"]:  # Don't delete current log
            try:
                os.remove(old_log)
            except OSError:
                pass  # File might not exist or be in use
    logger.info(f"🧹 Cleaned up old log files")
except Exception as e:
    logger.warning(f"Could not clean old logs: {e}")

class MultiAgentProcessor:
    """
    Comprehensive multi-agent processor for Enlitens knowledge base generation.
    """

    def __init__(self, input_dir: str, output_file: str, st_louis_report: Optional[str] = None):
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.st_louis_report = Path(st_louis_report) if st_louis_report else None
        self.context_dir = Path(__file__).parent
        self.temp_file = Path(f"{output_file}.temp")

        # Initialize components
        self.supervisor = SupervisorAgent()
        self.pdf_extractor = EnhancedPDFExtractor()
        self.extraction_tools = EnhancedExtractionTools()
        self.extraction_team = ExtractionTeam()
        self.knowledge_base = EnlitensKnowledgeBase()

        # Vector store ingestion pipeline (optional)
        disable_vector_ingestion = os.getenv("ENLITENS_DISABLE_VECTOR_INGESTION", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.embedding_ingestion: Optional[EmbeddingIngestionPipeline] = None
        if disable_vector_ingestion:
            logger.info("🔌 Vector store ingestion disabled via environment toggle")
        else:
            try:
                self.embedding_ingestion = EmbeddingIngestionPipeline()
                logger.info("🧠 Vector store ingestion pipeline initialized")
                self._bootstrap_context_ingestion()
            except Exception as exc:
                logger.warning("⚠️ Failed to initialize vector ingestion pipeline: %s", exc)
                self.embedding_ingestion = None

        # St. Louis regional context
        self.st_louis_context = self._load_st_louis_context()
        try:
            self.locality_reference: Dict[str, LocalityRecord] = load_locality_reference()
        except FileNotFoundError as exc:
            logger.warning("Locality reference unavailable: %s", exc)
            self.locality_reference = {}

        # Optional context assets
        (
            self.client_intake_text,
            self.client_intake_path,
        ) = self._load_optional_context_text(
            "intakes.txt", description="client intake narratives"
        )
        (
            self.transcript_text,
            self.transcript_path,
        ) = self._load_optional_context_text(
            "transcripts.txt", description="founder transcripts"
        )
        self.health_report_summary = self._extract_health_report_summary()
        self.intake_registry = self._build_intake_registry(self.client_intake_text)
        self.transcript_registry = self._build_transcript_registry(self.transcript_text)
        self.locality_research_backlog = self._identify_locality_backlog()
        self.theme_landscape = self._compile_theme_landscape()
        self.regional_atlas = self._build_regional_atlas()

        # Processing configuration
        self.max_concurrent_documents = 1  # Sequential processing for memory management
        self.checkpoint_interval = 1  # Save after each document
        self.retry_attempts = 3

        logger.info("🚀 Multi-Agent Processor initialized")

    def _bootstrap_context_ingestion(self) -> None:
        """Seed the vector store with static context documents when available."""

        if not self.embedding_ingestion:
            return

        context_documents: List[Tuple[str, str, Dict[str, Any]]] = []

        intakes_text, intakes_path = self._load_optional_context_text(
            "intakes.txt", description="client intake narratives"
        )
        if intakes_text:
            context_documents.append(
                (
                    "context:intakes",
                    intakes_text,
                    {
                        "doc_type": "context_reference",
                        "source_type": "client_intakes",
                        "description": "Aggregated client intake insights for Enlitens clients",
                        "filename": intakes_path.name if intakes_path else "intakes.txt",
                    },
                )
            )

        transcripts_text, transcripts_path = self._load_optional_context_text(
            "transcripts.txt", description="founder voice transcripts"
        )
        if transcripts_text:
            context_documents.append(
                (
                    "context:transcripts",
                    transcripts_text,
                    {
                        "doc_type": "context_reference",
                        "source_type": "founder_transcripts",
                        "description": "Founder transcript excerpts capturing Liz Wooten's voice",
                        "filename": transcripts_path.name if transcripts_path else "transcripts.txt",
                    },
                )
            )

        stl_report_text = self._extract_st_louis_report_text()
        if stl_report_text:
            context_documents.append(
                (
                    "context:stl_health_report",
                    stl_report_text,
                    {
                        "doc_type": "context_reference",
                        "source_type": "stl_health_report",
                        "description": "Regional health indicators for the St. Louis metro area",
                        "filename": self.st_louis_report.name if self.st_louis_report else "st_louis_report.pdf",
                    },
                )
            )

        for document_id, full_text, metadata in context_documents:
            try:
                if full_text.strip():
                    self.embedding_ingestion.ingest_document(
                        document_id=document_id,
                        full_text=full_text,
                        metadata={**metadata, "document_id": document_id},
                    )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Unable to ingest context document %s: %s", document_id, exc
                )

    def _load_optional_context_text(
        self, filename: str, *, description: str
    ) -> Tuple[Optional[str], Optional[Path]]:
        """Read optional context files from known locations."""

        for candidate in self._context_file_candidates(filename):
            if not candidate.exists():
                continue
            try:
                return candidate.read_text(encoding="utf-8"), candidate
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Failed to read %s at %s: %s", description, candidate, exc
                )
                return None, None

        logger.debug("No %s available for ingestion", description)
        return None, None

    def _context_file_candidates(self, filename: str) -> List[Path]:
        """Return candidate paths for locating optional context files."""

        return [
            self.context_dir / filename,
            self.context_dir / "enlitens_knowledge_base" / filename,
        ]

    def _extract_st_louis_report_text(self) -> Optional[str]:
        """Extract markdown text from the optional St. Louis health report."""

        if not self.st_louis_report or not self.st_louis_report.exists():
            return None

        try:
            extraction = self.pdf_extractor.extract(str(self.st_louis_report))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Failed to extract St. Louis health report for ingestion: %s", exc
            )
            return None

        if isinstance(extraction, dict):
            archival = extraction.get("archival_content", {})
            text = archival.get("full_document_text_markdown")
            if text:
                return text
            legacy_text = extraction.get("text")
            if isinstance(legacy_text, str) and legacy_text.strip():
                return legacy_text

        logger.debug("St. Louis health report extraction did not yield text for ingestion")
        return None

    def _load_st_louis_context(self) -> Dict[str, Any]:
        """Load St. Louis regional context."""
        context = {
            "demographics": {
                "population": "2.8 million metro area",
                "mental_health_challenges": [
                    "High trauma rates from urban violence",
                    "Complex PTSD and intergenerational trauma",
                    "ADHD and executive function challenges",
                    "Anxiety and depression in high-stress environments",
                    "Treatment resistance and medication questions",
                    "Stigma and access barriers",
                    "Cultural diversity and inclusion needs"
                ],
                "socioeconomic_factors": [
                    "Poverty and unemployment challenges",
                    "Housing instability and homelessness",
                    "Transportation barriers to care",
                    "Insurance coverage gaps",
                    "Racial and ethnic disparities"
                ]
            },
            "clinical_priorities": [
                "Trauma-informed neuroscience approaches",
                "ADHD executive function support",
                "Anxiety regulation techniques",
                "Cultural competence in therapy",
                "Community-based mental health solutions"
            ],
            "founder_voice": [
                "Traditional therapy missed the neurobiology",
                "Your brain isn't broken, it's adapting",
                "Neuroscience shows us the way forward",
                "Real therapy for real people in St. Louis",
                "Challenge the status quo of mental health treatment"
            ]
        }

        # Load additional context from St. Louis health report if provided
        if self.st_louis_report and self.st_louis_report.exists():
            try:
                context["health_report"] = self.pdf_extractor.extract(str(self.st_louis_report))
                logger.info(f"✅ Loaded St. Louis health report: {self.st_louis_report}")
            except Exception as e:
                logger.warning(f"Could not load St. Louis health report: {e}")

        return context

    def _extract_health_report_summary(self) -> Dict[str, Any]:
        """Summarise health report content for downstream context."""

        report_path: Optional[Path] = None
        if self.st_louis_report and self.st_louis_report.exists():
            report_path = self.st_louis_report
        else:
            candidate = self.context_dir / "enlitens_knowledge_base" / "st_louis_health_report.pdf"
            if candidate.exists():
                report_path = candidate

        if not report_path:
            return {}

        try:
            extraction = self.pdf_extractor.extract(str(report_path))
        except Exception as exc:
            logger.warning("Failed to extract health report summary: %s", exc)
            return {}

        if isinstance(extraction, dict):
            text = extraction.get("archival_content", {}).get("full_document_text_markdown", "")
        elif isinstance(extraction, str):
            text = extraction
        else:
            text = ""

        if not text:
            return {"path": str(report_path)}

        paragraphs = [
            " ".join(paragraph.split())[:360]
            for paragraph in re.split(r"\n{2,}", text)
            if paragraph and len(paragraph.strip()) > 160
        ]
        highlights = paragraphs[:20]

        regional_mentions = match_municipalities(text)

        theme_counter: Counter[str] = Counter()
        lowered = text.lower()
        for keyword, bucket in HEALTH_PRIORITY_TERMS.items():
            occurrences = lowered.count(keyword)
            if occurrences:
                theme_counter[bucket] += occurrences

        top_themes = theme_counter.most_common(15)
        locality_weight = sum(regional_mentions.values()) if regional_mentions else 0
        weighted_theme_signals: List[Dict[str, Any]] = []
        for theme, count in top_themes:
            weighted_theme_signals.append(
                {
                    "theme": theme,
                    "frequency": count,
                    "weighted_frequency": round(count + 0.5 * locality_weight, 2),
                    "locality_tags": sorted(
                        (regional_mentions or {}).items(),
                        key=lambda item: (-item[1], item[0]),
                    )[:10],
                }
            )

        return {
            "path": str(report_path),
            "highlights": highlights,
            "regional_mentions": regional_mentions,
            "priority_signals": top_themes,
            "weighted_theme_signals": weighted_theme_signals,
        }

    def _build_intake_registry(self, raw_text: Optional[str]) -> Dict[str, Any]:
        """Create a structured registry from client intake narratives."""

        if not raw_text:
            return {}

        entries: List[Dict[str, Any]] = []
        location_counter: Counter[str] = Counter()
        theme_counter: Counter[str] = Counter()
        theme_signals: Dict[str, Dict[str, Any]] = {}

        blocks = re.findall(r"\{([^{}]+)\}", raw_text, flags=re.DOTALL)
        for block in blocks:
            snippet = " ".join(block.split())
            if not snippet:
                continue

            regional_mentions = match_municipalities(snippet)
            for name, count in regional_mentions.items():
                location_counter[name] += count

            themes = self._infer_themes(snippet)
            locality_weight = sum(regional_mentions.values()) if regional_mentions else 0
            for theme in themes:
                theme_counter[theme] += 1
                signal_entry = theme_signals.setdefault(
                    theme,
                    {
                        "frequency": 0,
                        "weighted_frequency": 0.0,
                        "localities": Counter(),
                    },
                )
                signal_entry["frequency"] += 1
                signal_entry["weighted_frequency"] += 1 + 0.5 * locality_weight
                if regional_mentions:
                    signal_entry["localities"].update(regional_mentions)

            entries.append(
                {
                    "snippet": snippet[:420],
                    "regional_mentions": regional_mentions,
                    "themes": themes,
                }
            )

        top_locations = location_counter.most_common(30)
        top_themes = theme_counter.most_common(25)
        weighted_theme_signals: List[Dict[str, Any]] = []
        for theme, data in theme_signals.items():
            locality_tags = sorted(
                data.get("localities", {}).items(),
                key=lambda item: (-item[1], item[0]),
            )[:10]
            weighted_theme_signals.append(
                {
                    "theme": theme,
                    "frequency": data.get("frequency", 0),
                    "weighted_frequency": round(data.get("weighted_frequency", 0.0), 2),
                    "locality_tags": locality_tags,
                }
            )

        weighted_theme_signals.sort(
            key=lambda item: (-item.get("weighted_frequency", 0.0), item.get("theme", ""))
        )

        return {
            "entry_count": len(entries),
            "top_locations": top_locations,
            "location_counts": dict(location_counter),
            "top_themes": top_themes,
            "weighted_theme_signals": weighted_theme_signals,
            "entries": entries[:80],
        }

    def _build_transcript_registry(self, raw_text: Optional[str]) -> Dict[str, Any]:
        """Summarise founder transcripts for consistent downstream usage."""

        if not raw_text:
            return {}

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            return {}

        segments: List[str] = []
        buffer: List[str] = []
        for line in lines:
            buffer.append(line)
            joined = " ".join(buffer)
            if len(joined) >= 240:
                segments.append(joined[:360])
                buffer = []
            if len(segments) >= 60:
                break

        if buffer and len(segments) < 60:
            segments.append(" ".join(buffer)[:360])

        combined_text = " ".join(lines)
        regional_mentions = match_municipalities(combined_text)

        theme_counter: Counter[str] = Counter()
        theme_signals: Dict[str, Dict[str, Any]] = {}
        for segment in segments:
            segment_localities = match_municipalities(segment)
            locality_weight = sum(segment_localities.values()) if segment_localities else 0
            for theme in self._infer_themes(segment):
                theme_counter[theme] += 1
                signal_entry = theme_signals.setdefault(
                    theme,
                    {
                        "frequency": 0,
                        "weighted_frequency": 0.0,
                        "localities": Counter(),
                    },
                )
                signal_entry["frequency"] += 1
                signal_entry["weighted_frequency"] += 1 + 0.5 * locality_weight
                if segment_localities:
                    signal_entry["localities"].update(segment_localities)

        weighted_theme_signals: List[Dict[str, Any]] = []
        for theme, data in theme_signals.items():
            locality_tags = sorted(
                data.get("localities", {}).items(),
                key=lambda item: (-item[1], item[0]),
            )[:10]
            weighted_theme_signals.append(
                {
                    "theme": theme,
                    "frequency": data.get("frequency", 0),
                    "weighted_frequency": round(data.get("weighted_frequency", 0.0), 2),
                    "locality_tags": locality_tags,
                }
            )

        weighted_theme_signals.sort(
            key=lambda item: (-item.get("weighted_frequency", 0.0), item.get("theme", ""))
        )

        return {
            "segment_count": len(segments),
            "segments": segments,
            "regional_mentions": regional_mentions,
            "top_themes": theme_counter.most_common(20),
            "weighted_theme_signals": weighted_theme_signals,
        }

    @staticmethod
    def _normalize_locality_tags(locality_tags: Any) -> List[Tuple[str, float]]:
        """Normalize locality tag structures into name/count tuples."""

        if not locality_tags:
            return []

        normalized: List[Tuple[str, float]] = []
        items: Iterable[Any]
        if isinstance(locality_tags, dict):
            items = locality_tags.items()
        else:
            items = locality_tags

        for item in items:
            name: Optional[str] = None
            count_value: float = 1.0
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("locality") or item.get("label") or "").strip()
                raw_count = (
                    item.get("count")
                    or item.get("mentions")
                    or item.get("signal_strength")
                    or 1
                )
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                name = str(item[0]).strip()
                raw_count = item[1]
            else:
                name = str(item).strip()
                raw_count = 1

            if not name:
                continue

            try:
                count_value = float(raw_count)
            except (TypeError, ValueError):
                count_value = 1.0

            normalized.append((name, count_value))

        return normalized

    @staticmethod
    def _income_band_to_value(band: Optional[str]) -> Optional[int]:
        """Convert a textual income band into an approximate numeric value."""

        if not band:
            return None

        normalized = band.lower().replace(",", "")
        matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)", normalized)
        if not matches:
            return None

        try:
            value = float(matches[0])
        except ValueError:
            return None

        if "k" in normalized and value < 1000:
            value *= 1000
        elif "m" in normalized and value < 1_000_000:
            value *= 1_000_000
        elif value < 1000 and "000" in normalized:
            value *= 1000

        return int(value)

    def _compile_theme_landscape(self) -> Dict[str, Any]:
        """Aggregate theme signals across context assets with locality awareness."""

        sources: Dict[str, Any] = {
            "intake": (self.intake_registry or {}).get("weighted_theme_signals", []),
            "transcript": (self.transcript_registry or {}).get("weighted_theme_signals", []),
            "health_report": (self.health_report_summary or {}).get("weighted_theme_signals", []),
        }

        theme_map: Dict[str, Dict[str, Any]] = {}
        for source, entries in sources.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                theme = entry.get("theme")
                if not theme:
                    continue

                try:
                    weighted = float(entry.get("weighted_frequency", entry.get("frequency", 0.0)))
                except (TypeError, ValueError):
                    weighted = 0.0
                try:
                    frequency = float(entry.get("frequency", 0.0))
                except (TypeError, ValueError):
                    frequency = 0.0

                record = theme_map.setdefault(
                    theme,
                    {
                        "total_weight": 0.0,
                        "sources": {},
                        "locality_counter": Counter(),
                    },
                )
                record["total_weight"] += weighted
                record["sources"][source] = {
                    "weighted_frequency": round(weighted, 2),
                    "frequency": round(frequency, 2),
                }
                for name, count in self._normalize_locality_tags(entry.get("locality_tags")):
                    record["locality_counter"][name] += count

        if not theme_map:
            return {
                "dominant_themes": [],
                "theme_gaps": [],
                "socioeconomic_contrast_flags": [],
            }

        def _sorted_localities(counter: Counter) -> List[Tuple[str, float]]:
            return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:10]

        dominant_themes: List[Dict[str, Any]] = []
        for theme, data in theme_map.items():
            dominant_themes.append(
                {
                    "theme": theme,
                    "total_weight": round(data["total_weight"], 2),
                    "source_breakdown": data["sources"],
                    "locality_tags": _sorted_localities(data["locality_counter"]),
                }
            )

        dominant_themes.sort(
            key=lambda item: (-item.get("total_weight", 0.0), item.get("theme", ""))
        )

        all_sources = [source for source in sources if isinstance(sources[source], list)]
        theme_gaps: List[Dict[str, Any]] = []
        for theme, data in theme_map.items():
            present_sources = sorted(data["sources"].keys())
            missing_sources = [source for source in all_sources if source not in data["sources"]]
            if missing_sources and present_sources:
                theme_gaps.append(
                    {
                        "theme": theme,
                        "present_sources": present_sources,
                        "missing_sources": missing_sources,
                        "total_weight": round(data["total_weight"], 2),
                        "locality_tags": _sorted_localities(data["locality_counter"]),
                    }
                )

        theme_gaps.sort(
            key=lambda item: (-item.get("total_weight", 0.0), item.get("theme", ""))
        )

        socioeconomic_flags: List[Dict[str, Any]] = []
        locality_reference = getattr(self, "locality_reference", {}) or {}
        for theme, data in theme_map.items():
            locality_counter: Counter = data["locality_counter"]
            if not locality_counter:
                continue

            profiles: List[Dict[str, Any]] = []
            income_values: List[int] = []
            delmar_detected = False
            for name, count in locality_counter.items():
                record = locality_reference.get(name.strip().lower())
                profile = {
                    "name": name,
                    "signal_strength": round(float(count), 2),
                }
                if record:
                    profile.update(
                        {
                            "median_income_band": record.median_income_band,
                            "demographics": record.demographic_descriptors,
                            "jurisdiction": record.jurisdiction,
                        }
                    )
                    income_value = self._income_band_to_value(record.median_income_band)
                    if income_value is not None:
                        income_values.append(income_value)
                    if "delmar" in record.name.lower() or "delmar" in record.demographic_descriptors.lower():
                        delmar_detected = True
                else:
                    profile.update(
                        {
                            "median_income_band": None,
                            "demographics": None,
                            "jurisdiction": None,
                        }
                    )
                profiles.append(profile)

            profiles.sort(
                key=lambda item: (-item.get("signal_strength", 0.0), item.get("name", ""))
            )

            income_gap: Optional[int] = None
            if len(income_values) >= 2:
                income_gap = max(income_values) - min(income_values)

            indicators: List[str] = []
            if income_gap is not None and income_gap >= 20_000:
                indicators.append("income_gap")
            if delmar_detected:
                indicators.append("delmar_indicator")

            if indicators:
                socioeconomic_flags.append(
                    {
                        "theme": theme,
                        "indicators": indicators,
                        "estimated_income_gap": income_gap,
                        "locality_profiles": profiles[:6],
                    }
                )

        socioeconomic_flags.sort(
            key=lambda item: (
                -(item.get("estimated_income_gap") or 0),
                item.get("theme", ""),
            )
        )

        return {
            "dominant_themes": dominant_themes,
            "theme_gaps": theme_gaps,
            "socioeconomic_contrast_flags": socioeconomic_flags,
        }

    def _identify_locality_backlog(self) -> List[Dict[str, Any]]:
        """Determine high-signal localities missing from the reference table."""

        if not self.locality_reference:
            return []

        known = set(self.locality_reference.keys())
        backlog: Dict[str, Dict[str, Any]] = {}

        def register(source: str, mapping: Dict[str, Any]) -> None:
            for raw_name, raw_count in (mapping or {}).items():
                if not raw_name:
                    continue
                try:
                    count = int(raw_count)
                except (TypeError, ValueError):
                    continue
                key = raw_name.strip().lower()
                if key in known:
                    continue
                entry = backlog.setdefault(
                    key,
                    {
                        "name": raw_name.strip(),
                        "source_counts": {},
                    },
                )
                entry["name"] = entry.get("name") or raw_name.strip()
                entry["source_counts"][source] = entry["source_counts"].get(source, 0) + count

        register("intake", self.intake_registry.get("location_counts", {}))
        register("transcript", self.transcript_registry.get("regional_mentions", {}))
        health_mentions = (
            self.health_report_summary.get("regional_mentions", {})
            if isinstance(self.health_report_summary, dict)
            else {}
        )
        register("health_report", health_mentions)

        backlog_list: List[Dict[str, Any]] = []
        for entry in backlog.values():
            signal = sum(entry["source_counts"].values())
            backlog_list.append(
                {
                    "name": entry["name"],
                    "source_counts": entry["source_counts"],
                    "signal_strength": signal,
                }
            )

        backlog_list.sort(key=lambda item: item["signal_strength"], reverse=True)
        return backlog_list[:25]

    def _build_regional_atlas(self) -> Dict[str, Any]:
        """Combine curated geography with observed intake/transcript signals."""

        atlas: Dict[str, Any] = {}
        for key, value in REGIONAL_CONTEXT_DATA.items():
            if isinstance(value, dict):
                atlas[key] = {sub_key: list(sub_value) for sub_key, sub_value in value.items()}
            else:
                atlas[key] = list(value)

        atlas["top_intake_locations"] = self.intake_registry.get("top_locations", [])
        atlas["intake_themes"] = self.intake_registry.get("top_themes", [])
        atlas["top_transcript_themes"] = self.transcript_registry.get("top_themes", [])

        health_locations = self.health_report_summary.get("regional_mentions", {}) if self.health_report_summary else {}
        atlas["top_health_report_locations"] = sorted(
            health_locations.items(), key=lambda item: (-item[1], item[0])
        )[:30]
        atlas["health_priority_signals"] = self.health_report_summary.get("priority_signals", []) if self.health_report_summary else []

        if getattr(self, "theme_landscape", None):
            atlas["dominant_themes"] = self.theme_landscape.get("dominant_themes", [])
            atlas["theme_gaps"] = self.theme_landscape.get("theme_gaps", [])
            atlas["socioeconomic_contrast_flags"] = self.theme_landscape.get(
                "socioeconomic_contrast_flags", []
            )

        locality_reference = getattr(self, "locality_reference", {}) or {}
        atlas["locality_reference"] = [
            record.to_dict() for record in sorted(locality_reference.values(), key=lambda item: item.name)
        ]
        atlas["locality_research_gaps"] = getattr(self, "locality_research_backlog", [])

        return atlas

    def _compile_locality_signals(
        self, document_localities: Dict[str, int]
    ) -> Dict[str, Any]:
        """Aggregate locality signals across document, intake, and transcript sources."""

        matches: List[Dict[str, Any]] = []
        gaps: List[Dict[str, Any]] = []

        locality_reference = getattr(self, "locality_reference", {}) or {}
        if not locality_reference:
            return {"matches": matches, "gaps": gaps}

        candidate_map: Dict[str, Dict[str, Any]] = {}

        def register_counts(source: str, mapping: Dict[str, Any]) -> None:
            for raw_name, raw_count in (mapping or {}).items():
                if not raw_name:
                    continue
                try:
                    count = int(raw_count)
                except (TypeError, ValueError):
                    continue
                key = raw_name.strip().lower()
                if key not in candidate_map:
                    candidate_map[key] = {
                        "display_name": raw_name.strip(),
                        "document_mentions": 0,
                        "intake_mentions": 0,
                        "transcript_mentions": 0,
                        "health_report_mentions": 0,
                    }
                candidate_map[key][f"{source}_mentions"] = candidate_map[key].get(f"{source}_mentions", 0) + count
                if not candidate_map[key].get("display_name"):
                    candidate_map[key]["display_name"] = raw_name.strip()

        register_counts("document", document_localities)
        register_counts("intake", self.intake_registry.get("location_counts", {}))
        register_counts("transcript", self.transcript_registry.get("regional_mentions", {}))
        health_mentions = (
            self.health_report_summary.get("regional_mentions", {})
            if isinstance(self.health_report_summary, dict)
            else {}
        )
        register_counts("health_report", health_mentions)

        for key, counts in candidate_map.items():
            signal = (
                counts.get("document_mentions", 0)
                + 0.6 * counts.get("intake_mentions", 0)
                + 0.6 * counts.get("transcript_mentions", 0)
                + 0.4 * counts.get("health_report_mentions", 0)
            )
            record = locality_reference.get(key)
            payload = {
                "name": record.name if record else counts.get("display_name", key),
                "document_mentions": counts.get("document_mentions", 0),
                "intake_mentions": counts.get("intake_mentions", 0),
                "transcript_mentions": counts.get("transcript_mentions", 0),
                "health_report_mentions": counts.get("health_report_mentions", 0),
                "signal_strength": round(signal, 2),
            }
            if record:
                payload["locality"] = record.to_dict()
                matches.append(payload)
            else:
                gaps.append(payload)

        matches.sort(key=lambda item: item["signal_strength"], reverse=True)
        gaps.sort(key=lambda item: item["signal_strength"], reverse=True)

        return {
            "matches": matches[:15],
            "gaps": gaps[:15],
        }

    @staticmethod
    def _infer_themes(text: str) -> List[str]:
        """Map free-text content onto high-level intake themes."""

        themes: List[str] = []
        lowered = text.lower()
        for keyword, label in INTAKE_THEME_KEYWORDS.items():
            if keyword in lowered and label not in themes:
                themes.append(label)
        return themes

    def _analyze_client_insights(self) -> Dict[str, Any]:
        """Analyze client intake data for enhanced context."""
        try:
            intakes_path = self.context_dir / "intakes.txt"
            analyzer = None
            if hasattr(self, 'extraction_tools') and hasattr(self.extraction_tools, 'analyze_client_intakes'):
                analyzer = self.extraction_tools.analyze_client_intakes
            return analyze_optional_context(
                intakes_path,
                description="client intake insights",
                analyzer=analyzer,
            )
        except Exception as e:
            logger.error(f"Error analyzing client insights: {e}")
            return {}

    def _analyze_founder_insights(self) -> Dict[str, Any]:
        """Analyze founder transcripts for voice patterns."""
        try:
            transcripts_path = self.context_dir / "transcripts.txt"
            analyzer = None
            if hasattr(self, 'extraction_tools') and hasattr(self.extraction_tools, 'analyze_founder_transcripts'):
                analyzer = self.extraction_tools.analyze_founder_transcripts
            return analyze_optional_context(
                transcripts_path,
                description="founder transcript insights",
                analyzer=analyzer,
            )
        except Exception as e:
            logger.error(f"Error analyzing founder insights: {e}")
            return {}

    async def _extract_pdf_text(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """Extract text from PDF with error handling."""
        for attempt in range(self.retry_attempts):
            try:
                logger.info(f"📄 Extracting text from: {pdf_path.name} (attempt {attempt + 1})")
                extraction_result = self.pdf_extractor.extract(str(pdf_path))

                if extraction_result and isinstance(extraction_result, dict):
                    # Enhanced extractor returns structured data
                    full_text = extraction_result.get('archival_content', {}).get('full_document_text_markdown', '')
                    if full_text and len(full_text.strip()) > 100:
                        logger.info(f"✅ Successfully extracted {len(full_text)} characters from {pdf_path.name}")
                        return extraction_result
                    else:
                        logger.warning(f"⚠️ Poor extraction quality from {pdf_path.name}, retrying...")
                        continue
                elif extraction_result and isinstance(extraction_result, str):
                    # Fallback for simple string extraction
                    if len(extraction_result.strip()) > 100:
                        logger.info(f"✅ Successfully extracted {len(extraction_result)} characters from {pdf_path.name}")
                        return {"archival_content": {"full_document_text_markdown": extraction_result}}
                    else:
                        logger.warning(f"⚠️ Poor extraction quality from {pdf_path.name}, retrying...")
                        continue
                else:
                    logger.warning(f"⚠️ Invalid extraction result type from {pdf_path.name}, retrying...")
                    continue

            except Exception as e:
                logger.error(f"❌ PDF extraction failed for {pdf_path.name} (attempt {attempt + 1}): {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue

        logger.error(f"❌ All PDF extraction attempts failed for {pdf_path.name}")
        logger.info(f"🔄 Trying fallback extraction for {pdf_path.name}")
        return self._extract_pdf_text_simple(pdf_path)

    def _extract_pdf_text_simple(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """
        Simple PDF text extraction using PyMuPDF as fallback.
        """
        try:
            import fitz  # PyMuPDF

            logger.info(f"📄 Fallback extraction from PDF: {pdf_path.name}")

            # Open PDF
            doc = fitz.open(str(pdf_path))
            text = ""

            # Extract text from all pages
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += page.get_text()
                text += "\n\n"  # Add spacing between pages

            doc.close()

            if len(text.strip()) > 100:
                logger.info(f"✅ Fallback extraction successful: {len(text)} characters")
                return {"archival_content": {"full_document_text_markdown": text}}
            else:
                logger.warning(f"⚠️ Fallback extraction too short: {len(text)} characters")
                return None

        except Exception as e:
            logger.error(f"❌ Fallback extraction failed for {pdf_path.name}: {e}")
            return None

    def _get_pdf_file_size(self, pdf_path: Path) -> Optional[int]:
        """Return the file size for the provided PDF path."""
        try:
            size = pdf_path.stat().st_size
            logger.debug(f"📏 PDF file size for {pdf_path.name}: {size} bytes")
            return size
        except OSError as exc:
            logger.warning(f"⚠️ Unable to determine file size for {pdf_path.name}: {exc}")
            return None

    def _get_page_count_from_extraction(self, extraction_result: Any, pdf_path: Path) -> Optional[int]:
        """Attempt to derive page count from extraction metadata or the original PDF."""
        page_count: Optional[int] = None

        if isinstance(extraction_result, dict):
            # Direct page count fields from extractor payloads
            direct_page_count = extraction_result.get('page_count')
            if isinstance(direct_page_count, int) and direct_page_count > 0:
                page_count = direct_page_count

            if page_count is None:
                pages_payload = extraction_result.get('pages')
                if isinstance(pages_payload, list) and pages_payload:
                    page_count = len(pages_payload)

            if page_count is None:
                candidate_paths = (
                    ('source_metadata', 'page_count'),
                    ('quality_metrics', 'page_count'),
                    ('document_stats', 'page_count'),
                )
                for top_level, nested_key in candidate_paths:
                    nested = extraction_result.get(top_level, {})
                    if isinstance(nested, dict):
                        nested_value = nested.get(nested_key)
                        if isinstance(nested_value, int) and nested_value > 0:
                            page_count = nested_value
                            break

        if page_count is not None:
            logger.debug(f"📄 Page count from extraction metadata for {pdf_path.name}: {page_count}")
            return page_count

        # Fallback: inspect PDF directly if libraries are available
        try:
            import fitz  # type: ignore

            try:
                document = fitz.open(str(pdf_path))
            except Exception as exc:
                logger.debug(f"⚠️ Unable to open PDF for page count ({pdf_path.name}): {exc}")
                return None

            try:
                page_total = document.page_count
                logger.debug(f"📄 Page count from PDF for {pdf_path.name}: {page_total}")
                return page_total
            finally:
                document.close()
        except Exception:
            # fitz may be unavailable in lightweight environments
            logger.debug(f"ℹ️ PyMuPDF not available; skipping page count fallback for {pdf_path.name}")
            return None

    async def _load_progress(self) -> EnlitensKnowledgeBase:
        """Load progress from temporary file."""
        try:
            if self.temp_file.exists():
                with open(self.temp_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                knowledge_base = EnlitensKnowledgeBase.model_validate(data)
                logger.info(f"📋 Loaded progress: {len(knowledge_base.documents)} documents processed")
                return knowledge_base
            else:
                logger.info("📋 No previous progress found, starting fresh")
                return EnlitensKnowledgeBase()
        except Exception as e:
            logger.error(f"❌ Error loading progress: {e}")
            return EnlitensKnowledgeBase()

    async def _save_progress(self, knowledge_base: EnlitensKnowledgeBase,
                           processed_count: int, total_files: int):
        """Save progress to temporary file."""
        try:
            knowledge_base.total_documents = len(knowledge_base.documents)
            with open(self.temp_file, 'w', encoding='utf-8') as f:
                json.dump(knowledge_base.model_dump(), f, indent=2, default=str)
            logger.info(f"💾 Progress saved: {processed_count}/{total_files} documents processed")

            latest_path = self.output_file.parent / "enlitens_knowledge_base_latest.json"
            tmp_latest = latest_path.with_suffix(".json.tmp") if latest_path.suffix else latest_path.with_name(f"{latest_path.name}.tmp")
            try:
                with open(tmp_latest, 'w', encoding='utf-8') as latest_f:
                    json.dump(knowledge_base.model_dump(), latest_f, indent=2, default=str)
                os.replace(tmp_latest, latest_path)
            except Exception as latest_exc:
                logger.error(f"❌ Error updating latest knowledge base file: {latest_exc}")
        except Exception as e:
            logger.error(f"❌ Error saving progress: {e}")

    def _extract_document_passages(
        self,
        text: str,
        document_id: str,
        *,
        max_passages: int = 8,
        min_chars: int = 160,
        max_chars: int = 600,
    ) -> List[Dict[str, Any]]:
        """Slice the document into reusable passages for downstream agents."""

        if not text:
            return []

        normalized_paragraphs = [segment.strip() for segment in re.split(r"\n{2,}", text) if segment.strip()]
        seen: set[str] = set()
        passages: List[Dict[str, Any]] = []

        for paragraph in normalized_paragraphs:
            cleaned = " ".join(paragraph.split())
            if len(cleaned) < min_chars:
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)

            index = len(passages) + 1
            snippet = cleaned[:max_chars]
            metadata = {
                "document_id": document_id,
                "source_type": "document",
                "chunk_index": index - 1,
                "doc_type": "primary_source",
            }

            passages.append(
                {
                    "chunk_id": f"{document_id}::doc::{index}",
                    "text": snippet,
                    "document_id": document_id,
                    "source_type": "document",
                    "metadata": metadata,
                }
            )

            if len(passages) >= max_passages:
                break

        return passages

    def _derive_verbatim_quotes(self, raw_text: Optional[str], max_quotes: int = 5) -> List[str]:
        """Extract representative quotes from raw intake narratives."""

        if not raw_text:
            return []

        quotes: List[str] = []
        seen: set[str] = set()

        quote_pattern = re.compile(r'[“\"]([^”\"]{30,320})[”\"]', re.DOTALL)
        for match in quote_pattern.finditer(raw_text):
            fragment = match.group(0).strip()
            cleaned = " ".join(fragment.split())
            if cleaned and cleaned not in seen:
                quotes.append(cleaned)
                seen.add(cleaned)
            if len(quotes) >= max_quotes:
                break

        if len(quotes) < max_quotes:
            sentences = re.split(r"(?<=[.!?])\s+", raw_text)
            for sentence in sentences:
                cleaned_sentence = " ".join(sentence.strip().split())
                if len(cleaned_sentence) < 80:
                    continue
                if cleaned_sentence.lower().startswith(("note", "page", "chapter")):
                    continue
                normalized = cleaned_sentence
                if not normalized.startswith(("\"", "“")):
                    normalized = f'"{normalized}"'
                if normalized not in seen:
                    quotes.append(normalized)
                    seen.add(normalized)
                if len(quotes) >= max_quotes:
                    break

        return quotes[:max_quotes]

    async def _create_processing_context(self, text: str, document_id: str) -> Dict[str, Any]:
        """Create processing context for the supervisor."""
        # Analyze client and founder data for enhanced context
        client_analysis = self._analyze_client_insights()
        founder_analysis = self._analyze_founder_insights()

        raw_client_context = client_analysis.get("raw_content") if isinstance(client_analysis, dict) else None
        raw_founder_context = founder_analysis.get("raw_content") if isinstance(founder_analysis, dict) else None

        document_passages = self._extract_document_passages(text, document_id)

        derived_quotes: List[str] = []
        if isinstance(client_analysis, dict):
            existing_quotes = client_analysis.get("verbatim_quotes") or []
            if isinstance(existing_quotes, list):
                derived_quotes.extend(str(quote).strip() for quote in existing_quotes if str(quote).strip())
        if raw_client_context:
            for quote in self._derive_verbatim_quotes(raw_client_context):
                if quote not in derived_quotes:
                    derived_quotes.append(quote)

        client_insights = {
            "challenges": self.st_louis_context["demographics"]["mental_health_challenges"],
            "priorities": self.st_louis_context["clinical_priorities"],
            "enhanced_analysis": client_analysis,
            "topic_modeling": client_analysis.get("topic_modeling", {}),
            "sentiment_analysis": client_analysis.get("sentiment_analysis", {}),
            "pain_points": client_analysis.get("pain_points", []),
            "key_themes": client_analysis.get("key_themes", []),
        }
        if raw_client_context:
            client_insights["raw_context"] = raw_client_context
        if derived_quotes:
            client_insights["verbatim_quotes"] = derived_quotes[:5]

        founder_insights = {
            "voice_characteristics": self.st_louis_context["founder_voice"],
            "clinical_philosophy": [
                "Bottom-up sensory meets top-down cognitive",
                "Neuroplasticity as hope",
                "Interoceptive awareness foundation",
                "Executive function neuroscience support",
            ],
            "enhanced_analysis": founder_analysis,
            "topic_modeling": founder_analysis.get("topic_modeling", {}),
            "sentiment_analysis": founder_analysis.get("sentiment_analysis", {}),
            "voice_profile": founder_analysis.get("voice_characteristics", {}),
            "key_messages": founder_analysis.get("key_messages", []),
        }
        if raw_founder_context:
            founder_insights["raw_context"] = raw_founder_context

        document_localities = match_municipalities(text)
        sorted_localities = sorted(
            document_localities.items(), key=lambda item: (-item[1], item[0])
        )[:20]
        if sorted_localities:
            client_insights["regional_focus"] = [name for name, _ in sorted_localities]

        locality_context = self._compile_locality_signals(document_localities)
        if locality_context["matches"]:
            client_insights["locality_briefings"] = [
                {
                    "name": match["name"],
                    "jurisdiction": match["locality"].get("jurisdiction"),
                    "median_income_band": match["locality"].get("median_income_band"),
                    "demographic_descriptors": match["locality"].get("demographic_descriptors"),
                    "signal_strength": match["signal_strength"],
                }
                for match in locality_context["matches"][:6]
            ]

        regional_atlas_source = getattr(self, "regional_atlas", None)
        regional_atlas = copy.deepcopy(regional_atlas_source) if regional_atlas_source else {}
        if regional_atlas and sorted_localities:
            regional_atlas["document_localities"] = sorted_localities
        if regional_atlas and locality_context["matches"]:
            regional_atlas["document_locality_context"] = locality_context["matches"]

        intake_registry = getattr(self, "intake_registry", {}) or {}
        transcript_registry = getattr(self, "transcript_registry", {}) or {}
        health_summary = getattr(self, "health_report_summary", {}) or {}
        theme_landscape = getattr(self, "theme_landscape", {}) or {}

        return {
            "document_id": document_id,
            "document_text": text,
            "client_insights": client_insights,
            "founder_insights": founder_insights,
            "st_louis_context": self.st_louis_context["demographics"],
            "locality_context": locality_context,
            "insight_registry": {
                "client": client_analysis,
                "founder": founder_analysis,
            },
            "raw_client_context": raw_client_context,
            "raw_founder_context": raw_founder_context,
            "processing_stage": "initial",
            "document_passages": document_passages,
            "intake_registry": intake_registry,
            "transcript_registry": transcript_registry,
            "regional_atlas": regional_atlas,
            "health_report_summary": health_summary,
            "document_locality_matches": sorted_localities,
            "locality_research_gaps": locality_context["gaps"],
            "dominant_themes": theme_landscape.get("dominant_themes", []),
            "theme_gaps": theme_landscape.get("theme_gaps", []),
            "socioeconomic_contrast_flags": theme_landscape.get(
                "socioeconomic_contrast_flags", []
            ),
        }

    async def process_document(self, pdf_path: Path) -> Optional[EnlitensKnowledgeEntry]:
        """Process a single document through the complete multi-agent system."""
        document_id = pdf_path.stem

        try:
            logger.info(f"🧠 Starting multi-agent processing: {document_id}")

            # Extract text from PDF
            logger.info(f"📄 Extracting text from PDF: {pdf_path.name}")
            extraction_result = await self._extract_pdf_text(pdf_path)
            if not extraction_result:
                logger.error(f"❌ No text extracted from {document_id}")
                return None

            # Extract the full text from the structured result
            if isinstance(extraction_result, dict) and 'archival_content' in extraction_result:
                text = extraction_result['archival_content'].get('full_document_text_markdown', '')
                if not text:
                    logger.error(f"❌ No full text found in extraction result for {document_id}")
                    return None
            elif isinstance(extraction_result, str):
                text = extraction_result
            else:
                logger.error(f"❌ Unexpected extraction result type for {document_id}: {type(extraction_result)}")
                return None

            logger.info(f"✅ Text extracted successfully: {len(text)} characters")

            # Collect file metadata for knowledge entry
            file_size_bytes = self._get_pdf_file_size(pdf_path)

            # Entity enrichment via ExtractionTeam
            logger.info(f"🧬 Running entity extraction for {document_id}")
            entities = await self.extraction_team.extract_entities(extraction_result)

            total_entities = 0
            if isinstance(entities, dict):
                for value in entities.values():
                    if value is None:
                        continue
                    if isinstance(value, int):
                        total_entities += value
                    elif isinstance(value, (list, tuple, set)):
                        total_entities += len(value)
                    elif isinstance(value, dict):
                        total_entities += len(value)
                    else:
                        # Fallback: count unexpected scalar as single entity
                        total_entities += 1

            logger.info(f"✅ Entity extraction complete: {total_entities} entities")

            # Create processing context
            logger.info(f"🔧 Creating processing context for {document_id}")
            context = await self._create_processing_context(text, document_id)
            context["extracted_entities"] = entities
            logger.info(f"✅ Processing context created")

            # Process through supervisor (multi-agent system)
            logger.info(f"🚀 Starting supervisor processing for {document_id}")
            start_time = time.time()
            result = await self.supervisor.process_document(context)
            processing_time = time.time() - start_time

            logger.info(f"⏱️ Supervisor processing completed in {processing_time:.2f}s")

            if not result:
                logger.error(f"❌ Multi-agent processing returned no result for {document_id}")
                return None

            supervisor_status = result.get("supervisor_status")
            if supervisor_status not in {"completed", "completed_with_issues"}:
                logger.error(f"❌ Multi-agent processing failed for {document_id}: {result}")
                return None

            if supervisor_status == "completed_with_issues":
                logger.warning(
                    "⚠️ Document %s completed with issues; persisting partial outputs",
                    document_id,
                )

            result.setdefault("agent_outputs", {})["extracted_entities"] = entities

            if self.embedding_ingestion:
                try:
                    ingestion_metadata = {
                        "document_id": document_id,
                        "filename": pdf_path.name,
                        "doc_type": context.get("doc_type"),
                        "processing_timestamp": datetime.utcnow().isoformat(),
                        "quality_score": result.get("quality_score"),
                        "confidence_score": result.get("confidence_score"),
                        "supervisor_status": supervisor_status,
                    }
                    self.embedding_ingestion.ingest_document(
                        document_id=document_id,
                        full_text=result.get("document_text", text),
                        agent_outputs=result.get("agent_outputs", {}),
                        metadata=ingestion_metadata,
                        rebuild=True,
                    )
                except Exception as exc:
                    logger.warning("⚠️ Vector ingestion failed for %s: %s", document_id, exc)

            # Convert result to EnlitensKnowledgeEntry format
            logger.info(f"🔄 Converting result to knowledge entry for {document_id}")
            page_count = self._get_page_count_from_extraction(extraction_result, pdf_path)
            knowledge_entry = await self._convert_to_knowledge_entry(
                result,
                document_id,
                processing_time,
                file_size=file_size_bytes,
                page_count=page_count,
                filename=pdf_path.name,
                fallback_full_text=text,
            )

            logger.info(
                "✅ Document %s processed with status %s in %.2fs",
                document_id,
                supervisor_status,
                processing_time,
            )
            return knowledge_entry

        except Exception as e:
            logger.error(f"❌ Error processing document {document_id}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    async def _convert_to_knowledge_entry(
        self,
        result: Dict[str, Any],
        document_id: str,
        processing_time: float,
        *,
        file_size: Optional[int] = None,
        page_count: Optional[int] = None,
        filename: Optional[str] = None,
        fallback_full_text: str = "",
    ) -> EnlitensKnowledgeEntry:
        """Convert multi-agent result to EnlitensKnowledgeEntry format."""
        try:
            from src.models.enlitens_schemas import (
                DocumentMetadata, ExtractedEntities, RebellionFramework,
                MarketingContent, SEOContent, WebsiteCopy, BlogContent,
                SocialMediaContent, EducationalContent, ClinicalContent,
                ResearchContent, ContentCreationIdeas, ClientProfileSet
            )

            # Determine full document text to persist for verification and metadata
            full_document_text = result.get("document_text") or fallback_full_text or ""
            logger.info(f"📄 Storing full document text: {len(full_document_text)} characters")

            # Create metadata
            metadata = DocumentMetadata(
                document_id=document_id,
                filename=filename or f"{document_id}.pdf",
                processing_timestamp=datetime.now(),
                processing_time=processing_time,
                word_count=len(full_document_text.split()) if full_document_text else 0,
                file_size=file_size,
                page_count=page_count,
                full_text_stored=bool(full_document_text),
            )

            # Extract content from agent outputs
            agent_outputs = result.get("agent_outputs", {})

            # Extract entities (simplified for now)
            extracted_entities_payload = agent_outputs.get("extracted_entities", {})
            entities = ExtractedEntities()
            if extracted_entities_payload:
                biomedical = [e.get('text', '') for e in extracted_entities_payload.get('biomedical', []) if e.get('text')]
                neuroscience = [e.get('text', '') for e in extracted_entities_payload.get('neuroscience', []) if e.get('text')]
                clinical = [e.get('text', '') for e in extracted_entities_payload.get('clinical', []) if e.get('text')]
                statistical = [e.get('text', '') for e in extracted_entities_payload.get('statistical', []) if e.get('text')]

                entities.biomedical_entities = biomedical
                entities.neuroscience_entities = neuroscience
                entities.clinical_entities = clinical
                entities.statistical_entities = statistical
                entities.total_entities = sum(
                    len(bucket)
                    for bucket in (biomedical, neuroscience, clinical, statistical)
                )

            # Get rebellion framework
            rebellion_data = agent_outputs.get("rebellion_framework", {})
            rebellion_framework = RebellionFramework()
            if rebellion_data:
                for field, value in rebellion_data.items():
                    if hasattr(rebellion_framework, field) and isinstance(value, list):
                        setattr(rebellion_framework, field, value)

            # Get clinical content
            clinical_data = agent_outputs.get("clinical_content", {})
            clinical_content = ClinicalContent()
            if clinical_data:
                for field, value in clinical_data.items():
                    if hasattr(clinical_content, field) and isinstance(value, list):
                        setattr(clinical_content, field, value)

            # Get educational content
            educational_data = agent_outputs.get("educational_content", {})
            educational_content = EducationalContent()
            if educational_data:
                for field, value in educational_data.items():
                    if hasattr(educational_content, field) and isinstance(value, list):
                        setattr(educational_content, field, value)

            # Get research content
            research_data = agent_outputs.get("research_content", {})
            research_content = ResearchContent()
            if research_data:
                for field, value in research_data.items():
                    if hasattr(research_content, field) and isinstance(value, list):
                        setattr(research_content, field, value)

            # Get marketing content
            marketing_data = agent_outputs.get("marketing_content", {})
            marketing_content = MarketingContent()
            if marketing_data:
                for field, value in marketing_data.items():
                    if hasattr(marketing_content, field) and isinstance(value, list):
                        setattr(marketing_content, field, value)

            # Get SEO content
            seo_data = agent_outputs.get("seo_content", {})
            seo_content = SEOContent()
            if seo_data:
                filtered_seo_data = {
                    field: value
                    for field, value in seo_data.items()
                    if field != "title_tags"
                }
                for field, value in filtered_seo_data.items():
                    if hasattr(seo_content, field) and isinstance(value, list):
                        setattr(seo_content, field, value)

            # Get website copy
            website_data = agent_outputs.get("website_copy", {})
            website_copy = WebsiteCopy()
            if website_data:
                allowed_website_fields = set(WebsiteCopy.model_fields.keys())
                for field, value in website_data.items():
                    if field not in allowed_website_fields:
                        logger.debug(
                            "Dropping deprecated website_copy field '%s' for document %s",
                            field,
                            document_id,
                        )
                        continue
                    if isinstance(value, list):
                        setattr(website_copy, field, value)

            # Get social media content
            social_data = agent_outputs.get("social_media_content", {})
            social_media_content = SocialMediaContent()
            if social_data:
                for field, value in social_data.items():
                    if hasattr(social_media_content, field) and isinstance(value, list):
                        setattr(social_media_content, field, value)

            # Get content creation ideas
            ideas_data = agent_outputs.get("content_creation_ideas", {})
            content_creation_ideas = ContentCreationIdeas()
            if ideas_data:
                for field, value in ideas_data.items():
                    if hasattr(content_creation_ideas, field) and isinstance(value, list):
                        setattr(content_creation_ideas, field, value)

            # Get blog content with validation context for statistics verification
            blog_data = agent_outputs.get("blog_content", {})
            blog_context = {"source_text": full_document_text} if full_document_text else {}
            try:
                blog_content = BlogContent.model_validate(blog_data or {}, context=blog_context)
            except Exception as exc:
                logger.warning(
                    "⚠️ Failed to validate blog content for %s: %s", document_id, exc
                )
                blog_content = BlogContent()
            if blog_data:
                for field, value in blog_data.items():
                    if hasattr(blog_content, field) and isinstance(value, list):
                        setattr(blog_content, field, value)

            client_profiles_data = agent_outputs.get("client_profiles")
            client_profiles = None
            if client_profiles_data:
                try:
                    client_profiles = ClientProfileSet.model_validate(client_profiles_data or {})
                except Exception as exc:
                    logger.warning(
                        "⚠️ Failed to validate client profiles for %s: %s", document_id, exc
                    )

            entry = EnlitensKnowledgeEntry(
                metadata=metadata,
                extracted_entities=entities,
                rebellion_framework=rebellion_framework,
                marketing_content=marketing_content,
                seo_content=seo_content,
                website_copy=website_copy,
                blog_content=blog_content,
                social_media_content=social_media_content,
                educational_content=educational_content,
                clinical_content=clinical_content,
                research_content=research_content,
                content_creation_ideas=content_creation_ideas,
                client_profiles=client_profiles,
                full_document_text=full_document_text  # CRITICAL: Store for citation verification
            )

            # Enforce terminology policy across all text fields
            try:
                sanitized = sanitize_structure(entry.model_dump())
                if sanitized != entry.model_dump():
                    entry = EnlitensKnowledgeEntry.model_validate(sanitized)
                    logger.info("🔧 Applied terminology sanitizer to knowledge entry: %s", document_id)
            except Exception as _san_exc:
                logger.warning("Terminology sanitizer failed for %s: %s", document_id, _san_exc)

            # Log if any banned terms remain
            try:
                dump_text = json.dumps(entry.model_dump(), ensure_ascii=False)
                if contains_banned_terms(dump_text):
                    logger.warning("⚠️ Banned terminology detected post-sanitize for %s", document_id)
            except Exception:
                pass

            return entry

        except Exception as e:
            logger.error(f"Error converting to knowledge entry: {e}")
            # Return minimal valid entry with document text if available
            full_document_text = ""
            if isinstance(result, dict):
                full_document_text = result.get("document_text") or fallback_full_text or ""
            return EnlitensKnowledgeEntry(
                metadata=DocumentMetadata(
                    document_id=document_id,
                    filename=filename or f"{document_id}.pdf",
                    processing_timestamp=datetime.now(),
                    file_size=file_size,
                    page_count=page_count,
                    full_text_stored=bool(full_document_text),
                ),
                extracted_entities=ExtractedEntities(),
                rebellion_framework=RebellionFramework(),
                marketing_content=MarketingContent(),
                seo_content=SEOContent(),
                website_copy=WebsiteCopy(),
                blog_content=BlogContent(),
                social_media_content=SocialMediaContent(),
                educational_content=EducationalContent(),
                clinical_content=ClinicalContent(),
                research_content=ResearchContent(),
                content_creation_ideas=ContentCreationIdeas(),
                client_profiles=None,
                full_document_text=full_document_text  # Store even in fallback case
            )

    async def process_corpus(self):
        """Process the entire corpus using the multi-agent system."""
        start_time = time.time()  # Track processing start time
        failed_count = 0
        try:
            logger.info("🚀 Starting MULTI-AGENT Enlitens Corpus Processing")
            logger.info(f"📁 Input directory: {self.input_dir}")
            logger.info(f"📄 Output file: {self.output_file}")
            logger.info(f"📊 Log file: {log_filename} (comprehensive log for all 344 files)")
            logger.info(f"🏙️ St. Louis context: {len(self.st_louis_context)} categories loaded")

            # Check system resources
            await self._check_system_resources()

            # Get list of PDF files
            pdf_files = list(self.input_dir.glob("*.pdf"))
            total_files = len(pdf_files)
            logger.info(f"📚 Found {total_files} PDF files to process")

            if total_files == 0:
                logger.error("❌ No PDF files found in input directory")
                return

            # Ensure the supervisor and all specialized agents are ready
            if not self.supervisor.is_initialized:
                logger.info("🧠 Initializing multi-agent supervisor and specialized agents")
                init_success = await self.supervisor.initialize()
                if not init_success:
                    logger.error("❌ Supervisor initialization failed; aborting corpus processing")
                    return

            self.knowledge_base = await self._load_progress()
            processed_docs = {doc.metadata.document_id for doc in self.knowledge_base.documents}

            post_monitor_stats({
                "status": "running",
                "total_documents": total_files,
                "documents_processed": len(self.knowledge_base.documents),
                "documents_failed": failed_count,
            })

            # Process documents with multi-agent system
            for i, pdf_path in enumerate(pdf_files):
                if pdf_path.stem in processed_docs:
                    logger.info(f"⏭️ Skipping already processed: {pdf_path.name}")
                    continue

                logger.info(f"📖 Processing file {i+1}/{total_files}: {pdf_path.name}")

                post_monitor_stats({
                    "status": "processing",
                    "current_document": pdf_path.name,
                    "current_index": i + 1,
                    "total_documents": total_files,
                    "documents_processed": len(self.knowledge_base.documents),
                    "documents_failed": failed_count,
                })

                doc_start_time = time.time()

                # Process document with multi-agent system
                knowledge_entry = await self.process_document(pdf_path)
                doc_duration = time.time() - doc_start_time
                if knowledge_entry:
                    self.knowledge_base.documents.append(knowledge_entry)
                    logger.info(f"✅ Successfully processed: {pdf_path.name}")
                    post_monitor_stats({
                        "status": "processing",
                        "documents_processed": len(self.knowledge_base.documents),
                        "documents_failed": failed_count,
                        "last_document": pdf_path.name,
                        "last_duration": round(doc_duration, 2),
                    })
                else:
                    logger.error(f"❌ Failed to process: {pdf_path.name}")
                    failed_count += 1
                    post_monitor_stats({
                        "status": "processing",
                        "documents_processed": len(self.knowledge_base.documents),
                        "documents_failed": failed_count,
                        "last_document": pdf_path.name,
                        "last_error": "processing_failed",
                    })

                # Save progress after each document
                await self._save_progress(self.knowledge_base, i+1, total_files)

                # Memory management
                if (i + 1) % 3 == 0:  # Every 3 documents
                    await self._cleanup_memory()

            # Save final results
            self.knowledge_base.total_documents = len(self.knowledge_base.documents)
            # Atomic write: write to temp then replace to reduce risk during transient I/O issues
            try:
                tmp_final = Path(str(self.output_file) + ".tmp")
                with open(tmp_final, 'w', encoding='utf-8') as f:
                    json.dump(self.knowledge_base.model_dump(), f, indent=2, default=str)
                os.replace(tmp_final, self.output_file)
            except Exception as e:
                logger.error(f"❌ Atomic write failed, falling back to direct write: {e}")
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump(self.knowledge_base.model_dump(), f, indent=2, default=str)

            # Clean up temporary file
            if self.temp_file.exists():
                self.temp_file.unlink()

            # Get final system status
            system_status = await self.supervisor.get_system_status()

            processing_duration = time.time() - start_time

            logger.info("🎉 MULTI-AGENT PROCESSING COMPLETE!")
            logger.info(f"📊 Final Results:")
            logger.info(f"   - Total documents processed: {len(self.knowledge_base.documents)}/{total_files}")
            logger.info(f"   - Average quality score: {system_status.get('average_quality_score', 0):.2f}")
            logger.info(f"   - Average confidence score: {self._calculate_avg_confidence():.2f}")
            logger.info(f"   - Total retries performed: {self._count_total_retries()}")
            logger.info(f"   - Generated file: {self.output_file}")
            logger.info(f"   - Comprehensive log: {log_filename}")
            logger.info(f"   - Processing duration: {processing_duration:.2f} seconds")

            # Quality breakdown
            quality_breakdown = self._get_quality_breakdown()
            logger.info(f"📈 Quality Breakdown:")
            for category, stats in quality_breakdown.items():
                logger.info(f"   - {category}: {stats['avg']:.2f} (min: {stats['min']:.2f}, max: {stats['max']:.2f})")

            post_monitor_stats({
                "status": "completed",
                "documents_processed": len(self.knowledge_base.documents),
                "documents_failed": failed_count,
                "total_documents": total_files,
                "runtime_seconds": round(processing_duration, 2),
            })

        except Exception as e:
            logger.error(f"❌ Error processing corpus: {e}")
            post_monitor_stats({"status": "error", "error": str(e)})
            raise
        finally:
            # Clean up resources
            await self._cleanup()
            post_monitor_stats({"status": "idle"})

    async def _check_system_resources(self):
        """Check system resources and provide recommendations."""
        try:
            import torch
            import psutil

            # GPU information
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"🔥 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                logger.warning("⚠️ No GPU available - using CPU")

            # Memory information
            memory = psutil.virtual_memory()
            logger.info(f"💾 RAM: {memory.total / 1e9:.1f}GB available / {memory.available / 1e9:.1f}GB available")

            # Disk space
            disk = psutil.disk_usage(self.input_dir)
            logger.info(f"💽 Disk: {disk.free / 1e9:.1f}GB available")

        except Exception as e:
            logger.warning(f"Could not check system resources: {e}")

    async def _cleanup_memory(self):
        """Clean up memory between processing batches."""
        try:
            import gc
            import torch

            # Force garbage collection
            gc.collect()

            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("🧹 GPU memory cleared")

            logger.info("🧹 Memory cleanup completed")

        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")

    async def _cleanup(self):
        """Clean up all resources."""
        try:
            logger.info("🧹 Cleaning up resources...")

            # Clean up supervisor and agents
            if hasattr(self, 'supervisor'):
                await self.supervisor.cleanup()

            # Clean up PDF extractor
            if hasattr(self, 'pdf_extractor'):
                await self.pdf_extractor.cleanup()

            # Final memory cleanup
            await self._cleanup_memory()

            logger.info("✅ Cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _calculate_avg_confidence(self) -> float:
        """Calculate average confidence score across all processed documents."""
        try:
            if not self.knowledge_base.documents:
                return 0.0

            total_confidence = 0.0
            count = 0

            for doc in self.knowledge_base.documents:
                # Check if document has quality metrics
                if hasattr(doc, 'confidence_score') and doc.confidence_score:
                    total_confidence += doc.confidence_score
                    count += 1
                elif hasattr(doc, 'validation_results') and doc.validation_results:
                    # Fallback to validation results
                    confidence = doc.validation_results.get('confidence_scoring', {}).get('confidence_score', 0)
                    if confidence:
                        total_confidence += confidence
                        count += 1

            return total_confidence / count if count > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating average confidence: {e}")
            return 0.0

    def _count_total_retries(self) -> int:
        """Count total retry attempts across all documents."""
        try:
            total_retries = 0

            for doc in self.knowledge_base.documents:
                if hasattr(doc, 'retry_count'):
                    total_retries += doc.retry_count
                elif hasattr(doc, 'validation_results') and doc.validation_results:
                    # Fallback to validation results
                    total_retries += doc.validation_results.get('retry_count', 0)

            return total_retries

        except Exception as e:
            logger.error(f"Error counting retries: {e}")
            return 0

    def _get_quality_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get quality breakdown statistics."""
        try:
            quality_stats = {}

            if not self.knowledge_base.documents:
                return quality_stats

            # Initialize categories
            categories = ["clinical_accuracy", "founder_voice", "marketing_effectiveness", "completeness", "fact_checking"]

            for category in categories:
                scores = []
                for doc in self.knowledge_base.documents:
                    if hasattr(doc, 'validation_results') and doc.validation_results:
                        score = doc.validation_results.get('quality_scores', {}).get(category, 0)
                        if score:
                            scores.append(score)

                if scores:
                    quality_stats[category] = {
                        "avg": sum(scores) / len(scores),
                        "min": min(scores),
                        "max": max(scores)
                    }

            return quality_stats

        except Exception as e:
            logger.error(f"Error getting quality breakdown: {e}")
            return {}

async def main():
    """Main function to run the multi-agent processor."""
    parser = argparse.ArgumentParser(description="Multi-Agent Enlitens Corpus Processor")
    parser.add_argument("--input-dir", required=True, help="Input directory containing PDF files")
    parser.add_argument("--output-file", required=True, help="Output JSON file")
    parser.add_argument("--st-louis-report", help="Path to St. Louis health report PDF")

    args = parser.parse_args()

    # Display startup banner
    log_startup_banner()

    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"❌ Input directory does not exist: {args.input_dir}")
        return

    # Create processor and run
    processor = MultiAgentProcessor(args.input_dir, args.output_file, args.st_louis_report)
    await processor.process_corpus()

if __name__ == "__main__":
    # Set environment variables for optimal performance
    os.environ["OLLAMA_MAX_LOADED_MODELS"] = "1"
    os.environ["OLLAMA_MAX_QUEUE"] = "1"
    os.environ["OLLAMA_RUNNERS_DIR"] = "/tmp/ollama-runners"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    asyncio.run(main())

