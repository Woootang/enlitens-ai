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

import asyncio
import json
import logging
import os
import sys
import time
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.supervisor_agent import SupervisorAgent
from src.agents.context_curator_agent import ContextCuratorAgent
from src.models.enlitens_schemas import EnlitensKnowledgeBase, EnlitensKnowledgeEntry
from src.extraction.enhanced_pdf_extractor import EnhancedPDFExtractor
from src.extraction.enhanced_extraction_tools import EnhancedExtractionTools
from src.agents.extraction_team import ExtractionTeam
from src.utils.enhanced_logging import setup_enhanced_logging, log_startup_banner
from src.retrieval.embedding_ingestion import EmbeddingIngestionPipeline
from src.utils.terminology import sanitize_structure, contains_banned_terms

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
        self.temp_file = Path(f"{output_file}.temp")

        # Initialize components
        self.supervisor = SupervisorAgent()
        self.context_curator = ContextCuratorAgent()  # NEW: Intelligent context curation
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
            except Exception as exc:
                logger.warning("⚠️ Failed to initialize vector ingestion pipeline: %s", exc)
                self.embedding_ingestion = None

        # St. Louis regional context
        self.st_louis_context = self._load_st_louis_context()

        # Processing configuration
        self.max_concurrent_documents = 1  # Sequential processing for memory management
        self.checkpoint_interval = 1  # Save after each document
        self.retry_attempts = 3

        logger.info("🚀 Multi-Agent Processor initialized")

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

    def _analyze_client_insights(self) -> Dict[str, Any]:
        """Analyze client intake data for enhanced context."""
        try:
            intakes_path = Path(__file__).parent / "enlitens_knowledge_base" / "intakes.txt"
            if intakes_path.exists():
                with open(intakes_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Use extraction tools to analyze intakes
                if hasattr(self, 'extraction_tools'):
                    intake_analysis = self.extraction_tools.analyze_client_intakes([content])
                    return intake_analysis
                else:
                    return {"raw_content": content[:1000]}
            else:
                logger.warning("Client intakes file not found")
                return {}
        except Exception as e:
            logger.error(f"Error analyzing client insights: {e}")
            return {}

    def _analyze_founder_insights(self) -> Dict[str, Any]:
        """Analyze founder transcripts for voice patterns."""
        try:
            transcripts_path = Path(__file__).parent / "enlitens_knowledge_base" / "transcripts.txt"
            if transcripts_path.exists():
                with open(transcripts_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Use extraction tools to analyze transcripts
                if hasattr(self, 'extraction_tools'):
                    founder_analysis = self.extraction_tools.analyze_founder_transcripts([content])
                    return founder_analysis
                else:
                    return {"raw_content": content[:1000]}
            else:
                logger.warning("Founder transcripts file not found")
                return {}
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
        except Exception as e:
            logger.error(f"❌ Error saving progress: {e}")

    async def _create_processing_context(self, text: str, document_id: str) -> Dict[str, Any]:
        """Create processing context for the supervisor."""
        # Analyze client and founder data for enhanced context
        client_analysis = self._analyze_client_insights()
        founder_analysis = self._analyze_founder_insights()

        return {
            "document_id": document_id,
            "document_text": text,
            "client_insights": {
                "challenges": self.st_louis_context["demographics"]["mental_health_challenges"],
                "priorities": self.st_louis_context["clinical_priorities"],
                "enhanced_analysis": client_analysis,
                "topic_modeling": client_analysis.get("topic_modeling", {}),
                "sentiment_analysis": client_analysis.get("sentiment_analysis", {}),
                "pain_points": client_analysis.get("pain_points", []),
                "key_themes": client_analysis.get("key_themes", []),
            },
            "founder_insights": {
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
            },
            "st_louis_context": self.st_louis_context["demographics"],
            "insight_registry": {
                "client": client_analysis,
                "founder": founder_analysis,
            },
            "processing_stage": "initial",
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

            # Entity enrichment via ExtractionTeam
            logger.info(f"🧬 Running entity extraction for {document_id}")
            entities = await self.extraction_team.extract_entities(extraction_result)
            logger.info(f"✅ Entity extraction complete: {sum(len(v) for v in entities.values())} entities")

            # Create processing context
            logger.info(f"🔧 Creating processing context for {document_id}")
            context = await self._create_processing_context(text, document_id)
            context["extracted_entities"] = entities
            logger.info(f"✅ Processing context created")

            # NEW: Curate intelligent context using pre-processing agents
            logger.info(f"🎯 Running intelligent context curation for {document_id}")
            try:
                from src.synthesis.ollama_client import VLLMClient
                llm_client = VLLMClient()  # For agent operations
                
                curated_context = await self.context_curator.curate_context(
                    paper_text=text,
                    entities=entities,
                    health_report_text=context.get("health_report", {}).get("text", ""),
                    llm_client=llm_client
                )
                
                # Add curated context to processing context
                context["curated_context"] = curated_context
                logger.info(f"✅ Context curation complete: ~{curated_context['token_estimate']['total_curated']:,} tokens")
            except Exception as e:
                logger.warning(f"⚠️ Context curation failed, proceeding without: {e}")
                context["curated_context"] = None

            # Process through supervisor (multi-agent system)
            logger.info(f"🚀 Starting supervisor processing for {document_id}")
            start_time = time.time()
            result = await self.supervisor.process_document(context)
            processing_time = time.time() - start_time

            logger.info(f"⏱️ Supervisor processing completed in {processing_time:.2f}s")

            if result and result.get("supervisor_status") == "completed":
                result.setdefault("agent_outputs", {})["extracted_entities"] = entities

                if self.embedding_ingestion:
                    try:
                        ingestion_metadata = {
                            "document_id": document_id,
                            "filename": pdf_path.name,
                            "doc_type": context.get("doc_type"),
                            "processing_timestamp": datetime.utcnow().isoformat(),
                            "quality_score": float(result.get("quality_score")) if result.get("quality_score") is not None else None,
                            "confidence_score": float(result.get("confidence_score")) if result.get("confidence_score") is not None else None,
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
                knowledge_entry = await self._convert_to_knowledge_entry(result, document_id, processing_time)

                logger.info(f"✅ Document {document_id} processed successfully in {processing_time:.2f}s")
                return knowledge_entry
            else:
                logger.error(f"❌ Multi-agent processing failed for {document_id}: {result}")
                return None

        except Exception as e:
            logger.error(f"❌ Error processing document {document_id}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    async def _convert_to_knowledge_entry(self, result: Dict[str, Any], document_id: str,
                                        processing_time: float) -> EnlitensKnowledgeEntry:
        """Convert multi-agent result to EnlitensKnowledgeEntry format."""
        try:
            from src.models.enlitens_schemas import (
                DocumentMetadata, ExtractedEntities, RebellionFramework,
                MarketingContent, SEOContent, WebsiteCopy, BlogContent,
                SocialMediaContent, EducationalContent, ClinicalContent,
                ResearchContent, ContentCreationIdeas
            )

            # Create metadata
            metadata = DocumentMetadata(
                document_id=document_id,
                filename=f"{document_id}.pdf",
                processing_timestamp=datetime.now(),
                processing_time=processing_time,
                word_count=len(result.get("document_text", "").split())
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
                for field, value in seo_data.items():
                    if hasattr(seo_content, field) and isinstance(value, list):
                        setattr(seo_content, field, value)

            # Get website copy
            website_data = agent_outputs.get("website_copy", {})
            website_copy = WebsiteCopy()
            if website_data:
                for field, value in website_data.items():
                    if hasattr(website_copy, field) and isinstance(value, list):
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

            # Get full document text for citation verification
            full_document_text = result.get("document_text", "")
            logger.info(f"📄 Storing full document text: {len(full_document_text)} characters")

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
            full_document_text = result.get("document_text", "") if isinstance(result, dict) else ""
            return EnlitensKnowledgeEntry(
                metadata=DocumentMetadata(
                    document_id=document_id,
                    filename=f"{document_id}.pdf",
                    processing_timestamp=datetime.now()
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

            # === NEW: Integrate Personas, Confidence Scoring, and External Search ===
            logger.info("="*80)
            logger.info("🎯 POST-PROCESSING: Personas, Confidence, & External Search")
            logger.info("="*80)
            
            # 1. Integrate Personas
            try:
                from src.agents.persona_integration_agent import load_and_integrate_personas
                logger.info("📊 Integrating 57 client personas...")
                self.knowledge_base = load_and_integrate_personas(self.knowledge_base.model_dump())
                logger.info("✅ Personas integrated")
            except Exception as e:
                logger.warning(f"⚠️  Persona integration failed: {e}")
            
            # 2. Calculate Confidence Scores
            try:
                from src.utils.confidence_scorer import ConfidenceScorer
                logger.info("📊 Calculating confidence scores...")
                scorer = ConfidenceScorer(low_threshold=0.5, high_threshold=0.8)
                
                # Add all documents to scorer
                for doc in self.knowledge_base.get("documents", []):
                    doc_id = doc.get("document_id", "unknown")
                    scorer.add_document(doc_id, doc)
                
                # Calculate scores
                scorer.calculate_scores()
                
                # Add to knowledge base
                kb_dict = self.knowledge_base if isinstance(self.knowledge_base, dict) else self.knowledge_base.model_dump()
                kb_dict = scorer.add_to_knowledge_base(kb_dict)
                self.knowledge_base = kb_dict
                
                logger.info("✅ Confidence scores calculated")
                
                # 3. External Search for Low-Confidence Entities
                low_conf_entities = scorer.get_low_confidence_entities()
                
                if low_conf_entities:
                    from src.retrieval.external_search import ExternalSearchClient
                    logger.info(f"🔍 Searching external APIs for {len(low_conf_entities)} low-confidence entities...")
                    
                    search_client = ExternalSearchClient()
                    self.knowledge_base = search_client.enrich_knowledge_base(
                        self.knowledge_base, 
                        low_conf_entities
                    )
                    search_client.close()
                    
                    logger.info("✅ External search completed")
                else:
                    logger.info("✅ No low-confidence entities need external search")
                    
            except Exception as e:
                logger.warning(f"⚠️  Confidence scoring/external search failed: {e}")
            
            logger.info("="*80)
            logger.info("✅ POST-PROCESSING COMPLETE")
            logger.info("="*80)
            
            # Save final results
            self.knowledge_base.total_documents = len(self.knowledge_base.get("documents", [])) if isinstance(self.knowledge_base, dict) else len(self.knowledge_base.documents)
            # Atomic write: write to temp then replace to reduce risk during transient I/O issues
            try:
                tmp_final = Path(str(self.output_file) + ".tmp")
                with open(tmp_final, 'w', encoding='utf-8') as f:
                    kb_to_save = self.knowledge_base if isinstance(self.knowledge_base, dict) else self.knowledge_base.model_dump()
                    json.dump(kb_to_save, f, indent=2, default=str)
                os.replace(tmp_final, self.output_file)
            except Exception as e:
                logger.error(f"❌ Atomic write failed, falling back to direct write: {e}")
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    kb_to_save = self.knowledge_base if isinstance(self.knowledge_base, dict) else self.knowledge_base.model_dump()
                    json.dump(kb_to_save, f, indent=2, default=str)

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

