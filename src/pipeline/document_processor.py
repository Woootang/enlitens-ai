"""
Main Document Processing Pipeline

This module orchestrates the complete PDF processing pipeline:
1. Extract PDF → Docling + Marker
2. Extract entities → Specialized models
3. Synthesize insights → Qwen3 32B
4. Validate quality → Automated checks
5. Save to JSON → Append to knowledge base
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import traceback
from types import SimpleNamespace

# Import our modules
from src.extraction.pdf_extractor import HybridExtractor
from src.extraction.quality_validator import ExtractionQualityValidator
from src.models.specialized_models import NeuroscienceEntityExtractor, ModelManager
from src.synthesis.ai_synthesizer import NeuroscienceSynthesizer, OllamaClient
from src.schema.knowledge_schema import (
    EnlitensKnowledgeDocument,
    SourceMetadata, 
    SourceMetadata,
    ArchivalContent,
    EntityExtraction,
    AISynthesis,
    ProcessingMetadata,
    KnowledgeBase
)
from src.retrieval.chunker import DocumentChunker
from src.retrieval.vector_store import QdrantVectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.retry import IntelligentRetryManager
from src.validation.layered_validation import LayeredValidationPipeline

from src.monitoring.observability import get_observability

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Main orchestrator for document processing pipeline
    
    Why this architecture:
    - Centralized coordination prevents deadlocks
    - Modular design allows easy component swapping
    - Error handling and retry logic
    - Quality validation at each stage
    - Checkpointing for resume capability
    """
    
    def __init__(self, 
                 pdf_input_dir: str = "./enlitens_corpus/input_pdfs",
                 output_dir: str = "./enlitens_corpus/output",
                 cache_dir: str = "./enlitens_corpus/cache_markdown",
                 ollama_url: str = "http://localhost:8000/v1",
                 ollama_model: str = "qwen2.5-32b-instruct-q4_k_m"):
        
        self.pdf_input_dir = Path(pdf_input_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.pdf_extractor = HybridExtractor(str(self.cache_dir))
        self.quality_validator = ExtractionQualityValidator()
        self.model_manager = ModelManager()
        self.entity_extractor = NeuroscienceEntityExtractor(self.model_manager)
        self.ollama_client = OllamaClient(ollama_url, ollama_model)
        self.synthesizer = NeuroscienceSynthesizer(self.ollama_client)
        self.chunker = DocumentChunker()
        self.vector_store = QdrantVectorStore()
        self.retriever = HybridRetriever(self.vector_store)
        self.retry_manager = IntelligentRetryManager()
        self.layered_validator = LayeredValidationPipeline()
        
        # Initialize knowledge base
        self.knowledge_base_path = self.output_dir / "enlitens-knowledge-core.json"
        self.knowledge_base = self._load_knowledge_base()
        
        logger.info("DocumentProcessor initialised with vLLM backend")

        # Observability
        self.observability = get_observability()

        logger.info("DocumentProcessor initialized")
    
    def _load_knowledge_base(self) -> KnowledgeBase:
        """Load existing knowledge base or create new one"""
        if self.knowledge_base_path.exists():
            try:
                with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return KnowledgeBase(**data)
            except Exception as e:
                logger.warning(f"Failed to load knowledge base: {e}. Creating new one.")
        
        return KnowledgeBase()
    
    def _save_knowledge_base(self):
        """Save knowledge base to JSON file"""
        try:
            with open(self.knowledge_base_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base.dict(), f, indent=2, ensure_ascii=False, default=str)
            logger.info("Knowledge base saved")
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")
    
    def process_document(self, pdf_path: str) -> Tuple[bool, Optional[EnlitensKnowledgeDocument], str]:
        """
        Process a single PDF document through the complete pipeline
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (success, document, error_message)
        """
        document_id = Path(pdf_path).stem
        logger.info(
            f"Processing document: {pdf_path}",
            extra={"document_id": document_id, "processing_stage": "start"}
        )

        try:
            # Stage 1: PDF Extraction
            logger.info("Stage 1: PDF Extraction")
            extraction_result = self.pdf_extractor.extract(pdf_path)

            chunks = self.chunker.chunk(
                extraction_result.get('full_text', ''),
                extraction_result.get('metadata', {}),
            )
            extraction_result['chunks'] = chunks
            self.vector_store.upsert(chunks)
            self.retriever.index_chunks(chunks)

            # Validate extraction quality
            extraction_metrics = self.quality_validator.validate_extraction(extraction_result)
            extraction_result['quality_score'] = extraction_metrics.overall_score
            if extraction_metrics.overall_score < 0.95:
                logger.warning(f"Extraction quality below threshold: {extraction_metrics.overall_score:.2f}")
            
            extraction_start = datetime.utcnow()
            extraction_perf = time.perf_counter()
            logger.info("Stage 1: PDF Extraction", extra={"processing_stage": "extraction"})
            try:
                with self.observability.start_span(
                    "pipeline.pdf_extraction",
                    {
                        "document.id": document_id,
                        "pdf.path": str(pdf_path),
                    },
                ) as span:
                    extraction_result = self.pdf_extractor.extract(pdf_path)
                    page_count = len(extraction_result.get('pages', [])) if isinstance(extraction_result, dict) else 0
                    span.set_attribute("extraction.page_count", page_count)
            except Exception as extraction_error:
                extraction_end = datetime.utcnow()
                self.observability.record_stage_timing(
                    document_id,
                    "PDF Extraction",
                    "HybridExtractor",
                    extraction_start,
                    extraction_end,
                    "error",
                    {"error": str(extraction_error)},
                )
                raise

            extraction_end = datetime.utcnow()
            extraction_duration = time.perf_counter() - extraction_perf
            self.observability.record_stage_timing(
                document_id,
                "PDF Extraction",
                "HybridExtractor",
                extraction_start,
                extraction_end,
                "success",
                {"duration_seconds": extraction_duration},
            )
            self.observability.check_latency_anomaly("pdf_extraction", extraction_duration)

            extraction_metrics = self.quality_validator.validate_extraction(extraction_result)
            extraction_score = getattr(extraction_metrics, "overall_score", 0.0)
            if extraction_score < 0.95:
                logger.warning(
                    f"Extraction quality below threshold: {extraction_score:.2f}",
                    extra={"document_id": document_id}
                )
            extraction_metrics_payload = {}
            if hasattr(extraction_metrics, "dict"):
                try:
                    extraction_metrics_payload = extraction_metrics.dict()
                except TypeError:
                    extraction_metrics_payload = getattr(extraction_metrics, "__dict__", {})
            else:
                extraction_metrics_payload = {"overall_score": extraction_score}
            self.observability.record_quality_metrics(
                document_id,
                {"stage": "extraction", **extraction_metrics_payload}
            )

            # Stage 2: Entity Extraction
            entity_start = datetime.utcnow()
            entity_perf = time.perf_counter()
            logger.info("Stage 2: Entity Extraction", extra={"processing_stage": "entity_extraction"})
            with self.observability.start_span(
                "pipeline.entity_extraction",
                {"document.id": document_id},
            ) as span:
                if not self.entity_extractor.initialize():
                    self.observability.record_stage_timing(
                        document_id,
                        "Entity Extraction",
                        "NeuroscienceEntityExtractor",
                        entity_start,
                        datetime.utcnow(),
                        "error",
                        {"reason": "initialization failed"},
                    )
                    self.observability.emit_alert(
                        "Entity extractor unavailable",
                        "error",
                        "Failed to initialize entity extractor",
                        {"document_id": document_id},
                    )
                    return False, None, "Entity extractor initialization failed"

                source_text = ""
                if isinstance(extraction_result, dict):
                    source_text = extraction_result.get('full_text', '')
                entities = self.entity_extractor.extract_entities(source_text)
                if isinstance(entities, dict):
                    total_entities = sum(len(v) for v in entities.values() if isinstance(v, list))
                    span.set_attribute("entity.count", total_entities)
            entity_end = datetime.utcnow()
            entity_duration = time.perf_counter() - entity_perf
            self.observability.record_stage_timing(
                document_id,
                "Entity Extraction",
                "NeuroscienceEntityExtractor",
                entity_start,
                entity_end,
                "success",
                {"duration_seconds": entity_duration},
            )
            self.observability.check_latency_anomaly("entity_extraction", entity_duration)

            # Stage 3: AI Synthesis
            synthesis_start = datetime.utcnow()
            synthesis_perf = time.perf_counter()
            logger.info("Stage 3: AI Synthesis", extra={"processing_stage": "synthesis"})
            if not self.ollama_client.is_available():
                logger.error("Ollama is not available")
                return False, None, "Ollama is not available"

            synthesis_result = self.synthesizer.synthesize(extraction_result, retriever=self.retriever)
                logger.error("vLLM inference service is not available")
                return False, None, "vLLM inference service is not available"
            
            synthesis_result = self.synthesizer.synthesize(extraction_result)
            
                self.observability.record_stage_timing(
                    document_id,
                    "AI Synthesis",
                    "NeuroscienceSynthesizer",
                    synthesis_start,
                    datetime.utcnow(),
                    "error",
                    {"reason": "ollama unavailable"},
                )
                self.observability.emit_alert(
                    "Ollama unavailable",
                    "error",
                    "Local Ollama server is not responding",
                    {"document_id": document_id},
                )
                return False, None, "Ollama is not available"

            with self.observability.start_span(
                "pipeline.ai_synthesis",
                {"document.id": document_id},
            ) as span:
                synthesis_result = self.synthesizer.synthesize(extraction_result)
                span.set_attribute("synthesis.quality_score", synthesis_result.quality_score)
                if getattr(synthesis_result, "validation_issues", None):
                    span.set_attribute(
                        "synthesis.issues",
                        ",".join(synthesis_result.validation_issues)
                    )
            synthesis_end = datetime.utcnow()
            synthesis_duration = time.perf_counter() - synthesis_perf
            self.observability.record_stage_timing(
                document_id,
                "AI Synthesis",
                "NeuroscienceSynthesizer",
                synthesis_start,
                synthesis_end,
                "success",
                {"duration_seconds": synthesis_duration, "quality": synthesis_result.quality_score},
            )
            self.observability.check_latency_anomaly("ai_synthesis", synthesis_duration)
            self.observability.check_quality_anomaly(
                document_id,
                synthesis_result.quality_score,
                getattr(synthesis_result, "validation_issues", None),
            )

            # Stage 4: Create Knowledge Document
            creation_start = datetime.utcnow()
            creation_perf = time.perf_counter()
            logger.info("Stage 4: Creating Knowledge Document", extra={"processing_stage": "document_build"})
            with self.observability.start_span(
                "pipeline.document_assembly",
                {"document.id": document_id},
            ):
                document = self._create_knowledge_document(
                    pdf_path, extraction_result, entities, synthesis_result
                )
            creation_end = datetime.utcnow()
            creation_duration = time.perf_counter() - creation_perf
            self.observability.record_stage_timing(
                document_id,
                "Document Assembly",
                "DocumentProcessor",
                creation_start,
                creation_end,
                "success",
                {"duration_seconds": creation_duration},
            )

            # Stage 5: Quality Validation
            validation_start = datetime.utcnow()
            validation_perf = time.perf_counter()
            logger.info("Stage 5: Quality Validation", extra={"processing_stage": "quality_validation"})
            is_valid = self._validate_document_quality(document)
            validation_end = datetime.utcnow()
            validation_duration = time.perf_counter() - validation_perf
            self.observability.record_stage_timing(
                document_id,
                "Quality Validation",
                "QualityValidator",
                validation_start,
                validation_end,
                "success" if is_valid else "warning",
                {"duration_seconds": validation_duration, "quality_score": document.get_quality_score()},
            )
            if not is_valid:
                logger.warning("Document quality validation failed", extra={"document_id": document_id})

            # Aggregate metrics and RAG exports
            prompt_text = getattr(synthesis_result, "prompt_text", "")
            estimated_tokens = int(len(prompt_text) / 4) if prompt_text else 0
            self.observability.check_cost_anomaly(document_id, estimated_tokens)

            sections = extraction_result.get('sections', []) if isinstance(extraction_result, dict) else []
            context_chunks = []
            for section in sections[:5]:
                if isinstance(section, dict):
                    chunk = section.get('content') or section.get('text') or ''
                else:
                    chunk = str(section)
                if chunk:
                    context_chunks.append(chunk[:500])
            if not context_chunks and isinstance(extraction_result, dict):
                full_text = extraction_result.get('full_text')
                if full_text:
                    context_chunks.append(full_text[:500])

            citations = []
            for idx, finding in enumerate(synthesis_result.key_findings):
                text = finding.get('finding_text', '') if isinstance(finding, dict) else str(finding)
                if text:
                    citations.append({"id": f"finding-{idx}", "label": text[:120]})

            self.observability.record_rag_metrics(
                document_id,
                prompt_text,
                context_chunks,
                synthesis_result.enlitens_takeaway,
                citations,
                synthesis_result.quality_score,
            )

            concept_nodes = []
            for idx, concept in enumerate(synthesis_result.neuroscientific_concepts):
                label = concept.get('concept_name', '') if isinstance(concept, dict) else str(concept)
                if label:
                    concept_nodes.append({"id": f"concept-{idx}", "label": label, "type": "concept"})

            graph_nodes = (
                [{"id": document_id, "label": document.source_metadata.title or document_id, "type": "document"}]
                + concept_nodes
                + [{"id": citation["id"], "label": citation["label"], "type": "finding"} for citation in citations]
            )
            graph_edges = []
            for concept in concept_nodes:
                graph_edges.append({"source": concept["id"], "target": document_id})
            for citation in citations:
                graph_edges.append({"source": citation["id"], "target": document_id})
            self.observability.record_citation_graph(document_id, graph_nodes, graph_edges)

            summary_metrics = {
                "stage": "document",
                "quality_score": document.get_quality_score(),
                "synthesis_quality": synthesis_result.quality_score,
                "extraction_quality": extraction_score,
                "entity_total": sum(len(v) for v in entities.values()) if isinstance(entities, dict) else 0,
            }
            self.observability.record_quality_metrics(document_id, summary_metrics)

            logger.info(f"Document processed successfully: {document.document_id}")
            return True, document, ""

        except Exception as e:
            error_msg = f"Document processing failed: {str(e)}"
            self.observability.capture_exception(
                e,
                {
                    "pdf_path": pdf_path,
                    "document_id": document_id,
                },
            )
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False, None, error_msg
    
    def _create_knowledge_document(self, 
                                 pdf_path: str, 
                                 extraction_result: Dict, 
                                 entities: Dict, 
                                 synthesis_result) -> EnlitensKnowledgeDocument:
        """Create a complete knowledge document"""
        
        # Extract metadata
        metadata = extraction_result.get('metadata', {})
        source_metadata = SourceMetadata(
            title=extraction_result.get('title') or metadata.get('title', 'Unknown Title'),
            authors=[],  # Would need to extract from PDF
            source_filename=os.path.basename(pdf_path),
            journal=metadata.get('journal', ''),
            doi=extraction_result.get('doi') or metadata.get('doi', ''),
            keywords=metadata.get('keywords', [])
        )

        # Create archival content
        archival_content = ArchivalContent(
            full_document_text_markdown=extraction_result.get('full_text', ''),
            abstract_markdown=extraction_result.get('abstract', ''),
            sections=[],  # Would need to convert from extraction_result
            tables=[],    # Would need to convert from extraction_result
            figures=[],   # Would need to convert from extraction_result
            references=extraction_result.get('references', []),
            equations=extraction_result.get('equations', []),
            extraction_quality_score=extraction_result.get('quality_score', 0.0)
        )
        
        # Create entity extraction
        entity_extraction = EntityExtraction(
            biomedical=entities.get('biomedical', []),
            scientific=entities.get('scientific', []),
            neuroscience_specific=entities.get('neuroscience_specific', []),
            brain_regions=entities.get('brain_regions', []),
            neurotransmitters=entities.get('neurotransmitters', []),
            neural_circuits=entities.get('neural_circuits', []),
            neural_processes=entities.get('neural_processes', []),
            extraction_quality_score=0.9  # Would calculate from actual results
        )
        
        # Create AI synthesis
        ai_synthesis = AISynthesis(
            enlitens_takeaway=synthesis_result.enlitens_takeaway,
            eli5_summary=synthesis_result.eli5_summary,
            key_findings=synthesis_result.key_findings,
            neuroscientific_concepts=synthesis_result.neuroscientific_concepts,
            clinical_applications=synthesis_result.clinical_applications,
            therapeutic_targets=synthesis_result.therapeutic_targets,
            client_presentations=synthesis_result.client_presentations,
            intervention_suggestions=synthesis_result.intervention_suggestions,
            contraindications=synthesis_result.contraindications,
            evidence_strength=synthesis_result.evidence_strength,
            synthesis_quality_score=synthesis_result.quality_score,
            powerful_quotes=synthesis_result.powerful_quotes,
            source_citations=synthesis_result.source_citations
        )
        
        # Create processing metadata
        processing_metadata = ProcessingMetadata(
            extraction_method="pymupdf4llm_marker",
            synthesis_method="two_stage_qwen3_32b",
            extraction_method="hybrid_docling_marker",
            synthesis_method="qwen2.5_32b_vllm",
            quality_scores={
                'extraction': extraction_result.get('quality_score', 0.0),
                'entity_extraction': 0.9,
                'synthesis': synthesis_result.quality_score
            }
        )
        
        # Create complete document
        document = EnlitensKnowledgeDocument(
            source_metadata=source_metadata,
            archival_content=archival_content,
            entity_extraction=entity_extraction,
            ai_synthesis=ai_synthesis,
            processing_metadata=processing_metadata
        )
        
        return document
    
    def _validate_document_quality(self, document: EnlitensKnowledgeDocument) -> bool:
        """Validate overall document quality"""
        quality_score = document.get_quality_score()

        if quality_score < 0.8:
            logger.warning(f"Document quality below threshold: {quality_score:.2f}")
            return False
        
        # Check for critical components
        if not document.source_metadata.title or document.source_metadata.title == "Unknown Title":
            logger.warning("Document missing title")
            return False
        
        if not document.archival_content.full_document_text_markdown:
            logger.warning("Document missing content")
            return False
        
        if not document.ai_synthesis or not document.ai_synthesis.enlitens_takeaway:
            logger.warning("Document missing synthesis")
            return False

        layered_result = self.layered_validator.validate_document(document)
        logger.info("QUALITY_METRICS %s", json.dumps(layered_result.to_quality_payload()))
        if not layered_result.passed:
            for layer in layered_result.layers:
                for issue in layer.issues:
                    logger.warning("Layered validation issue (%s): %s", layer.name, issue)
            return False

        return True

    def _extract_entities(self, extraction_result: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        if not self.entity_extractor.initialize():
            raise RuntimeError("Entity extractor initialization failed")
        full_text = extraction_result.get('full_text') or extraction_result.get('full_document_text', '')
        return self.entity_extractor.extract_entities(full_text)

    def _fallback_extraction(self, source_filename: str) -> Dict[str, Any]:
        logger.info("Producing fallback extraction for %s", source_filename)
        return {
            'title': f"Fallback extraction for {source_filename}",
            'abstract': 'Extraction failed; this is a placeholder abstract ensuring pipeline continuity.',
            'full_text': 'Extraction failed; placeholder content inserted to allow downstream validation.',
            'sections': [],
            'tables': [],
            'references': [],
        }

    def _fallback_synthesis(self, extraction_result: Dict[str, Any]):
        logger.info("Producing fallback synthesis due to generation failure")
        placeholder_claim = "Fallback synthesis generated due to upstream failure"
        return SimpleNamespace(
            enlitens_takeaway="We were unable to generate a full synthesis. Please review the source document manually.",
            eli5_summary="An issue occurred during synthesis; content needs human review.",
            key_findings=[SimpleNamespace(finding_text=placeholder_claim, evidence_strength="preliminary", relevance_to_enlitens="Manual follow-up required")],
            neuroscientific_concepts=[],
            clinical_applications=[],
            therapeutic_targets=[],
            client_presentations=[],
            intervention_suggestions=[],
            contraindications=[],
            evidence_strength="preliminary",
            quality_score=0.0,
        )
    
    def process_batch(self, pdf_paths: List[str], max_retries: int = 3) -> Dict[str, Any]:
        """
        Process a batch of PDF documents
        
        Args:
            pdf_paths: List of PDF file paths
            max_retries: Maximum number of retries for failed documents
            
        Returns:
            Processing results summary
        """
        logger.info(f"Processing batch of {len(pdf_paths)} documents")
        
        results = {
            'total': len(pdf_paths),
            'successful': 0,
            'failed': 0,
            'retried': 0,
            'documents': [],
            'errors': []
        }
        
        for pdf_path in pdf_paths:
            success, document, error_msg = self.process_document(pdf_path)
            
            if success:
                results['successful'] += 1
                results['documents'].append(document)
                self.knowledge_base.add_document(document)
            else:
                results['failed'] += 1
                results['errors'].append({
                    'pdf_path': pdf_path,
                    'error': error_msg
                })
        
        # Save knowledge base
        self._save_knowledge_base()
        
        # Generate summary
        results['knowledge_base_stats'] = self.knowledge_base.get_statistics()
        
        logger.info(f"Batch processing completed: {results['successful']}/{results['total']} successful")
        return results
    
    def process_all_pdfs(self, max_retries: int = 3) -> Dict[str, Any]:
        """
        Process all PDFs in the input directory
        
        Args:
            max_retries: Maximum number of retries for failed documents
            
        Returns:
            Processing results summary
        """
        # Find all PDF files
        pdf_files = list(self.pdf_input_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_input_dir}")
            return {'error': 'No PDF files found'}
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process in batches to manage memory
        batch_size = 5
        all_results = {
            'total': len(pdf_files),
            'successful': 0,
            'failed': 0,
            'batches_processed': 0,
            'errors': []
        }
        
        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i + batch_size]
            batch_results = self.process_batch([str(f) for f in batch], max_retries)
            
            all_results['successful'] += batch_results['successful']
            all_results['failed'] += batch_results['failed']
            all_results['batches_processed'] += 1
            all_results['errors'].extend(batch_results['errors'])
            
            logger.info(f"Processed batch {all_results['batches_processed']}: "
                       f"{batch_results['successful']}/{len(batch)} successful")
        
        # Final knowledge base stats
        all_results['knowledge_base_stats'] = self.knowledge_base.get_statistics()
        
        logger.info(f"All PDFs processed: {all_results['successful']}/{all_results['total']} successful")
        return all_results
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        return {
            'knowledge_base_path': str(self.knowledge_base_path),
            'knowledge_base_exists': self.knowledge_base_path.exists(),
            'total_documents': len(self.knowledge_base.documents),
            'vllm_available': self.ollama_client.is_available(),
            'model_manager_status': {
                'loaded_models': self.model_manager.list_loaded_models(),
                'memory_usage': self.model_manager.get_memory_usage()
            }
        }
    
    def get_document_summary(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a specific document"""
        document = self.knowledge_base.get_document(document_id)
        if document:
            return document.get_summary()
        return None
    
    def export_knowledge_base(self, output_path: Optional[str] = None) -> str:
        """Export knowledge base to JSON file"""
        if output_path is None:
            output_path = self.output_dir / f"enlitens-knowledge-export-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base.dict(), f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Knowledge base exported to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to export knowledge base: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Test the document processor
    processor = DocumentProcessor()
    
    # Check status
    status = processor.get_processing_status()
    print("Processing status:", status)
    
    # Test with a single PDF (if available)
    pdf_files = list(Path("./enlitens_corpus/input_pdfs").glob("*.pdf"))
    if pdf_files:
        test_pdf = str(pdf_files[0])
        print(f"Testing with: {test_pdf}")
        
        success, document, error = processor.process_document(test_pdf)
        if success:
            print("Document processed successfully!")
            print(f"Document ID: {document.document_id}")
            print(f"Quality Score: {document.get_quality_score():.2f}")
            print(f"Summary: {document.get_summary()}")
        else:
            print(f"Processing failed: {error}")
    else:
        print("No PDF files found for testing")
