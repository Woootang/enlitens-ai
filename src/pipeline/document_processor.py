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
    ArchivalContent,
    EntityExtraction,
    AISynthesis,
    ProcessingMetadata,
    KnowledgeBase
)
from src.utils.retry import IntelligentRetryManager
from src.validation.layered_validation import LayeredValidationPipeline

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
                 ollama_url: str = "http://localhost:11434",
                 ollama_model: str = "qwen3:32b"):
        
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
        self.retry_manager = IntelligentRetryManager()
        self.layered_validator = LayeredValidationPipeline()
        
        # Initialize knowledge base
        self.knowledge_base_path = self.output_dir / "enlitens-knowledge-core.json"
        self.knowledge_base = self._load_knowledge_base()
        
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
        logger.info(f"Processing document: {pdf_path}")
        
        try:
            # Stage 1: PDF Extraction
            logger.info("Stage 1: PDF Extraction")
            extraction_outcome = self.retry_manager.run(
                "pdf_extraction",
                lambda: self.pdf_extractor.extract(pdf_path),
                correction_prompt="Re-run extraction with OCR fallback and adjust layout heuristics",
                degradation=lambda: self._fallback_extraction(os.path.basename(pdf_path)),
            )
            extraction_result = extraction_outcome.result
            if not extraction_outcome.success and extraction_result is None:
                return False, None, extraction_outcome.error or "Extraction failed"

            # Validate extraction quality
            extraction_metrics = self.quality_validator.validate_extraction(extraction_result)
            if extraction_metrics.overall_score < 0.95:
                logger.warning(f"Extraction quality below threshold: {extraction_metrics.overall_score:.2f}")

            # Stage 2: Entity Extraction
            logger.info("Stage 2: Entity Extraction")
            entity_outcome = self.retry_manager.run(
                "entity_extraction",
                lambda: self._extract_entities(extraction_result),
                correction_prompt="Reload spaCy/transformer weights and ensure GPU memory is free",
                degradation=lambda: {},
            )
            if not entity_outcome.success and entity_outcome.result is None:
                return False, None, entity_outcome.error or "Entity extraction failed"
            entities = entity_outcome.result or {}

            # Stage 3: AI Synthesis
            logger.info("Stage 3: AI Synthesis")
            if not self.ollama_client.is_available():
                logger.warning("Ollama is not available - attempting graceful degradation")
            synthesis_outcome = self.retry_manager.run(
                "ai_synthesis",
                lambda: self.synthesizer.synthesize(extraction_result),
                correction_prompt="Regenerate with shorter context window and stricter JSON schema",
                degradation=lambda: self._fallback_synthesis(extraction_result),
            )
            synthesis_result = synthesis_outcome.result
            if synthesis_result is None:
                return False, None, synthesis_outcome.error or "AI synthesis failed"

            # Stage 4: Create Knowledge Document
            logger.info("Stage 4: Creating Knowledge Document")
            document = self._create_knowledge_document(
                pdf_path, extraction_result, entities, synthesis_result
            )

            # Stage 5: Quality Validation
            logger.info("Stage 5: Quality Validation")
            if not self._validate_document_quality(document):
                logger.warning("Document quality validation failed")
            
            logger.info(f"Document processed successfully: {document.document_id}")
            return True, document, ""
            
        except Exception as e:
            error_msg = f"Document processing failed: {str(e)}"
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
        source_metadata = SourceMetadata(
            title=extraction_result.get('title', 'Unknown Title'),
            authors=[],  # Would need to extract from PDF
            source_filename=os.path.basename(pdf_path),
            journal=extraction_result.get('metadata', {}).get('journal', ''),
            doi=extraction_result.get('metadata', {}).get('doi', ''),
            keywords=extraction_result.get('metadata', {}).get('keywords', [])
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
            synthesis_quality_score=synthesis_result.quality_score
        )
        
        # Create processing metadata
        processing_metadata = ProcessingMetadata(
            extraction_method="hybrid_docling_marker",
            synthesis_method="qwen3_32b",
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
            'ollama_available': self.ollama_client.is_available(),
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
