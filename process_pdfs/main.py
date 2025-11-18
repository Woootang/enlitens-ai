#!/usr/bin/env python3
"""
Main PDF Processing Orchestrator
Coordinates ingestion, extraction, translation, and storage
"""
import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import time

sys.path.insert(0, '/home/antons-gs/enlitens-ai')

from process_pdfs.ingestion import process_pdf
from process_pdfs.extraction import extract_scientific_content
from process_pdfs.translation import translate_to_clinical
from src.utils.llm_client import LLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/antons-gs/enlitens-ai/logs/processing.log'),
        logging.StreamHandler()
    ],
    force=True,
)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Main processor for research PDFs"""
    
    def __init__(self, knowledge_base_path: str = "data/knowledge_base/main_kb.jsonl"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM client
        logger.info("Initializing LLM client...")
        self.llm_client = LLMClient()
        logger.info("✅ LLM client ready")
    
    def process_single_pdf(self, pdf_path: Path, paper_id: Optional[str] = None) -> Dict:
        """
        Process a single PDF through the complete pipeline
        
        Args:
            pdf_path: Path to PDF file
            paper_id: Optional unique identifier
            
        Returns:
            Complete knowledge base entry
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)
        
        if not paper_id:
            paper_id = pdf_path.stem
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {pdf_path.name}")
        logger.info(f"Paper ID: {paper_id}")
        logger.info(f"{'='*80}\n")
        
        try:
            # Step 1: Ingestion
            logger.info("[1/4] Ingesting PDF...")
            ingestion_result = process_pdf(pdf_path, paper_id)
            logger.info(f"✅ Ingestion complete: {len(ingestion_result['verbatim_text'])} chars")
            
            # Step 2: Scientific Extraction
            logger.info("[2/4] Extracting scientific content...")
            extraction_result = extract_scientific_content(
                ingestion_result['verbatim_text'],
                self.llm_client
            )
            logger.info("✅ Extraction complete")
            
            # Step 3: Clinical Translation
            logger.info("[3/4] Translating to clinical guidance...")
            translation_result = translate_to_clinical(
                extraction_result,
                self.llm_client
            )
            logger.info("✅ Translation complete")
            
            # Step 4: Assemble Knowledge Base Entry
            logger.info("[4/4] Assembling knowledge base entry...")
            kb_entry = self._assemble_kb_entry(
                paper_id=paper_id,
                ingestion=ingestion_result,
                extraction=extraction_result,
                translation=translation_result,
                processing_duration=time.time() - start_time
            )
            
            # Save to knowledge base
            self._save_to_kb(kb_entry)
            
            logger.info(f"\n✅ Processing complete: {paper_id}")
            logger.info(f"Duration: {time.time() - start_time:.1f} seconds")
            logger.info(f"{'='*80}\n")
            
            return kb_entry
            
        except Exception as e:
            logger.error(f"❌ Processing failed for {paper_id}: {e}", exc_info=True)
            raise
    
    def _assemble_kb_entry(
        self,
        paper_id: str,
        ingestion: Dict,
        extraction: Dict,
        translation: Dict,
        processing_duration: float
    ) -> Dict:
        """Assemble complete knowledge base entry"""
        return {
            "id": paper_id,
            "type": "research_paper",
            "metadata": {
                "title": ingestion["metadata"].get("title", ""),
                "authors": ingestion["metadata"].get("authors", []),
                "page_count": ingestion["metadata"].get("page_count", 0),
                "source_pdf": ingestion["source_pdf"],
                "source_sha256": ingestion["source_sha256"]
            },
            "full_text": ingestion["verbatim_text"],
            "extraction_scientific": extraction,
            "clinical_translation": translation,
            "tables": ingestion.get("tables", []),
            "figures": ingestion.get("figures", []),
            "processing_timestamp": datetime.utcnow().isoformat() + "Z",
            "processing_duration_seconds": round(processing_duration, 2),
            "extraction_method": ingestion.get("extraction_method", "unknown")
        }
    
    def _save_to_kb(self, entry: Dict):
        """Append entry to knowledge base JSONL file"""
        with open(self.knowledge_base_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False))
            f.write('\n')
        logger.info(f"💾 Saved to knowledge base: {self.knowledge_base_path}")
    
    def process_batch(self, pdf_directory: Path, resume: bool = True):
        """
        Process all PDFs in a directory
        
        Args:
            pdf_directory: Directory containing PDFs
            resume: Skip already-processed papers
        """
        pdf_directory = Path(pdf_directory)
        pdf_files = sorted(pdf_directory.glob("*.pdf"))
        
        logger.info(f"\n{'='*80}")
        logger.info(f"BATCH PROCESSING")
        logger.info(f"Directory: {pdf_directory}")
        logger.info(f"Total PDFs: {len(pdf_files)}")
        logger.info(f"{'='*80}\n")
        
        # Load already-processed IDs if resuming
        processed_ids = set()
        if resume and self.knowledge_base_path.exists():
            with open(self.knowledge_base_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    processed_ids.add(entry['id'])
            logger.info(f"Resume mode: {len(processed_ids)} papers already processed")
        
        # Process each PDF
        success_count = 0
        skip_count = 0
        fail_count = 0
        
        for i, pdf_path in enumerate(pdf_files, 1):
            paper_id = pdf_path.stem
            
            if resume and paper_id in processed_ids:
                logger.info(f"[{i}/{len(pdf_files)}] ⏭️  Skipping (already processed): {paper_id}")
                skip_count += 1
                continue
            
            logger.info(f"[{i}/{len(pdf_files)}] Processing: {paper_id}")
            
            try:
                self.process_single_pdf(pdf_path, paper_id)
                success_count += 1
            except Exception as e:
                logger.error(f"❌ Failed: {paper_id} - {e}")
                fail_count += 1
        
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info(f"BATCH COMPLETE")
        logger.info(f"Total: {len(pdf_files)}")
        logger.info(f"Success: {success_count}")
        logger.info(f"Skipped: {skip_count}")
        logger.info(f"Failed: {fail_count}")
        logger.info(f"{'='*80}\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process research PDFs")
    parser.add_argument("input", help="PDF file or directory")
    parser.add_argument("--no-resume", action="store_true", help="Don't skip processed files")
    parser.add_argument("--kb", default="data/knowledge_base/main_kb.jsonl", help="Knowledge base path")
    
    args = parser.parse_args()
    
    processor = PDFProcessor(knowledge_base_path=args.kb)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        processor.process_single_pdf(input_path)
    elif input_path.is_dir():
        processor.process_batch(input_path, resume=not args.no_resume)
    else:
        logger.error(f"Invalid input: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()

