#!/usr/bin/env python3
"""
Add New PDFs to Knowledge Base

This script allows you to easily add new PDFs to your existing knowledge base
without reprocessing everything. Just drop new PDFs in the input directory
and run this script.

Usage:
    python add_new_pdfs.py --input-dir enlitens_corpus/input_pdfs --knowledge-base knowledge_base.json
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

from extraction.comprehensive_extractor import ComprehensiveExtractor
from knowledge_base.knowledge_manager import KnowledgeBaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def add_new_pdfs(input_dir: str, knowledge_base_file: str) -> bool:
    """
    Add new PDFs to existing knowledge base
    
    Args:
        input_dir: Directory containing new PDF files
        knowledge_base_file: Existing knowledge base file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Adding new PDFs to knowledge base...")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Knowledge base: {knowledge_base_file}")
        
        # Initialize components
        extractor = ComprehensiveExtractor()
        knowledge_manager = KnowledgeBaseManager(knowledge_base_file)
        
        # Get current stats
        stats = knowledge_manager.get_knowledge_base_stats()
        logger.info(f"Current knowledge base: {stats['total_papers']} papers")
        
        # Get unprocessed files
        unprocessed_files = knowledge_manager.get_unprocessed_files(input_dir)
        
        if not unprocessed_files:
            logger.info("No new files to process")
            return True
        
        logger.info(f"Found {len(unprocessed_files)} new files to process")
        
        # Process each new file
        successful = 0
        failed = 0
        
        for i, pdf_file in enumerate(unprocessed_files, 1):
            logger.info(f"Processing new file {i}/{len(unprocessed_files)}: {pdf_file}")
            
            try:
                # Extract comprehensive content
                extracted_content = extractor.extract(pdf_file)
                
                if extracted_content:
                    # Add to knowledge base
                    if knowledge_manager.add_paper_to_knowledge_base(extracted_content, pdf_file):
                        successful += 1
                        logger.info(f"✓ Successfully added {pdf_file}")
                    else:
                        failed += 1
                        logger.error(f"✗ Failed to add {pdf_file}")
                else:
                    failed += 1
                    logger.error(f"✗ Failed to extract content from {pdf_file}")
                    
            except Exception as e:
                failed += 1
                logger.error(f"✗ Error processing {pdf_file}: {e}")
                continue
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"ADDITION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"New files processed: {len(unprocessed_files)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        
        # Get updated stats
        updated_stats = knowledge_manager.get_knowledge_base_stats()
        logger.info(f"Updated knowledge base: {updated_stats['total_papers']} papers")
        logger.info(f"Added {updated_stats['total_papers'] - stats['total_papers']} new papers")
        
        return successful > 0
        
    except Exception as e:
        logger.error(f"Failed to add new PDFs: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Add new PDFs to knowledge base')
    parser.add_argument('--input-dir', required=True, help='Input directory containing new PDFs')
    parser.add_argument('--knowledge-base', default='knowledge_base.json', help='Knowledge base file')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1
    
    # Validate knowledge base file
    if not os.path.exists(args.knowledge_base):
        logger.error(f"Knowledge base file not found: {args.knowledge_base}")
        logger.info("Run process_knowledge_base.py first to create the knowledge base")
        return 1
    
    # Add new PDFs
    if not add_new_pdfs(args.input_dir, args.knowledge_base):
        logger.error("Failed to add new PDFs")
        return 1
    
    logger.info("Successfully added new PDFs to knowledge base!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
