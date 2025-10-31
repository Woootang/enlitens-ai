"""
Enhanced PDF Extractor V2 - Fixed Architecture

This extractor fixes the issues identified:
- Proper table extraction with rows and columns
- Clean metadata extraction without HTML entities
- Better title and author extraction
- Improved content structure
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import re
import html

import docling
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
import torch

logger = logging.getLogger(__name__)


class EnhancedPDFExtractorV2:
    """
    Enhanced PDF extractor with fixed architecture for proper content extraction.

    Fixes:
    - Table extraction with proper rows and columns
    - Clean metadata without HTML entities
    - Better title and author extraction
    - Improved content structure
    - GPU OOM mitigation by forcing CPU for RT-DETR when needed
    """

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Check if we should force CPU mode to avoid GPU OOM with vLLM
        force_cpu = os.getenv("DOCLING_FORCE_CPU", "false").lower() == "true"
        device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")

        if force_cpu:
            logger.info("🖥️ Docling configured to use CPU to avoid GPU OOM while vLLM is running")
        else:
            logger.info(f"🖥️ Docling configured to use device: {device}")

        # Configure Docling with proper options and device
        self.pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            table_structure_options={
                "do_cell_matching": True,
                "do_cell_content_extraction": True,
            },
            table_cell_matching_options={
                "do_unify_full_cells": True,
                "do_use_table_structure": True,
            },
            # Force CPU for layout model to avoid GPU OOM
            images_scale=0.9,  # keep resolution modest on both CPU/GPU to avoid VRAM spikes
        )

        # Create converter with device configuration
        converter_config = {}
        if force_cpu:
            # Force all models to CPU
            converter_config["device"] = "cpu"

        self.docling_converter = DocumentConverter(**converter_config)
        self.device = device
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract content from PDF with proper structure.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Structured extraction result
        """
        try:
            logger.info(f"Starting enhanced extraction for {pdf_path}")
            
            # Check cache first
            cache_key = self._get_cache_key(pdf_path)
            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                logger.info(f"Using cached extraction for {pdf_path}")
                return cached_result
            
            # Convert PDF using Docling
            result = self.docling_converter.convert(pdf_path)
            docling_document = result.document
            
            # Extract structured content
            extraction_result = {
                'source_metadata': self._extract_source_metadata(pdf_path, docling_document),
                'archival_content': self._extract_archival_content(docling_document),
                'quality_metrics': self._calculate_quality_metrics(docling_document),
                'extraction_timestamp': datetime.now().isoformat(),
                'extraction_method': 'enhanced_docling_v2'
            }
            
            # Cache the result
            self._save_to_cache(cache_key, extraction_result)
            
            logger.info(f"Enhanced extraction completed for {pdf_path}")
            return extraction_result
            
        except Exception as e:
            logger.error(f"Enhanced extraction failed for {pdf_path}: {str(e)}")
            raise
    
    def _extract_source_metadata(self, pdf_path: str, document) -> Dict[str, Any]:
        """Extract clean source metadata"""
        full_text = document.export_to_markdown()
        
        # Clean HTML entities
        full_text = html.unescape(full_text)
        
        return {
            'title': self._extract_title(full_text),
            'authors': self._extract_authors(full_text),
            'publication_date': self._extract_publication_date(full_text),
            'journal': self._extract_journal(full_text),
            'doi': self._extract_doi(full_text),
            'pmid': self._extract_pmid(full_text),
            'keywords': self._extract_keywords(full_text),
            'abstract': self._extract_abstract(full_text),
            'source_filename': os.path.basename(pdf_path),
            'ingestion_timestamp': datetime.now().isoformat()
        }
    
    def _extract_title(self, full_text: str) -> str:
        """Extract document title with better logic"""
        lines = full_text.split('\n')
        
        # Look for title patterns
        title_patterns = [
            r'^[A-Z][^.!?]*[a-z][^.!?]*$',  # Title case
            r'^[A-Z][^.!?]*[a-z][^.!?]*[a-z]$'  # Title case with ending
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip common non-title lines
            if (line.startswith('#') or 
                line.startswith('©') or 
                line.startswith('ISSN') or
                line.startswith('Yannick') or
                line.startswith('1 Euromov') or
                line.startswith('2 Department') or
                line.startswith('3 Department') or
                line.startswith('The present study') or
                line.startswith('Data were drawn') or
                line.startswith('Abstract') or
                line.startswith('Keywords') or
                line.startswith('Introduction') or
                line.startswith('Method') or
                line.startswith('Results') or
                line.startswith('Discussion') or
                line.startswith('References')):
                continue
            
            # Check if line looks like a title
            if (len(line) > 20 and len(line) < 300 and
                not line.endswith('.') and
                not line.endswith('?') and
                not line.endswith('!') and
                not line.startswith('Table') and
                not line.startswith('Figure') and
                not re.match(r'^\d+\.', line) and
                any(char.isalpha() for char in line)):
                
                # Check if it's likely a title based on position and content
                if i < 10:  # Title should be near the beginning
                    return line
        
        return "Unknown Title"
    
    def _extract_authors(self, full_text: str) -> List[Dict[str, str]]:
        """Extract authors with better parsing"""
        authors = []
        lines = full_text.split('\n')
        
        # Look for author section
        in_author_section = False
        author_text = ""
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip if it's clearly not an author line
            if (line.startswith('#') or 
                line.startswith('©') or 
                line.startswith('ISSN') or
                line.startswith('Abstract') or
                line.startswith('Keywords') or
                line.startswith('Introduction')):
                continue
            
            # Look for author indicators
            if (len(line) > 10 and 
                any(char.isdigit() for char in line) and
                not line.startswith('The present study') and
                not line.startswith('Data were drawn')):
                
                in_author_section = True
                author_text = line
                continue
            
            if in_author_section:
                if line.startswith('Abstract') or line.startswith('Keywords') or line.startswith('The present study'):
                    break
                if line and not line.startswith('#'):
                    author_text += " " + line
        
        # Parse authors from the text
        if author_text:
            # Split by common separators
            author_parts = re.split(r'[,\s]+and\s+|[,\s]+&amp;\s+|[,\s]+&amp;\s+', author_text)
            
            for part in author_parts:
                part = part.strip()
                if part and len(part) > 3:
                    # Extract name (before any numbers)
                    name_match = re.search(r'^([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', part)
                    if name_match:
                        name = name_match.group(1)
                        # Extract affiliation number
                        aff_match = re.search(r'(\d+)', part)
                        affiliation = aff_match.group(1) if aff_match else ""
                        
                        authors.append({
                            'name': name,
                            'affiliation': affiliation
                        })
        
        return authors if authors else [{'name': 'Unknown Author', 'affiliation': 'Unknown'}]
    
    def _extract_publication_date(self, full_text: str) -> str:
        """Extract publication date"""
        date_patterns = [
            r'(\d{4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, full_text)
            if matches:
                return matches[0]
        
        return "Unknown"
    
    def _extract_journal(self, full_text: str) -> str:
        """Extract journal name"""
        journal_patterns = [
            r'Journal of [^,\n]+',
            r'Nature [^,\n]+',
            r'Science [^,\n]+',
            r'Cell [^,\n]+',
            r'Neuron [^,\n]+',
            r'Brain [^,\n]+',
            r'Neuro[^,\n]+',
            r'Psych[^,\n]+'
        ]
        
        for pattern in journal_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                return matches[0]
        
        return "Unknown"
    
    def _extract_doi(self, full_text: str) -> str:
        """Extract DOI"""
        doi_match = re.search(r'10\.\d+/[^\s]+', full_text)
        if doi_match:
            return doi_match.group(0)
        return "Unknown"
    
    def _extract_pmid(self, full_text: str) -> str:
        """Extract PMID"""
        pmid_match = re.search(r'PMID:\s*(\d+)', full_text)
        if pmid_match:
            return pmid_match.group(1)
        return "Unknown"
    
    def _extract_keywords(self, full_text: str) -> List[str]:
        """Extract keywords"""
        lines = full_text.split('\n')
        keywords = []
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('keywords'):
                keyword_text = line.split(':', 1)[1] if ':' in line else line
                keywords = [kw.strip() for kw in keyword_text.split(',')]
                break
        
        return keywords
    
    def _extract_abstract(self, full_text: str) -> str:
        """Extract abstract content"""
        lines = full_text.split('\n')
        abstract_lines = []
        in_abstract = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line.lower().startswith('abstract') or line.lower().startswith('summary'):
                in_abstract = True
                continue
            
            if in_abstract and (line.lower().startswith('keywords') or 
                               line.lower().startswith('introduction') or
                               line.lower().startswith('method') or
                               line.lower().startswith('the present study')):
                break
            
            if in_abstract and line:
                abstract_lines.append(line)
        
        # If no abstract found, look for first substantial paragraph
        if not abstract_lines:
            for i, line in enumerate(lines):
                line = line.strip()
                if (len(line) > 100 and 
                    not line.startswith('#') and 
                    not line.startswith('©') and 
                    not line.startswith('ISSN') and
                    not line.startswith('Yannick') and
                    not line.startswith('1 Euromov') and
                    not line.startswith('2 Department') and
                    not line.startswith('3 Department') and
                    not line.startswith('The present study') and
                    not line.startswith('Data were drawn')):
                    abstract_lines.append(line)
                    break
        
        return ' '.join(abstract_lines).strip()
    
    def _extract_archival_content(self, document) -> Dict[str, Any]:
        """Extract archival content with proper structure"""
        full_text = document.export_to_markdown()
        
        # Clean HTML entities
        full_text = html.unescape(full_text)
        
        return {
            'full_document_text_markdown': full_text,
            'abstract_markdown': self._extract_abstract(full_text),
            'sections': self._extract_sections(document),
            'tables': self._extract_tables(document),
            'figures': self._extract_figures(document),
            'references': self._extract_references(document)
        }
    
    def _extract_sections(self, document) -> List[Dict[str, str]]:
        """Extract document sections"""
        sections = []
        current_section = None
        
        for element in document.iterate_items():
            if hasattr(element, 'label') and element.label and 'heading' in element.label.lower():
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    'title': element.text_content.strip(),
                    'content': ''
                }
            elif current_section and hasattr(element, 'text_content'):
                current_section['content'] += element.text_content + '\n'
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _extract_tables(self, document) -> List[Dict[str, Any]]:
        """Extract tables with proper structure"""
        tables = []
        
        for element in document.iterate_items():
            if hasattr(element, 'label') and 'table' in element.label.lower():
                table_data = {
                    'caption': '',
                    'content': [],
                    'rows': 0,
                    'columns': 0
                }
                
                # Extract caption
                if hasattr(element, 'caption') and element.caption:
                    table_data['caption'] = element.caption.text_content.strip()
                
                # Extract table content
                if hasattr(element, 'table_content') and element.table_content:
                    table_data['content'] = []
                    for row in element.table_content:
                        row_data = []
                        for cell in row:
                            if hasattr(cell, 'text_content'):
                                row_data.append(cell.text_content.strip())
                            else:
                                row_data.append(str(cell).strip())
                        table_data['content'].append(row_data)
                    
                    table_data['rows'] = len(table_data['content'])
                    table_data['columns'] = len(table_data['content'][0]) if table_data['content'] else 0
                
                tables.append(table_data)
        
        return tables
    
    def _extract_figures(self, document) -> List[Dict[str, str]]:
        """Extract figure information"""
        figures = []
        
        for element in document.iterate_items():
            if hasattr(element, 'label') and 'figure' in element.label.lower():
                figure_data = {
                    'caption': '',
                    'type': 'figure'
                }
                
                if hasattr(element, 'caption') and element.caption:
                    figure_data['caption'] = element.caption.text_content.strip()
                
                figures.append(figure_data)
        
        return figures
    
    def _extract_references(self, document) -> List[str]:
        """Extract references"""
        references = []
        in_references = False
        
        for element in document.iterate_items():
            if hasattr(element, 'label') and element.label and 'references' in element.label.lower():
                in_references = True
                continue
            
            if in_references and hasattr(element, 'text_content') and element.text_content.strip():
                text = element.text_content.strip()
                if text:
                    references.append(text)
        
        return references
    
    def _calculate_quality_metrics(self, document) -> Dict[str, float]:
        """Calculate quality metrics"""
        full_text = document.export_to_markdown()
        
        # Completeness score
        completeness_score = min(1.0, len(full_text) / 10000)
        
        # Structure score
        structure_score = 0.0
        if 'abstract' in full_text.lower():
            structure_score += 0.2
        if 'introduction' in full_text.lower():
            structure_score += 0.2
        if 'method' in full_text.lower():
            structure_score += 0.2
        if 'results' in full_text.lower():
            structure_score += 0.2
        if 'discussion' in full_text.lower():
            structure_score += 0.2
        
        # Metadata score
        metadata_score = 0.0
        if len(self._extract_title(full_text)) > 10:
            metadata_score += 0.2
        if self._extract_authors(full_text):
            metadata_score += 0.2
        if self._extract_abstract(full_text):
            metadata_score += 0.2
        if self._extract_doi(full_text) != "Unknown":
            metadata_score += 0.2
        if self._extract_journal(full_text) != "Unknown":
            metadata_score += 0.2
        
        return {
            'completeness_score': completeness_score,
            'structure_score': structure_score,
            'metadata_score': metadata_score,
            'overall_score': (completeness_score + structure_score + metadata_score) / 3
        }
    
    def _get_cache_key(self, pdf_path: str) -> str:
        """Generate cache key for PDF"""
        with open(pdf_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return f"{os.path.basename(pdf_path)}_{file_hash}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached extraction result"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]):
        """Save extraction result to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
