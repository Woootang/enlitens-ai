#!/usr/bin/env python3
"""
PDF Ingestion Module
Extracts complete text, tables, and figures from research PDFs
"""
import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

# Configure Docling to prefer CPU unless explicitly overridden.
cpu_threads = str(os.cpu_count() or 8)
os.environ.setdefault("DOCLING_DEVICE", "cpu")
os.environ.setdefault("DOCLING_NUM_THREADS", cpu_threads)
os.environ.setdefault("DOCLING_ACCELERATOR_OPTIONS__DEVICE", "cpu")
os.environ.setdefault("DOCLING_ACCELERATOR_OPTIONS__NUM_THREADS", cpu_threads)

# Primary extraction tool (Docling)
try:
    from docling.document_converter import DocumentConverter

    DOCLING_AVAILABLE = True
except Exception as exc:
    DOCLING_AVAILABLE = False
    logging.warning("Docling not available: %s", exc)

# Backup extraction tools
import fitz  # PyMuPDF
try:
    import camelot

    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logging.warning("Camelot not available")

try:
    import pytesseract
    from pdf2image import convert_from_path

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("OCR tools not available")

logger = logging.getLogger(__name__)


class PDFIngestionError(Exception):
    """Custom exception for PDF ingestion failures"""
    pass


def extract_pdf_docling(pdf_path: Path) -> Dict:
    """Extract PDF using Docling (IBM) - Primary method"""
    if not DOCLING_AVAILABLE:
        raise PDFIngestionError("Docling not installed")
    
    try:
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        
        # Extract full text
        verbatim_text = result.document.export_to_markdown()
        
        # Extract tables
        tables = []
        if hasattr(result.document, 'tables'):
            for table in result.document.tables:
                tables.append({
                    "caption": getattr(table, 'caption', '') or "",
                    "data": table.to_dataframe().values.tolist() if hasattr(table, 'to_dataframe') else [],
                    "page": getattr(table, 'page_number', None)
                })
        
        # Extract figures
        figures = []
        if hasattr(result.document, 'figures'):
            for figure in result.document.figures:
                figures.append({
                    "caption": getattr(figure, 'caption', '') or "",
                    "page": getattr(figure, 'page_number', None),
                    "bbox": getattr(figure, 'bbox', None)
                })
        
        # Extract metadata
        metadata = {
            "title": getattr(result.document, 'title', '') or "",
            "authors": getattr(result.document, 'authors', []) or [],
            "page_count": getattr(result.document, 'page_count', 0) if hasattr(result.document, 'page_count') else 0
        }
        
        return {
            "verbatim_text": verbatim_text,
            "tables": tables,
            "figures": figures,
            "metadata": metadata,
            "extraction_method": "docling"
        }
        
    except Exception as e:
        logger.error(f"Docling extraction failed: {e}")
        raise PDFIngestionError(f"Docling failed: {e}")


def extract_pdf_pymupdf(pdf_path: Path) -> Dict:
    """Extract PDF using PyMuPDF - Backup method"""
    try:
        doc = fitz.open(str(pdf_path))
        
        # Extract full text with page markers
        verbatim_text = ""
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text")
            verbatim_text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
        
        # Extract images
        figures = []
        for page_num, page in enumerate(doc, start=1):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    if base_image and base_image["width"] > 100 and base_image["height"] > 100:
                        figures.append({
                            "page": page_num,
                            "index": img_index,
                            "width": base_image["width"],
                            "height": base_image["height"],
                            "ext": base_image["ext"]
                        })
                except:
                    continue
        
        # Extract metadata
        metadata = {
            "title": doc.metadata.get("title", ""),
            "authors": [doc.metadata.get("author", "")] if doc.metadata.get("author") else [],
            "page_count": len(doc)
        }
        
        doc.close()
        
        return {
            "verbatim_text": verbatim_text,
            "tables": [],
            "figures": figures,
            "metadata": metadata,
            "extraction_method": "pymupdf"
        }
        
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {e}")
        raise PDFIngestionError(f"PyMuPDF failed: {e}")


def extract_tables_camelot(pdf_path: Path) -> List[Dict]:
    """Extract tables using Camelot"""
    if not CAMELOT_AVAILABLE:
        return []
    
    tables = []
    try:
        camelot_tables = camelot.read_pdf(
            str(pdf_path),
            flavor='lattice',
            pages='all'
        )
        
        for idx, table in enumerate(camelot_tables):
            tables.append({
                "caption": f"Table {idx + 1}",
                "data": table.df.values.tolist(),
                "page": table.page,
                "accuracy": table.accuracy
            })
            
    except Exception as e:
        logger.warning(f"Camelot table extraction failed: {e}")
    
    return tables


def perform_ocr_if_needed(pdf_path: Path, text_length: int) -> Optional[str]:
    """Perform OCR if text extraction yielded very little content"""
    if not OCR_AVAILABLE or text_length >= 1000:
        return None
    
    try:
        logger.info(f"Low text yield ({text_length} chars), attempting OCR...")
        images = convert_from_path(str(pdf_path))
        ocr_text = ""
        for i, image in enumerate(images):
            page_text = pytesseract.image_to_string(image)
            ocr_text += f"\n\n--- Page {i+1} (OCR) ---\n\n{page_text}"
        return ocr_text
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return None


def process_pdf(pdf_path: Path, paper_id: Optional[str] = None) -> Dict:
    """
    Main ingestion function - tries Docling first, falls back to PyMuPDF
    
    Args:
        pdf_path: Path to PDF file
        paper_id: Optional unique identifier
        
    Returns:
        Complete extraction dictionary
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise PDFIngestionError(f"PDF not found: {pdf_path}")
    
    if not paper_id:
        paper_id = pdf_path.stem
    
    logger.info(f"Processing PDF: {pdf_path.name}")

    result: Optional[Dict] = None

    # Try Docling (CPU) first for full-fidelity extraction
    if DOCLING_AVAILABLE:
        try:
            logger.info("Extracting with Docling (CPU)...")
            result = extract_pdf_docling(pdf_path)
            logger.info(
                "✅ Docling extraction successful (%s chars)", len(result["verbatim_text"])
            )
        except Exception as exc:
            logger.warning("Docling extraction failed, falling back to PyMuPDF: %s", exc)

    if result is None:
        logger.info("Extracting with PyMuPDF...")
        result = extract_pdf_pymupdf(pdf_path)
        logger.info(
            "✅ PyMuPDF extraction successful (%s chars)", len(result["verbatim_text"])
        )
    
    # Extract tables separately with Camelot
    if CAMELOT_AVAILABLE:
        logger.info("Extracting tables with Camelot...")
        result["tables"] = extract_tables_camelot(pdf_path)
        logger.info(f"Found {len(result['tables'])} tables")
    
    # Check if OCR is needed
    text_length = len(result["verbatim_text"])
    ocr_text = perform_ocr_if_needed(pdf_path, text_length)
    if ocr_text:
        result["verbatim_text"] = ocr_text
        result["extraction_method"] += "_ocr"
        logger.info(f"✅ OCR completed ({len(ocr_text)} chars)")
    
    # Add file metadata
    result["paper_id"] = paper_id
    result["source_pdf"] = str(pdf_path)
    result["source_sha256"] = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    
    # Validation
    if len(result["verbatim_text"]) < 500:
        logger.warning(f"⚠️ Very short extraction ({len(result['verbatim_text'])} chars)")
    
    logger.info(f"✅ Ingestion complete: {paper_id}")
    return result

