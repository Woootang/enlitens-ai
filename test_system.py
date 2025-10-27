#!/usr/bin/env python3
"""
Test the complete multi-agent system with a single document.
This verifies all components work before processing all 345 files.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from process_multi_agent_corpus import MultiAgentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_single_document():
    """Test processing a single document through the entire system."""
    
    print("=" * 60)
    print("🧪 TESTING ENLITENS MULTI-AGENT SYSTEM")
    print("=" * 60)
    print()
    
    # Initialize processor
    print("📦 Initializing multi-agent processor...")
    processor = MultiAgentProcessor(
        input_dir="enlitens_corpus/input_pdfs",
        output_file="test_single_output.json",
        st_louis_report="st_louis_health_report.pdf"
    )
    
    # Initialize supervisor
    print("🎯 Initializing supervisor agent...")
    success = await processor.supervisor.initialize()
    if not success:
        print("❌ Failed to initialize supervisor!")
        return False
    print("✅ Supervisor initialized successfully")
    print()
    
    # Get first PDF
    pdf_files = list(Path("enlitens_corpus/input_pdfs").glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found!")
        return False
    
    test_pdf = pdf_files[0]
    print(f"📄 Testing with: {test_pdf.name}")
    print()
    
    # Process the document
    print("🚀 Starting multi-agent processing...")
    print("-" * 60)
    
    try:
        result = await processor.process_document(test_pdf)
        
        if result:
            print()
            print("=" * 60)
            print("✅ TEST SUCCESSFUL!")
            print("=" * 60)
            print()
            print("📊 Results:")
            print(f"   Document ID: {result.metadata.document_id}")
            print(f"   Processing Time: {result.metadata.processing_time_seconds:.2f}s")
            print(f"   Quality Score: {result.metadata.quality_score:.2f}")
            print(f"   Confidence Score: {result.metadata.confidence_score:.2f}")
            print()
            print("📝 Content Generated:")
            print(f"   Rebellion Framework: {len(result.rebellion_framework.narrative_deconstruction)} insights")
            print(f"   Marketing Content: {len(result.marketing_content.value_propositions)} propositions")
            print(f"   SEO Content: {len(result.seo_content.primary_keywords)} keywords")
            print(f"   Blog Content: {len(result.blog_content.blog_post_ideas)} ideas")
            print(f"   Clinical Content: {len(result.clinical_content.therapeutic_applications)} applications")
            print()
            print("🎉 System is ready to process all 345 documents!")
            print()
            return True
        else:
            print()
            print("❌ TEST FAILED - No result returned")
            return False
            
    except Exception as e:
        print()
        print(f"❌ TEST FAILED - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        print()
        print("🧹 Cleaning up...")
        await processor.supervisor.cleanup()
        
        # Remove test output
        import os
        for f in ["test_single_output.json", "test_single_output.json.temp"]:
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    success = asyncio.run(test_single_document())
    sys.exit(0 if success else 1)

