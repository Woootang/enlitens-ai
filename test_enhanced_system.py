#!/usr/bin/env python3
"""
Test script for the enhanced Enlitens system.

This script tests the enhanced extraction tools, schema enforcement,
and content generation capabilities.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.enhanced_complete_enlitens_agent import EnhancedCompleteEnlitensAgent
from src.extraction.enhanced_extraction_tools import EnhancedExtractionTools
from src.synthesis.ollama_client import OllamaClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_enhanced_system():
    """Test the enhanced system components."""
    try:
        logger.info("🧪 Testing Enhanced Enlitens System")
        
        # Test 1: Ollama connection
        logger.info("Testing Ollama connection...")
        ollama_client = OllamaClient()
        if await ollama_client.check_connection():
            logger.info("✅ Ollama connection successful")
        else:
            logger.error("❌ Ollama connection failed")
            return
        
        # Test 2: Enhanced extraction tools
        logger.info("Testing enhanced extraction tools...")
        extraction_tools = EnhancedExtractionTools()
        
        # Test sentiment analysis
        test_text = "I'm so frustrated with traditional therapy. It feels like they're just labeling me instead of helping me understand my brain."
        sentiment = extraction_tools.analyze_sentiment(test_text)
        logger.info(f"✅ Sentiment analysis: {sentiment}")
        
        # Test keyword extraction
        keywords = extraction_tools.extract_semantic_keywords(test_text, top_n=5)
        logger.info(f"✅ Keyword extraction: {keywords}")
        
        # Test 3: Enhanced agent
        logger.info("Testing enhanced agent...")
        agent = EnhancedCompleteEnlitensAgent()
        
        # Test content extraction with sample text
        sample_text = """
        This study demonstrates that interoceptive awareness training significantly reduces 
        symptoms of generalized anxiety by enhancing the prefrontal cortex's ability to 
        regulate amygdala hyperactivity. The findings suggest that traditional cognitive 
        behavioral therapy may be missing a crucial neurobiological component in anxiety 
        treatment. Participants who received interoceptive training showed 40% greater 
        improvement in anxiety symptoms compared to the control group (p < 0.001).
        """
        
        # Test structured response generation
        from src.models.enlitens_schemas import MarketingContent
        
        prompt = f"""
        # ROLE
        You are a marketing strategist for Enlitens, a revolutionary mental health practice.
        
        # CONTEXT
        Enlitens challenges traditional mental health approaches through neuroscience-based care.
        
        # RESEARCH TEXT
        {sample_text}
        
        # INSTRUCTIONS
        Extract marketing content that aligns with our rebellion against traditional mental health approaches.
        
        # OUTPUT FORMAT
        Provide your analysis as a JSON object with the specified schema.
        """
        
        result = await agent._extract_marketing_content(
            sample_text,
            {}
        )
        
        if result:
            logger.info("✅ Structured response generation successful")
            logger.info(f"Generated {len(result.headlines)} headlines")
            logger.info(f"Generated {len(result.value_propositions)} value propositions")
        
        # Clean up
        await agent.close()
        extraction_tools.cleanup()
        
        logger.info("🎉 Enhanced system test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_enhanced_system())
