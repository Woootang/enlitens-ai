#!/usr/bin/env python3
"""
Test script for the Multi-Agent Enlitens System.

This script tests the multi-agent system with a single document to verify
all components are working correctly before full corpus processing.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.supervisor_agent import SupervisorAgent, ProcessingContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_multi_agent_system():
    """Test the multi-agent system with a sample document."""
    try:
        logger.info("🧪 Testing Multi-Agent System")

        # Initialize supervisor
        supervisor = SupervisorAgent()

        # Check initialization
        success = await supervisor.initialize()
        if not success:
            logger.error("❌ Supervisor initialization failed")
            return False

        logger.info("✅ Supervisor initialized successfully")

        # Get system status
        status = await supervisor.get_system_status()
        logger.info(f"📊 System Status: {status}")

        # Test with sample neuroscience content
        sample_text = """
        Neuroplasticity and Trauma Recovery: A Neuroscience-Based Approach

        Recent advances in neuroscience have revolutionized our understanding of trauma recovery.
        Traditional therapy approaches often focus on symptom management, but emerging research
        shows that trauma fundamentally alters brain structure and function.

        Key findings from recent studies indicate that:
        1. The amygdala becomes hyperactive in PTSD, leading to heightened fear responses
        2. Prefrontal cortex functioning is impaired, affecting executive decision-making
        3. Hippocampal volume reduction affects memory processing and emotional regulation

        Our neuroscience-based approach addresses these underlying neurobiological changes
        through targeted interventions that promote neuroplasticity and brain rewiring.
        This approach has shown significant promise in treating treatment-resistant trauma cases.

        Clinical implications suggest that combining bottom-up sensory approaches with
        top-down cognitive interventions provides the most comprehensive treatment pathway.
        """

        # Create processing context
        context = ProcessingContext(
            document_id="test_neuroscience_001",
            document_text=sample_text,
            client_insights={
                "challenges": ["trauma", "PTSD", "emotional regulation", "treatment resistance"],
                "priorities": ["neuroscience-based therapy", "brain rewiring", "bottom-up approaches"]
            },
            founder_insights={
                "voice_characteristics": ["direct", "neuroscience-focused", "hopeful"],
                "clinical_philosophy": ["brain plasticity", "trauma recovery", "evidence-based"]
            },
            st_louis_context={
                "population": "High trauma rates, diverse community",
                "mental_health_challenges": ["complex trauma", "treatment resistance"]
            }
        )

        logger.info("🚀 Starting test processing...")

        # Process through multi-agent system
        result = await supervisor.process(context.model_dump())

        if result and result.get("supervisor_status") == "completed":
            logger.info("✅ Multi-agent processing completed successfully")

            # Display results summary
            logger.info("📋 Results Summary:")
            logger.info(f"   - Processing time: {result.get('processing_time_seconds', 0):.2f}s")
".2f"            logger.info(f"   - Quality score: {result.get('quality_score', 0):.2f}")
".2f"            logger.info(f"   - Validation passed: {result.get('validation_passed', False)}")

            # Show agent outputs
            agent_outputs = result.get("agent_outputs", {})
            for agent_name, output in agent_outputs.items():
                if isinstance(output, dict) and output:
                    logger.info(f"   - {agent_name}: {len(output)} fields generated")

            return True
        else:
            logger.error("❌ Multi-agent processing failed")
            logger.error(f"Result: {result}")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False
    finally:
        # Cleanup
        try:
            await supervisor.cleanup()
            logger.info("🧹 Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

async def main():
    """Main test function."""
    logger.info("🚀 Starting Multi-Agent System Test")

    # Test the system
    success = await test_multi_agent_system()

    if success:
        logger.info("🎉 All tests passed! Multi-agent system is ready.")
        logger.info("💡 You can now run: python process_multi_agent_corpus.py --input-dir enlitens_corpus/input_pdfs --output-file enlitens_knowledge_base.json")
    else:
        logger.error("❌ Tests failed. Please check the system configuration.")
        logger.info("🔧 Troubleshooting tips:")
        logger.info("   1. Ensure Ollama is running: ollama serve")
        logger.info("   2. Check GPU availability: nvidia-smi")
        logger.info("   3. Verify CUDA installation: nvcc --version")
        logger.info("   4. Update dependencies: pip install -r requirements.txt")

    return success

if __name__ == "__main__":
    # Set environment variables for optimal performance
    import os
    os.environ["OLLAMA_MAX_LOADED_MODELS"] = "1"
    os.environ["OLLAMA_MAX_QUEUE"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    success = asyncio.run(main())
    sys.exit(0 if success else 1)
