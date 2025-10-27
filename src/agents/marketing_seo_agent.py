"""
Marketing SEO Agent - Generates marketing and SEO content.
"""

import logging
from typing import Dict, Any
from .base_agent import BaseAgent
from src.synthesis.ollama_client import OllamaClient
from src.models.enlitens_schemas import MarketingContent, SEOContent

logger = logging.getLogger(__name__)

class MarketingSEOAgent(BaseAgent):
    """Agent specialized in marketing and SEO content generation."""

    def __init__(self):
        super().__init__(
            name="MarketingSEO",
            role="Marketing and SEO Content Generation",
            model="llama3.1:8b"  # Use llama for creative marketing content
        )
        self.ollama_client = None

    async def initialize(self) -> bool:
        """Initialize the marketing SEO agent."""
        try:
            self.ollama_client = OllamaClient(default_model=self.model)
            self.is_initialized = True
            logger.info(f"✅ {self.name} agent initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            return False

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate marketing and SEO content."""
        try:
            final_context = context.get("final_context", {})
            clinical_content = final_context.get("clinical_content", {})
            research_content = final_context.get("research_content", {})
            
            # Generate marketing content using two-stage approach
            marketing_prompt = f"""
You are a marketing strategist for Enlitens, a neuroscience-based therapy practice in St. Louis.

CLINICAL INSIGHTS:
{clinical_content.get('treatment_approaches', [])}

RESEARCH FINDINGS:
{research_content.get('key_findings', [])}

Create compelling marketing content:
1. Headlines (3-5 attention-grabbing headlines)
2. Taglines (3-5 memorable slogans)
3. Value propositions (3-5 unique benefits)
4. Benefits (3-5 client outcomes)
5. Pain points (3-5 problems we solve)
6. Social proof (3-5 credibility statements)

Keep each item under 18 words. Use plain language, no numbering.
"""

            # Generate free-form marketing notes
            marketing_notes = await self.ollama_client.generate_text(
                prompt=marketing_prompt,
                temperature=0.8,
                num_predict=1024
            )
            
            # Format into structured JSON
            formatting_prompt = f"""
Convert these marketing notes into JSON with fields:
{{
  "headlines": [list of 3-5 strings],
  "taglines": [list of 3-5 strings],
  "value_propositions": [list of 3-5 strings],
  "benefits": [list of 3-5 strings],
  "pain_points": [list of 3-5 strings],
  "social_proof": [list of 3-5 strings]
}}

NOTES:
{marketing_notes}

Return valid JSON only.
"""

            qwen_client = OllamaClient(default_model="qwen3:32b")
            marketing_result = await qwen_client.generate_structured_response(
                prompt=formatting_prompt,
                response_model=MarketingContent,
                temperature=0.2,
                max_retries=3
            )

            # Generate SEO content
            seo_prompt = f"""
Generate SEO-optimized content for a neuroscience therapy practice.

TOPICS:
{research_content.get('key_findings', [])}

Create SEO content:
1. Meta titles (3-5 titles, 50-60 chars each)
2. Meta descriptions (3-5 descriptions, 150-160 chars each)
3. Keywords (10-15 relevant keywords)
4. Content topics (5-10 blog/article topics)
5. FAQ questions (5-10 common questions)

Return as JSON.
"""

            seo_result = await qwen_client.generate_structured_response(
                prompt=seo_prompt,
                response_model=SEOContent,
                temperature=0.3,
                max_retries=3
            )

            return {
                "marketing_content": marketing_result.model_dump() if marketing_result else MarketingContent().model_dump(),
                "seo_content": seo_result.model_dump() if seo_result else SEOContent().model_dump(),
                "generation_quality": "high"
            }

        except Exception as e:
            logger.error(f"Marketing SEO generation failed: {e}")
            return {
                "marketing_content": MarketingContent().model_dump(),
                "seo_content": SEOContent().model_dump()
            }

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate the marketing and SEO content."""
        marketing_content = output.get("marketing_content", {})
        seo_content = output.get("seo_content", {})
        
        has_marketing = any([
            marketing_content.get("headlines"),
            marketing_content.get("value_propositions")
        ])
        
        has_seo = any([
            seo_content.get("meta_titles"),
            seo_content.get("keywords")
        ])
        
        return has_marketing or has_seo

    async def cleanup(self):
        """Clean up resources."""
        logger.info(f"Cleaning up {self.name} agent")
