"""
Marketing SEO Agent - Generates marketing and SEO content.
"""

import json
import logging
from typing import Dict, Any

from .base_agent import BaseAgent
from src.models.enlitens_schemas import MarketingContent, SEOContent
from src.synthesis.few_shot_library import FEW_SHOT_LIBRARY
from src.synthesis.ollama_client import OllamaClient

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

            research_findings = (
                research_content.get("key_findings")
                or research_content.get("findings")
                or []
            )
            clinical_focus = (
                clinical_content.get("treatment_approaches")
                or clinical_content.get("interventions")
                or []
            )
            
            # Generate marketing content using creative approach
            # IMPORTANT: Marketing content is CREATIVE and forward-looking
            # It's about IDEAS for how to talk about the research, not extraction from sources
            marketing_few_shot = FEW_SHOT_LIBRARY.render_for_prompt(
                task="marketing_content",
                query=json.dumps(research_findings[:5]),
                k=1,
            )

            marketing_examples = (
                "FEW-SHOT EXEMPLAR (tone + compliance reference):\n"
                f"{marketing_few_shot}\n\n" if marketing_few_shot else ""
            )

            marketing_prompt = f"""
You are a marketing strategist for Enlitens, a neuroscience-based therapy practice in St. Louis.

Your goal is to CREATE compelling marketing messages inspired by (but not limited to) the research themes.

Compliance guardrails:
- No testimonials, guarantees, or unverifiable claims.
- Anchor messaging in St. Louis community needs.
- Keep tone rebellious yet trauma-informed.

{marketing_examples}
RESEARCH THEMES (inspiration only):
{research_findings[:5]}

CLINICAL FOCUS:
{clinical_focus[:5]}

CREATE marketing content that positions Enlitens as the neuroscience therapy leader in St. Louis:

1. Headlines (3-5 attention-grabbing headlines under 18 words each)
   - Should speak to client pain points
   - Emphasize neuroscience-based solutions

2. Taglines (3-5 memorable taglines under 12 words each)
   - Rebellious, direct tone
   - Challenge traditional therapy narratives

3. Value propositions (3-5 unique benefits under 20 words each)
   - What makes Enlitens different
   - Tie neuroscience to relief

4. Benefits (3-5 client outcomes under 15 words each)
   - Tangible wins clients will experience
   - Mix emotional relief + practical improvements

5. Pain points (3-5 problems we solve under 15 words each)
   - Real struggles St. Louis clients face
   - ADHD, anxiety, trauma, treatment resistance

NOTE: Generate NEW creative content - don't just quote research. This is about IDEAS and MARKETING ANGLES.

Return ONLY valid JSON in this exact format:
{{
  "headlines": ["headline 1", "headline 2", "headline 3"],
  "taglines": ["tagline 1", "tagline 2", "tagline 3"],
  "value_propositions": ["value 1", "value 2", "value 3"],
  "benefits": ["benefit 1", "benefit 2", "benefit 3"],
  "pain_points": ["pain 1", "pain 2", "pain 3"]
}}
"""

            marketing_result = await self.ollama_client.generate_structured_response(
                prompt=marketing_prompt,
                response_model=MarketingContent,
                temperature=0.3,
                max_retries=3,
                use_cot_prompt=False,  # CRITICAL: Disable CoT for creative content
                enforce_grammar=True,
                model="qwen3:32b",
            )

            # Generate SEO content
            seo_prompt = f"""
Generate SEO-optimized content for Enlitens, a neuroscience-based therapy practice in St. Louis.

RESEARCH THEMES (inspiration):
{research_findings[:5]}

TARGET AUDIENCE: St. Louis adults with ADHD, anxiety, trauma, autism

CREATE SEO content optimized for local search and mental health queries:

1. Primary keywords (5-10 keywords)
   - Focus on "neuroscience therapy St. Louis", "ADHD specialist", etc.

2. Secondary keywords (5-10 keywords)
   - Related terms like "trauma therapy", "anxiety treatment"

3. Long-tail keywords (5-10 specific phrases)
   - "neuroscience-based ADHD treatment St. Louis"

4. Meta descriptions (3-5 descriptions, 150-160 characters each)
   - Compelling descriptions for search results

5. Title tags (3-5 titles, 50-60 characters each)
   - SEO-optimized page titles

6. Content topics (5-10 blog/article ideas)
   - Topics that would rank well and attract clients

NOTE: Generate creative, searchable content - not just quotes from research.

Return ONLY valid JSON in this exact format:
{{
  "primary_keywords": ["keyword1", "keyword2"],
  "secondary_keywords": ["keyword1", "keyword2"],
  "long_tail_keywords": ["phrase1", "phrase2"],
  "meta_descriptions": ["desc1", "desc2"],
  "title_tags": ["title1", "title2"],
  "content_topics": ["topic1", "topic2"]
}}
"""

            seo_result = await self.ollama_client.generate_structured_response(
                prompt=seo_prompt,
                response_model=SEOContent,
                temperature=0.3,
                max_retries=3,
                use_cot_prompt=False,  # CRITICAL: Disable CoT for creative SEO content
                enforce_grammar=True,
                model="qwen3:32b",
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
        marketing_content = output.get("marketing_content") or {}
        seo_content = output.get("seo_content") or {}

        marketing_fields = ("headlines", "taglines", "value_propositions", "benefits", "pain_points")
        has_marketing = any(bool(marketing_content.get(field)) for field in marketing_fields)

        seo_fields = (
            "meta_descriptions",
            "title_tags",
            "primary_keywords",
            "secondary_keywords",
            "long_tail_keywords",
            "content_topics",
        )
        has_seo = any(bool(seo_content.get(field)) for field in seo_fields)

        return has_marketing or has_seo

    async def cleanup(self):
        """Clean up resources."""
        logger.info(f"Cleaning up {self.name} agent")
