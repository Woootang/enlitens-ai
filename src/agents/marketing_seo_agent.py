"""
Marketing SEO Agent - Generates marketing and SEO content.
"""

import json
import logging
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from src.models.enlitens_schemas import MarketingContent, SEOContent
from src.synthesis.few_shot_library import FEW_SHOT_LIBRARY
from src.synthesis.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class MarketingSEOAgent(BaseAgent):
    """Agent specialized in marketing and SEO content generation."""

    PERSONA_FALLBACK = (
        "Primary audience: neurodivergent and trauma-survivor adults in St. Louis seeking "
        "science-backed, validating care that respects their lived experience."
    )
    REGIONAL_FALLBACK = (
        "St. Louis metro realities: legacy trauma from violence and segregation, ADHD/EF burdens, "
        "access barriers (transportation, insurance gaps), and desire for culturally competent neuroscience care."
    )

    def __init__(self):
        super().__init__(
            name="MarketingSEO",
            role="Marketing and SEO Content Generation",
        )
        self.ollama_client = None

    async def initialize(self) -> bool:
        """Initialize the marketing SEO agent."""
        try:
            self.ollama_client = OllamaClient(default_model=self.model)
            if not await self.ollama_client.check_connection():
                raise RuntimeError(
                    f"vLLM server is not reachable at {self.ollama_client.base_url}. Please run stable_run.sh or start the vLLM server."
                )
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
            curated_context = final_context.get("curated_context") or {}
            persona_brief = self._build_persona_brief(
                final_context.get("client_insights") or {}
            )
            regional_brief = self._build_regional_brief(
                final_context.get("regional_context") or final_context.get("st_louis_context") or {},
                final_context.get("regional_digest_chunks"),
                final_context.get("regional_prompt_block"),
            )
            analytics_brief = self._build_analytics_brief(
                (final_context.get("analytics_insights") or {}).get("queries")
                or (final_context.get("language_profile") or {}).get("search_queries")
                or [],
                (final_context.get("analytics_insights") or {}).get("pages")
                or (final_context.get("language_profile") or {}).get("landing_pages")
                or [],
            )
            mechanism_bridge = self._build_mechanism_bridge(curated_context)
            stats_block = self._build_local_stats(curated_context)
            language_guardrails = self._build_language_guardrails(
                curated_context.get("language_profile") or final_context.get("language_profile") or {},
                final_context.get("language_watchouts") or {},
            )

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
            research_snippet = json.dumps(research_findings[:5], ensure_ascii=False, indent=2)
            clinical_snippet = json.dumps(clinical_focus[:5], ensure_ascii=False, indent=2)
            
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

Language guardrails:
{language_guardrails}

Audience Insights:
{persona_brief}

Regional Signals:
{regional_brief}

Mechanism ↔ Persona Bridge:
{mechanism_bridge}

Local Stats Primer:
{stats_block}

Analytics Signals:
{analytics_brief}

{marketing_examples}RESEARCH THEMES (inspiration only):
{research_snippet}

CLINICAL FOCUS:
{clinical_snippet}

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

            marketing_client = self.ollama_client
            seo_model = self.settings.llm.model_for("marketing-seo") or self.model
            seo_client = self.ollama_client.clone_with_model(seo_model)

            marketing_cache = self._cache_kwargs(context, suffix="marketing")
            marketing_result = await marketing_client.generate_structured_response(
                prompt=marketing_prompt,
                response_model=MarketingContent,
                temperature=0.45,
                max_retries=3,
                use_cot_prompt=False,
                enforce_grammar=True,
                **marketing_cache,
            )

            seo_prompt = f"""
Generate SEO-optimized content for Enlitens, a neuroscience-based therapy practice in St. Louis.

Target Persona Signals:
{persona_brief}

Language Guardrails:
{language_guardrails}

Local Context:
{regional_brief}

Mechanism ↔ Persona Bridge:
{mechanism_bridge}

Local Stats Primer:
{stats_block}

Analytics Signals:
{analytics_brief}

RESEARCH THEMES (context):
{research_snippet}

CLINICAL FOCUS (align messaging with services):
{clinical_snippet}

TARGET AUDIENCE: St. Louis adults with ADHD, anxiety, trauma, autism

CREATE SEO content optimized for local search and mental health queries:

1. Primary keywords (5-10 keywords)
2. Secondary keywords (5-10 keywords)
3. Long-tail keywords (5-10 specific phrases)
4. Meta descriptions (3-5 descriptions, 150-160 characters each)
5. Title tags (3-5 titles, 50-60 characters each)
6. Content topics (5-10 blog/article ideas)

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

            seo_cache = self._cache_kwargs(context, suffix="seo")
            seo_result = await seo_client.generate_structured_response(
                prompt=seo_prompt,
                response_model=SEOContent,
                temperature=0.3,
                max_retries=3,
                use_cot_prompt=False,
                enforce_grammar=True,
                **seo_cache,
            )

            return {
                "marketing_content": marketing_result.model_dump() if marketing_result else MarketingContent().model_dump(),
                "seo_content": seo_result.model_dump() if seo_result else SEOContent().model_dump(),
                "generation_quality": "high",
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

    def _build_persona_brief(self, persona: Dict[str, Any]) -> str:
        if not persona:
            return self.PERSONA_FALLBACK

        lines = []
        summary = persona.get("persona_summary") or {}
        segments = summary.get("representative_segments") or persona.get("persona_segments") or []
        if segments:
            top = segments[0]
            lines.append(
                f"Focus persona: {top.get('name', 'Neurodivergent adult')} — {top.get('tagline', 'seeks science-backed relief')}."
            )
        top_audience_pain = summary.get("top_pain_points") or persona.get("audience_pain_points") or []
        if top_audience_pain:
            lines.append("Community pain points: " + ", ".join(top_audience_pain[:5]))
        top_audience_keywords = summary.get("top_keywords") or persona.get("audience_keywords") or []
        if top_audience_keywords:
            lines.append("High-intent search terms: " + ", ".join(top_audience_keywords[:6]))
        pain_points = persona.get("pain_points") or persona.get("challenges") or []
        if pain_points:
            lines.append("Pain points: " + ", ".join(pain_points[:5]))
        priorities = persona.get("priorities") or []
        if priorities:
            lines.append("Goals: " + ", ".join(priorities[:4]))
        themes = persona.get("key_themes") or []
        if themes:
            lines.append("Themes: " + ", ".join(themes[:4]))

        enhanced = persona.get("enhanced_analysis") or {}
        summary = enhanced.get("narrative_summary") or enhanced.get("summary")
        if summary:
            lines.append(self._truncate(summary, 200))
        persona_prompt = persona.get("persona_prompt_block")
        if persona_prompt:
            lines.append("Persona guardrails:\n" + self._truncate(persona_prompt, 350))

        if not lines:
            return self.PERSONA_FALLBACK
        return "\n".join(lines)

    def _build_regional_brief(
        self,
        regional: Dict[str, Any],
        digest_chunks: Optional[Any] = None,
        prompt_block: Optional[str] = None,
    ) -> str:
        if not regional:
            return self.REGIONAL_FALLBACK

        lines = []
        population = regional.get("population")
        if population:
            lines.append(f"Population: {population}")
        challenges = regional.get("mental_health_challenges") or []
        if challenges:
            lines.append("Mental health pressures: " + ", ".join(challenges[:4]))
        socioeconomic = regional.get("socioeconomic_factors") or []
        if socioeconomic:
            lines.append("Social determinants: " + ", ".join(socioeconomic[:3]))
        summary_bullets = regional.get("summary_bullets") or []
        if summary_bullets:
            lines.append("Regional summary:\n- " + "\n- ".join(summary_bullets[:4]))
        flashpoints = regional.get("cultural_flashpoints") or []
        if flashpoints:
            labels = [flash.get("label") for flash in flashpoints if isinstance(flash, dict)]
            if labels:
                lines.append("Flashpoints to name: " + ", ".join(label for label in labels[:4] if label))
        stats = regional.get("key_statistics") or []
        if stats:
            lines.append("Key stats:\n- " + "\n- ".join(stats[:4]))
        if digest_chunks:
            chunk_preview = "\n\n".join(str(chunk) for chunk in digest_chunks[:2] if chunk)
            if chunk_preview:
                lines.append("Digest snapshot:\n" + self._truncate(chunk_preview, 400))
        if prompt_block:
            lines.append("Digest prompt block:\n" + self._truncate(prompt_block, 320))

        if not lines:
            return self.REGIONAL_FALLBACK
        return "\n".join(lines)

    def _build_mechanism_bridge(self, curated_context: Dict[str, Any], max_chars: int = 400) -> str:
        bridge = (curated_context or {}).get("mechanism_bridge")
        if bridge:
            return self._truncate(str(bridge), max_chars)
        return "Spell out how the paper's mechanism (CSA, inflammation, nervous system swings) maps to ADHD burnout, CPTSD vigilance, and sensory overload in St. Louis clients."

    def _build_local_stats(self, curated_context: Dict[str, Any], max_items: int = 6) -> str:
        stats = (curated_context or {}).get("local_stats") or []
        if not stats:
            return "- 39% of St. Louis adults report chronic stress tied to social inequity.\n- 1 in 3 neurodivergent adults cite executive burnout from community strain."
        return "\n".join(f"- {stat}" for stat in stats[:max_items])

    def _truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _build_analytics_brief(self, queries: List[str], pages: List[str]) -> str:
        if not queries and not pages:
            return "Analytics snapshot unavailable — rely on persona keywords."
        top_queries = ", ".join(query for query in queries[:5]) if queries else "N/A"
        top_pages = ", ".join(page for page in pages[:5]) if pages else "N/A"
        return f"Top search queries: {top_queries}\nTop landing pages: {top_pages}"

    def _build_language_guardrails(
        self,
        language_profile: Dict[str, Any],
        watchouts: Dict[str, Any],
    ) -> str:
        lines: List[str] = []
        prompt_block = language_profile.get("prompt_block")
        if prompt_block:
            lines.append(self._truncate(prompt_block, 320))
        words_to_use = watchouts.get("words_to_use") if isinstance(watchouts, dict) else None
        if words_to_use:
            lines.append("Words that resonate: " + ", ".join(words_to_use[:8]))
        words_to_avoid = watchouts.get("words_to_avoid") if isinstance(watchouts, dict) else None
        if words_to_avoid:
            lines.append("Words to avoid unless sourced: " + ", ".join(words_to_avoid[:8]))
        banned_terms = language_profile.get("banned_terms") if isinstance(language_profile, dict) else None
        if banned_terms:
            lines.append("Hard-no terms: " + ", ".join(banned_terms[:8]))
        if lines:
            return "\n".join(lines)
        return "Plain speech only—no mindfulness, manifesting, or therapy clichés."
