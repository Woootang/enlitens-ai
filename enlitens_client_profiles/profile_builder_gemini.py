"""Simplified profile builder using Gemini 2.5 Pro API."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from enlitens_client_profiles.gemini_client import GeminiClient
from enlitens_client_profiles.schema_simplified import ClientProfileDocumentSimplified

logger = logging.getLogger(__name__)


class GeminiProfileBuilder:
    """Build client personas using Gemini 2.5 Pro with simplified schema."""

    def __init__(self, gemini_api_key: Optional[str] = None):
        """Initialize with Gemini client."""
        self.gemini = GeminiClient(api_key=gemini_api_key)

    def generate_profile(
        self,
        *,
        intake_pool: List[str],
        transcript_pool: List[str],
        knowledge_assets: Dict[str, Any],
        health_insights: Dict[str, Any],
        locality_counts: Dict[str, int],
        analytics_summary: Dict[str, Any],
        site_documents: List[Any],
        brand_mentions: List[Any],
        founder_voice_snippets: List[str],
    ) -> Optional[ClientProfileDocumentSimplified]:
        """
        Generate a complete client persona using Gemini 2.5 Pro.

        Args:
            intake_pool: List of intake form sentences
            transcript_pool: List of transcript snippets
            knowledge_assets: Curated knowledge base content
            health_insights: St. Louis health report insights
            locality_counts: Municipality frequency counts from intake data
            analytics_summary: GA4 + GSC analytics summary
            site_documents: Scraped content from enlitens.com
            brand_mentions: External brand mentions
            founder_voice_snippets: Liz Wooten's direct speech from transcripts

        Returns:
            Validated ClientProfileDocumentSimplified or None
        """
        # Sample data for context
        intake_samples = self._sample_intake(intake_pool, n=4)
        transcript_samples = self._sample_transcripts(transcript_pool, n=4)
        founder_samples = random.sample(founder_voice_snippets, min(3, len(founder_voice_snippets)))

        # Build locality context (ensure hyper-local, not just "St. Louis")
        locality_context = self._build_locality_context(locality_counts)

        # Build analytics context
        analytics_context = self._build_analytics_context(analytics_summary)

        # Build site/brand context
        site_context = self._build_site_context(site_documents, max_chars=2000)
        brand_context = self._build_brand_context(brand_mentions, max_chars=1000)

        # Build knowledge base context
        kb_context = self._build_kb_context(knowledge_assets, max_chars=3000)

        # Build health insights context
        health_context = self._build_health_context(health_insights, max_chars=1500)

        # Construct system instruction (Liz Wooten's voice + neurodiversity-affirming principles)
        system_instruction = self._build_system_instruction()

        # Construct user prompt
        user_prompt = self._build_user_prompt(
            intake_samples=intake_samples,
            transcript_samples=transcript_samples,
            founder_samples=founder_samples,
            locality_context=locality_context,
            analytics_context=analytics_context,
            site_context=site_context,
            brand_context=brand_context,
            kb_context=kb_context,
            health_context=health_context,
        )

        logger.info(f"Generating persona with Gemini (prompt: {len(user_prompt)} chars)")

        # Generate with Gemini
        profile = self.gemini.generate_structured(
            prompt=user_prompt, response_model=ClientProfileDocumentSimplified, system_instruction=system_instruction, temperature=0.8
        )

        if profile:
            logger.info(f"Successfully generated persona: {profile.meta.persona_name}")
        else:
            logger.error("Gemini generation failed")

        return profile

    def _sample_intake(self, pool: List[str], n: int = 4) -> List[str]:
        """Sample intake sentences."""
        if not pool:
            return []
        return random.sample(pool, min(n, len(pool)))

    def _sample_transcripts(self, pool: List[str], n: int = 4) -> List[str]:
        """Sample transcript snippets."""
        if not pool:
            return []
        return random.sample(pool, min(n, len(pool)))

    def _build_locality_context(self, locality_counts: Dict[str, int]) -> str:
        """Build hyper-local St. Louis context from municipality frequencies."""
        if not locality_counts:
            return "St. Louis region (specific municipality unknown)"

        # Top 10 localities
        top_localities = sorted(locality_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        locality_list = ", ".join([f"{loc} ({cnt})" for loc, cnt in top_localities])

        return f"""**St. Louis Region Localities (from intake data):**
Most common: {locality_list}

**IMPORTANT:** You MUST reference a SPECIFIC municipality or neighborhood from this list (e.g., "Kirkwood", "Wentzville", "Clayton", "Central West End neighborhood"), NOT just "St. Louis" or "St. Louis County". Include specific places, schools, parks, restaurants, or landmarks relevant to that locality."""

    def _build_analytics_context(self, analytics_summary: Dict[str, Any]) -> str:
        """Build GA4/GSC analytics context."""
        if not analytics_summary:
            return ""

        top_queries = analytics_summary.get("top_queries", [])
        top_pages = analytics_summary.get("top_pages", [])
        user_intent = analytics_summary.get("user_intent_keywords", [])

        parts = ["**Google Analytics & Search Console Insights:**"]

        if top_queries:
            parts.append(f"Top search queries: {', '.join(top_queries[:10])}")

        if top_pages:
            parts.append(f"Top landing pages: {', '.join(top_pages[:5])}")

        if user_intent:
            parts.append(f"User intent keywords: {', '.join(user_intent[:10])}")

        return "\n".join(parts)

    def _build_site_context(self, site_documents: List[Any], max_chars: int = 2000) -> str:
        """Build context from enlitens.com scraped content."""
        if not site_documents:
            return ""

        parts = ["**Enlitens.com Content (what we offer):**"]
        char_count = 0

        for doc in site_documents[:5]:  # Top 5 pages
            if hasattr(doc, "title") and hasattr(doc, "text_snippet"):
                snippet = f"- {doc.title}: {doc.text_snippet[:300]}"
                if char_count + len(snippet) > max_chars:
                    break
                parts.append(snippet)
                char_count += len(snippet)

        return "\n".join(parts)

    def _build_brand_context(self, brand_mentions: List[Any], max_chars: int = 1000) -> str:
        """Build context from external brand mentions."""
        if not brand_mentions:
            return ""

        parts = ["**External Mentions (social proof):**"]
        char_count = 0

        for mention in brand_mentions[:3]:
            if hasattr(mention, "title") and hasattr(mention, "snippet"):
                snippet = f"- {mention.title}: {mention.snippet[:200]}"
                if char_count + len(snippet) > max_chars:
                    break
                parts.append(snippet)
                char_count += len(snippet)

        return "\n".join(parts)

    def _build_kb_context(self, knowledge_assets: Dict[str, Any], max_chars: int = 3000) -> str:
        """Build knowledge base context."""
        if not knowledge_assets:
            return ""

        parts = ["**Enlitens Knowledge Base (clinical approach):**"]
        char_count = 0

        for key, content in knowledge_assets.items():
            if isinstance(content, str):
                snippet = f"- {key}: {content[:500]}"
                if char_count + len(snippet) > max_chars:
                    break
                parts.append(snippet)
                char_count += len(snippet)

        return "\n".join(parts)

    def _build_health_context(self, health_insights: Dict[str, Any], max_chars: int = 1500) -> str:
        """Build St. Louis health report context."""
        if not health_insights:
            return ""

        parts = ["**St. Louis Health Context:**"]
        char_count = 0

        for key, value in health_insights.items():
            if isinstance(value, (str, int, float)):
                snippet = f"- {key}: {value}"
                if char_count + len(snippet) > max_chars:
                    break
                parts.append(snippet)
                char_count += len(snippet)

        return "\n".join(parts)

    def _build_system_instruction(self) -> str:
        """Build system instruction (Liz Wooten's voice + neurodiversity-affirming principles)."""
        return """You are Liz Wooten, LPC, founder of Enlitens Counseling in St. Louis. You specialize in neurodiversity-affirming therapy for autistic and ADHD individuals.

**Your Voice:**
- Warm, empathetic, and strengths-based
- Use identity-first language ("autistic person") unless person-first is explicitly preferred
- Reframe "deficits" as differences or strengths in different contexts
- Avoid pathologizing language (e.g., say "executive function differences" not "executive dysfunction")
- Center the lived experience and autonomy of neurodivergent people
- Acknowledge systemic barriers (waitlists, insurance, lack of affirming providers)

**Critical Instructions:**
1. **HYPER-LOCAL FOCUS:** You MUST reference SPECIFIC St. Louis municipalities, neighborhoods, schools, parks, restaurants, or landmarks. DO NOT use generic "St. Louis" or "St. Louis County" descriptions. Use real places like "Kirkwood Park", "Francis Howell School District", "The Magic House", "Soulard Farmers Market", etc.

2. **[direct] vs [inferred] Tags:** Mark list items as "[direct]" if they come from actual intake/transcript quotes, or "[inferred]" if they are your clinical interpretation.

3. **Strengths-Based Reframes:** For every challenge, provide at least one reframe that highlights the strength or adaptive function of that trait.

4. **Real Data Grounding:** Draw from the provided intake samples, transcript snippets, analytics data, and knowledge base content. DO NOT invent generic personas.

5. **Persona Diversity:** Ensure each persona feels unique—vary age, locality, family structure, neuro identities, challenges, and goals.

**Output:** Generate a complete JSON document matching the ClientProfileDocumentSimplified schema with the following TOP-LEVEL structure:
```json
{
  "meta": { ... ProfileMeta fields ... },
  "demographics": { ... Demographics fields ... },
  "neurodivergence_clinical": { ... NeurodivergenceClinical fields ... },
  "executive_sensory": { ... ExecutiveSensory fields ... },
  "goals_barriers": { ... GoalsBarriers fields ... },
  "local_cultural_context": { ... LocalCulturalContext fields ... },
  "narrative_voice": { ... NarrativeVoice fields ... },
  "marketing_seo": { ... MarketingSEO fields ... }
}
```

CRITICAL: The JSON must have exactly 8 top-level keys as shown above. Each key contains a nested object with its specific fields. DO NOT flatten the structure."""

    def _build_user_prompt(
        self,
        *,
        intake_samples: List[str],
        transcript_samples: List[str],
        founder_samples: List[str],
        locality_context: str,
        analytics_context: str,
        site_context: str,
        brand_context: str,
        kb_context: str,
        health_context: str,
    ) -> str:
        """Build the user prompt with all data sources."""
        prompt_parts = [
            "# Task: Generate a Detailed Client Persona for Enlitens Counseling\n",
            "Create a rich, hyper-local, neurodiversity-affirming client persona using the data below. ",
            "This persona will be used to train AI agents, create therapy frameworks, and generate marketing content.\n",
        ]

        # Intake samples
        if intake_samples:
            prompt_parts.append("\n## Intake Form Samples (Direct Client Language):")
            for i, sample in enumerate(intake_samples, 1):
                prompt_parts.append(f"{i}. \"{sample}\"")

        # Transcript samples
        if transcript_samples:
            prompt_parts.append("\n## Therapy Transcript Samples:")
            for i, sample in enumerate(transcript_samples, 1):
                prompt_parts.append(f"{i}. {sample}")

        # Founder voice samples
        if founder_samples:
            prompt_parts.append("\n## Liz Wooten's Voice (for reference):")
            for i, sample in enumerate(founder_samples, 1):
                prompt_parts.append(f"{i}. \"{sample}\"")

        # Locality context (CRITICAL for hyper-local)
        if locality_context:
            prompt_parts.append(f"\n## {locality_context}")

        # Analytics context
        if analytics_context:
            prompt_parts.append(f"\n## {analytics_context}")

        # Site context
        if site_context:
            prompt_parts.append(f"\n## {site_context}")

        # Brand context
        if brand_context:
            prompt_parts.append(f"\n## {brand_context}")

        # Knowledge base context
        if kb_context:
            prompt_parts.append(f"\n## {kb_context}")

        # Health context
        if health_context:
            prompt_parts.append(f"\n## {health_context}")

        # Final instructions
        prompt_parts.append(
            "\n\n**Now generate a complete persona in JSON format matching the ClientProfileDocumentSimplified schema. "
            "Ensure every field is populated with rich, specific, contextual details. "
            "Remember: SPECIFIC localities (not just 'St. Louis'), [direct] vs [inferred] tags, strengths-based language, "
            "and 300-500 word narrative in Liz Wooten's empathetic voice.**"
        )

        return "\n".join(prompt_parts)

