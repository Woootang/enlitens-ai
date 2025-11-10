"""
Founder Voice Agent for the Enlitens Multi-Agent System.

This agent captures and integrates Liz Wooten's authentic voice, personality,
and clinical philosophy into all content generation.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from .base_agent import BaseAgent
from ..synthesis.ollama_client import OllamaClient
from ..models.enlitens_schemas import (
    MarketingContent, SEOContent, WebsiteCopy, BlogContent,
    SocialMediaContent, ContentCreationIdeas, VerifiedStatistic, Citation
)

logger = logging.getLogger(__name__)

class FounderVoiceAgent(BaseAgent):
    """
    Specialized agent for capturing and integrating founder voice.
    """

    def __init__(self):
        super().__init__(
            name="FounderVoiceAgent",
            role="Liz Wooten Voice and Brand Integration Specialist",
        )
        self.ollama_client: Optional[OllamaClient] = None

        # Liz Wooten's authentic voice characteristics
        self.founder_persona = {
            "communication_style": [
                "Direct, authentic, no-nonsense approach",
                "Rebellious against traditional therapy norms",
                "Grounded in neuroscience, not fluff",
                "Empowering and hopeful messaging",
                "St. Louis roots - real talk for real people",
                "Trauma-informed but not trauma-focused",
                "Strength-based rather than deficit-based"
            ],
            "key_phrases": [
                "Your brain isn't broken, it's adapting",
                "Traditional therapy missed the neurobiology",
                "Stop treating symptoms, heal the brain",
                "Neuroscience shows us the way forward",
                "You're not disordered, you're responding to your environment",
                "Real therapy for real people in the real world"
            ],
            "clinical_philosophy": [
                "Bottom-up (body/sensory) meets top-down (cognitive)",
                "Neuroplasticity as hope and possibility",
                "Interoceptive awareness as foundation",
                "Executive function support through neuroscience",
                "Social connection through brain-based understanding"
            ]
        }

        # St. Louis client challenges from intakes
        self.client_challenges = [
            "ADHD and executive function struggles",
            "Anxiety and overwhelm in daily life",
            "Trauma responses and emotional regulation",
            "Social connection and relationship difficulties",
            "Work/school performance and organization",
            "Self-esteem and identity issues",
            "Sleep problems and emotional dysregulation",
            "Treatment resistance and medication questions"
        ]

    async def initialize(self) -> bool:
        """Initialize the founder voice agent."""
        try:
            self.ollama_client = OllamaClient()
            self.is_initialized = True
            logger.info("✅ Founder Voice Agent initialized with Liz persona")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Founder Voice Agent: {e}")
            return False

    async def _structured_generation(
        self,
        prompt: str,
        response_model,
        context: Dict[str, Any],
        suffix: str,
        **kwargs,
    ):
        cache_kwargs = self._cache_kwargs(context, suffix=suffix)
        return await self.ollama_client.generate_structured_response(
            prompt=prompt,
            response_model=response_model,
            **kwargs,
            **cache_kwargs,
        )

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate founder voice into all content types.

        Args:
            context: Contains clinical_data and other intermediate results

        Returns:
            Dictionary containing marketing, SEO, website, blog, and social content
        """
        try:
            clinical_data = context.get("clinical_data", {})
            enhanced_data = context.get("enhanced_data", {})
            document_id = context.get("document_id", "unknown")

            logger.info(f"🎙️ Founder Voice Agent processing: {document_id}")

            # Generate marketing content in Liz's voice
            marketing_content = await self._generate_marketing_content(
                clinical_data, context
            )

            # Generate SEO content optimized for St. Louis searches
            seo_content = await self._generate_seo_content(
                clinical_data, context
            )

            # Create website copy that converts
            website_copy = await self._generate_website_copy(
                clinical_data, context
            )

            # Generate blog content that engages
            blog_content = await self._generate_blog_content(
                clinical_data, context
            )

            # Create social media content that connects
            social_media_content = await self._generate_social_media_content(
                clinical_data, context
            )

            # Generate content creation ideas
            content_creation_ideas = await self._generate_content_ideas(
                clinical_data, context
            )

            return {
                "marketing_content": marketing_content.model_dump(),
                "seo_content": seo_content.model_dump(),
                "website_copy": website_copy.model_dump(),
                "blog_content": blog_content.model_dump(),
                "social_media_content": social_media_content.model_dump(),
                "content_creation_ideas": content_creation_ideas.model_dump(),
                "agent_name": self.name,
                "processing_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Founder voice integration failed: {e}")
            return {}

    async def _generate_marketing_content(self, clinical_data: Dict[str, Any],
                                         context: Dict[str, Any]) -> MarketingContent:
        """Generate marketing content in Liz's authentic voice."""
        try:
            document_text = context.get("document_text", "")
            document_id = context.get("document_id", "unknown")
            summary = self._summarize_research(document_text, max_chars=2500)
            
            # Use curated context if available, otherwise load personas normally
            curated_context = context.get("curated_context")
            if curated_context:
                personas_context = curated_context.get("personas_text", "")
                voice_guide = curated_context.get("voice_guide", "")
                health_context = curated_context.get("health_brief", "")
            else:
                personas_context = self._load_personas_context()
                voice_guide = ""
                health_context = ""
            
            voice_section = f"\n\nLIZ'S VOICE GUIDE:\n{voice_guide}\n" if voice_guide else ""
            health_section = f"\n\nLOCAL HEALTH CONTEXT (St. Louis):\n{health_context}\n" if health_context else ""
            
            prompt = f"""
You are Liz Wooten creating UNIQUE marketing copy for document {document_id}.

VOICE GUARDRAILS:
- Direct, rebellious, never corporate
- "Your brain isn't broken, it's adapting"
- Ground everything in neuroscience
- Hope + proof, not fluff
{voice_section}
RESEARCH FROM THIS SPECIFIC DOCUMENT:
{summary}

RELEVANT CLIENT PROFILES (10 selected for THIS paper):
{personas_context}
{health_section}
CRITICAL: Generate COMPLETELY UNIQUE copy for THIS document.
- Reference THIS document's specific findings
- Use varied language and angles
- NO generic templates or repetitive phrases
- Each headline/tagline must be distinct

Generate 3-5 items for each:
- headlines: Punchy, research-specific (≤18 words)
- taglines: Rebel flair, tied to THIS research
- value_propositions: THIS research → relief
- benefits: Emotional + tangible from THIS research
- pain_points: Real client language

NO social proof, testimonials, or practice stats (FTC).
Return ONLY valid JSON.
"""

            # Use default Qwen model (no llama fallback)
            raw_notes = await self.ollama_client.generate_text(
                prompt=prompt,
                temperature=0.85,  # Higher temp for more variation
                num_predict=1024
            )
            logger.debug(
                "Founder fallback notes length=%d sample=%s",
                len(raw_notes),
                raw_notes[:500]
            )

            normalized_notes = self._normalize_notes(raw_notes)

            structuring_prompt = f"""
You are a JSON formatter. Convert the following Liz Wooten marketing notes into JSON with fields:
{{{{
  "headlines": [string],
  "taglines": [string],
  "value_propositions": [string],
  "benefits": [string],
  "pain_points": [string]
}}}}
Each list must have 3-5 items. Preserve Liz's voice. Content:
---
{normalized_notes}
---
Respond with valid JSON only. DO NOT include social_proof field (removed for FTC compliance).
"""
            try:
                structured = await self._structured_generation(
                    prompt=structuring_prompt,
                    response_model=MarketingContent,
                    context=context,
                    suffix="founder_marketing_structured",
                    temperature=0.2,
                    max_retries=3,
                    use_cot_prompt=False,  # Creative marketing content - no CoT
                )
            except Exception as err:
                logger.error("Founder structured fallback failed: %s", err)
                structured = None

            if structured:
                logger.debug(
                    "Founder structured fallback: %s",
                    {k: structured.model_dump().get(k) for k in ("headlines", "value_propositions", "benefits")}
                )
                return structured

            logger.warning("Founder structured fallback failed to produce output; returning empty model")
            return MarketingContent()
        except Exception as e:
            logger.error(f"Marketing content generation failed: {e}")
            return MarketingContent()

    def _summarize_research(self, text: str, max_chars: int = 1200) -> str:
        snippet = text.strip().replace("\n", " ")
        if len(snippet) <= max_chars:
            return snippet
        return snippet[:max_chars - 3].rstrip() + "..."

    def _normalize_notes(self, notes: str) -> str:
        try:
            data = json.loads(notes)
            if isinstance(data, dict):
                flattened = []
                for key, value in data.items():
                    if isinstance(value, list):
                        flattened.extend(str(item) for item in value)
                    else:
                        flattened.append(str(value))
                return "\n".join(flattened)
        except Exception:
            pass

        lines = []
        for line in notes.splitlines():
            stripped = line.strip("- •\t ")
            if stripped:
                lines.append(stripped)
        return "\n".join(lines)

    def _load_personas_context(self, max_personas: int = 10) -> str:
        """Load a sample of client personas to inform content generation."""
        try:
            import glob
            from pathlib import Path
            
            personas_dir = Path("/home/antons-gs/enlitens-ai/enlitens_client_profiles/profiles")
            persona_files = list(personas_dir.glob("persona_*.json"))
            
            if not persona_files:
                logger.warning("No persona files found")
                return "No client profiles available."
            
            # Load a random sample
            import random
            sample_files = random.sample(persona_files, min(max_personas, len(persona_files)))
            
            personas_summary = []
            for pfile in sample_files:
                try:
                    with open(pfile, 'r') as f:
                        persona = json.load(f)
                    
                    # Extract key info
                    demo = persona.get('demographics', {})
                    challenges = persona.get('current_challenges', {})
                    
                    summary = f"- {demo.get('age_range', 'Adult')} with {', '.join(challenges.get('primary_concerns', [])[:2])}"
                    personas_summary.append(summary)
                except Exception as e:
                    logger.debug(f"Failed to load persona {pfile}: {e}")
                    continue
            
            if not personas_summary:
                return "Client profiles: Adults with ADHD, anxiety, trauma, and autism seeking neuroscience-based support."
            
            return "Real client profiles:\n" + "\n".join(personas_summary[:10])
            
        except Exception as e:
            logger.warning(f"Failed to load personas: {e}")
            return "Client profiles: Adults with ADHD, anxiety, trauma, and autism seeking neuroscience-based support."

    async def _generate_seo_content(self, clinical_data: Dict[str, Any],
                                  context: Dict[str, Any]) -> SEOContent:
        """Generate SEO content optimized for St. Louis mental health searches."""
        try:
            prompt = f"""
You are Liz Wooten optimizing content for St. Louis clients searching for real help.
Create SEO content that ranks well and speaks directly to local challenges.

SEO Context:
- Location: St. Louis, Missouri
- Target Audience: Adults with ADHD, anxiety, trauma, autism
- Search Intent: "neuroscience therapy St. Louis", "ADHD specialist near me"
- Competition: Traditional therapy practices, psychiatrists, counselors

Clinical Data:
{clinical_data}

St. Louis Mental Health Landscape:
- High trauma rates, poverty, racial disparities
- ADHD, anxiety, depression are common
- Clients want practical, science-based help
- Skeptical of traditional "talk therapy" approaches

CRITICAL: You MUST return ONLY valid JSON. NO markdown, NO headers, NO formatting.

Generate 5-10 items for each field below and return as JSON:

{{
  "primary_keywords": ["neuroscience therapy St. Louis", "ADHD specialist St. Louis", ...],
  "secondary_keywords": ["trauma therapy", "anxiety treatment", ...],
  "long_tail_keywords": ["neuroscience-based ADHD treatment St. Louis", ...],
  "meta_descriptions": ["Science-backed therapy for ADHD and anxiety in St. Louis...", ...],
  "title_tags": ["Neuroscience Therapy St. Louis | ADHD & Anxiety Treatment", ...],
  "content_topics": ["Why Your ADHD Meds Stopped Working", ...]
}}

Write in Liz's voice - make it authentic and trustworthy.
Target the specific pain points St. Louis clients face.
Position Enlitens as the neuroscience alternative to traditional therapy.

RETURN ONLY THE JSON OBJECT. NO OTHER TEXT.
"""

            response = await self._structured_generation(
                prompt=prompt,
                response_model=SEOContent,
                context=context,
                suffix="founder_seo",
                temperature=0.3,
                max_retries=3,
                use_cot_prompt=False,
            )

            return response or SEOContent()

        except Exception as e:
            logger.error(f"SEO content generation failed: {e}")
            return SEOContent()

    async def _generate_website_copy(self, clinical_data: Dict[str, Any],
                                   context: Dict[str, Any]) -> WebsiteCopy:
        """Generate website copy that converts visitors to clients."""
        try:
            document_text = context.get("document_text", "")
            document_id = context.get("document_id", "unknown")
            summary = self._summarize_research(document_text, max_chars=2500)

            # Load personas for context
            personas_context = self._load_personas_context()

            prompt = f"""
You are Liz Wooten writing UNIQUE website copy for document {document_id}.
Each document requires COMPLETELY DIFFERENT copy based on its specific research findings.

RESEARCH FROM THIS SPECIFIC DOCUMENT:
{summary}

CLINICAL INSIGHTS FROM THIS DOCUMENT:
{clinical_data}

REAL CLIENT PROFILES (use these to inform tone and pain points):
{personas_context}

CRITICAL RULES:
1. DO NOT use generic templates or examples
2. Every piece of copy MUST be unique to THIS document's research
3. Reference specific findings from the research above
4. Use different client pain points for each document
5. Vary your language, structure, and approach

Generate 3-5 UNIQUE paragraph-length STRINGS (NOT objects) for each field:

about_sections: [
  "Full paragraph from Liz's perspective about THIS research...",
  "Another unique paragraph...",
  ...
]

feature_descriptions: [
  "Service Name: Full description of service relevant to THIS research...",
  ...
]

benefit_statements: [
  "Complete sentence describing benefit from THIS research...",
  ...
]

faq_content: [
  "Q: Question? A: Full answer addressing THIS research...",
  "Q: Another question? A: Another full answer...",
  ...
]

service_descriptions: [
  "Service Name: Complete paragraph describing service for THIS research...",
  ...
]

CRITICAL: Each item must be a SINGLE STRING, not an object with fields.
Use Liz's voice. Ground in THIS document's findings.

RETURN ONLY VALID JSON with string arrays.
"""

            response = await self._structured_generation(
                prompt=prompt,
                response_model=WebsiteCopy,
                context=context,
                suffix="founder_website",
                temperature=0.8,  # Higher temp for more variation
                max_retries=3,
                use_cot_prompt=False,
            )

            return response or WebsiteCopy()

        except Exception as e:
            logger.error(f"Website copy generation failed: {e}")
            return WebsiteCopy()

    async def _generate_blog_content(self, clinical_data: Dict[str, Any],
                                   context: Dict[str, Any]) -> BlogContent:
        """Generate blog content that establishes thought leadership."""
        try:
            document_text = context.get("document_text", "")
            document_id = context.get("document_id", "unknown")
            summary = self._summarize_research(document_text, max_chars=2500)
            
            # Load personas
            personas_context = self._load_personas_context()

            prompt = f"""
You are Liz Wooten writing UNIQUE blog content for document {document_id}.

RESEARCH FROM THIS SPECIFIC DOCUMENT:
{summary}

CLINICAL INSIGHTS FROM THIS DOCUMENT:
{clinical_data}

REAL CLIENT PROFILES:
{personas_context}

CRITICAL RULES:
1. Generate content UNIQUE to THIS document's research
2. Reference specific findings from THIS document
3. NO generic templates - vary language and structure
4. Use different client scenarios from the personas
5. Each article idea must be distinct and research-specific

Generate 5-10 UNIQUE items for each field:

article_ideas: Titles based on THIS document's findings
blog_outlines: Structures specific to THIS research
talking_points: Key messages from THIS document
expert_quotes: Liz's perspective on THIS research
statistics: ONLY from THIS document's research (with citations)
case_studies: Hypothetical examples based on THIS research
how_to_guides: Practical exercises from THIS research
myth_busting: Myths challenged by THIS research

STATISTICS: Must cite THIS document with exact quotes
CASE STUDIES: Mark "[HYPOTHETICAL]" and base on THIS research
NO practice data, testimonials, or fabricated scenarios

Use Liz's voice. Ground in THIS document's neuroscience. Make it unique and valuable.

RETURN ONLY VALID JSON.
"""

            response = await self._structured_generation(
                prompt=prompt,
                response_model=BlogContent,
                context=context,
                suffix="founder_blog",
                temperature=0.85,  # Higher temp for more variation
                max_retries=3,
                use_cot_prompt=False,
                validation_context={'source_text': document_text},
            )

            return response or BlogContent()

        except Exception as e:
            logger.error(f"Blog content generation failed: {e}")
            return BlogContent()

    async def _generate_social_media_content(self, clinical_data: Dict[str, Any],
                                           context: Dict[str, Any]) -> SocialMediaContent:
        """Generate social media content that builds community."""
        try:
            document_text = context.get("document_text", "")
            document_id = context.get("document_id", "unknown")
            summary = self._summarize_research(document_text, max_chars=2000)
            
            # Load personas
            personas_context = self._load_personas_context()
            
            prompt = f"""
You are Liz Wooten creating UNIQUE social media content for document {document_id}.

RESEARCH FROM THIS SPECIFIC DOCUMENT:
{summary}

CLINICAL DATA FROM THIS DOCUMENT:
{clinical_data}

REAL CLIENT PROFILES:
{personas_context}

CRITICAL RULES:
1. Content must be UNIQUE to THIS document's research
2. Reference specific findings from THIS document
3. Use different client scenarios from personas
4. NO generic templates - vary language and tone
5. Each post idea must be distinct and research-specific

Generate 5-10 UNIQUE STRINGS (NOT objects) for each field:

post_ideas: ["Post idea based on THIS research...", ...]
captions: ["Caption tied to THIS research...", ...]
quotes: ["Liz's quote about THIS research...", ...]
hashtags: ["#HashtagSet1 #HashtagSet2", ...]
story_ideas: ["Story demonstrating THIS research...", ...]
reel_ideas: ["Reel explaining THIS research...", ...]
carousel_content: ["Slide 1: Content from THIS research...", "Slide 2: More content...", ...]
poll_questions: ["Poll question about THIS research? A) Option B) Option...", ...]

CRITICAL: Each item must be a SINGLE STRING, not an object with title/description fields.
Write as Liz - conversational, direct, caring. Ground in THIS document's neuroscience.

RETURN ONLY VALID JSON with string arrays.
"""

            response = await self._structured_generation(
                prompt=prompt,
                response_model=SocialMediaContent,
                context=context,
                suffix="founder_social",
                temperature=0.85,  # Higher creativity for social media
                max_retries=3,
                use_cot_prompt=False,
            )

            return response or SocialMediaContent()

        except Exception as e:
            logger.error(f"Social media content generation failed: {e}")
            return SocialMediaContent()

    async def _generate_content_ideas(self, clinical_data: Dict[str, Any],
                                    context: Dict[str, Any]) -> ContentCreationIdeas:
        """Generate content creation ideas for ongoing marketing."""
        try:
            # Get research summary for THIS specific document
            research_summary = context.get('research_summary', '')[:2500]
            document_id = context.get('document_id', 'unknown')
            
            # Get curated context
            curated_context = context.get('curated_context', {})
            personas_context = curated_context.get('personas_context', '')[:1000]
            
            prompt = f"""
You are Liz Wooten brainstorming UNIQUE content ideas based on THIS SPECIFIC research paper.

DOCUMENT ID: {document_id}

RESEARCH SUMMARY (THIS PAPER ONLY):
{research_summary}

SELECTED CLIENT PERSONAS:
{personas_context}

YOUR TASK:
Generate content ideas that are 100% SPECIFIC to THIS research paper's findings.
DO NOT use generic Enlitens topics. DO NOT repeat ideas from other papers.
EVERY idea must reference THIS paper's unique findings, methods, or implications.

Content Strategy:
- Connect THIS research's findings to client personas
- Challenge traditional therapy narratives using THIS paper's evidence
- Showcase THIS research as neuroscience-backed solution
- Create shareable content about THIS specific study

CRITICAL REQUIREMENTS:
1. EVERY topic/idea MUST reference THIS paper's specific findings
2. NO generic "ADHD" or "anxiety" topics unless THIS paper discusses them
3. NO repetition of standard Enlitens topics
4. Focus on what makes THIS research unique and newsworthy
5. Generate 10 COMPLETELY DIFFERENT ideas for each category

RETURN ONLY VALID JSON. NO markdown, NO headers, NO formatting.

{{
  "topic_ideas": [10 unique strings based on THIS research],
  "angle_ideas": [10 unique strings based on THIS research],
  "hook_ideas": [10 unique strings based on THIS research],
  "series_ideas": [10 unique strings based on THIS research],
  "collaboration_ideas": [10 unique strings based on THIS research],
  "trend_ideas": [10 unique strings based on THIS research],
  "seasonal_ideas": [10 unique strings based on THIS research]
}}

REMEMBER: If you generate ANY generic topic not tied to THIS paper, you have FAILED.
"""

            response = await self._structured_generation(
                prompt=prompt,
                response_model=ContentCreationIdeas,
                context=context,
                suffix="founder_ideas",
                temperature=0.95,  # Increased for more variation
                max_retries=3,
                use_cot_prompt=False,
            )

            return response or ContentCreationIdeas()

        except Exception as e:
            logger.error(f"Content ideas generation failed: {e}")
            return ContentCreationIdeas()

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate founder voice output."""
        required_keys = ["marketing_content", "seo_content", "website_copy",
                        "blog_content", "social_media_content", "content_creation_ideas"]
        return all(key in output for key in required_keys)

    async def cleanup(self):
        """Clean up founder voice agent."""
        if self.ollama_client:
            await self.ollama_client.cleanup()
        self.is_initialized = False
        logger.info("✅ Founder Voice Agent cleaned up")
