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
            model="qwen2.5-32b-instruct-q4_k_m"
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
            summary = self._summarize_research(document_text)
            prompt = f"""
You are Liz Wooten, founder of Enlitens, speaking directly to frustrated St. Louis clients.

# VOICE GUARDRAILS
- Direct and rebellious; never corporate.
- Honor the brain isn't broken; it adapts.
- Ground every promise in neuroscience or lived experience.
- Assume audience craves hope plus proof.

# RESEARCH SNAPSHOT (trimmed)
{summary}

# CLIENT CHALLENGES (top of mind)
{self.client_challenges}

# TASK
Craft bold marketing assets that turn this research into action.
- 3-5 headlines (≤18 words).
- 3-5 taglines with rebel flair.
- 3-5 value propositions tying neuroscience to relief.
- 3-5 benefits mixing emotional relief + tangible wins.
- 3-5 pain points echoing client language.

NOTE: Do NOT generate social proof, testimonials, credentials, or practice statistics (FTC violation).
Only reference research findings from the provided context.

Avoid repetition. Keep copy punchy; no bullets/numbers in strings.
Respond with JSON matching MarketingContent schema (NO social_proof field).
"""

            llama_client = self.ollama_client.clone_with_model("llama3.1:8b")

            raw_notes = await llama_client.generate_text(
                prompt=prompt,
                temperature=0.6,  # LOWERED from 0.75: Research optimal for creative
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
            summary = self._summarize_research(document_text, max_chars=1500)

            prompt = f"""
You are Liz Wooten writing website copy that converts St. Louis visitors into clients.
Your website needs to speak directly to people who've tried traditional therapy and want something different.

RESEARCH CONTEXT:
{summary}

CLINICAL INSIGHTS:
{clinical_data}

Website Goals:
- Convert visitors who are frustrated with traditional approaches
- Show immediate understanding of their struggles
- Present neuroscience as the clear solution
- Build trust through authenticity and expertise
- Address objections and skepticism

St. Louis Context:
- Clients often feel misunderstood by traditional providers
- Many have treatment-resistant conditions
- They want practical help, not just "coping skills"
- Skeptical of "one size fits all" approaches

CRITICAL: You MUST return ONLY valid JSON. NO markdown, NO headers, NO formatting.

Generate 3-8 paragraph-length strings for each field. Each string should be a complete, substantial piece of content.

{{
  "about_sections": [
    "I started Enlitens after watching countless clients struggle through years of traditional therapy without real change. The missing piece wasn't more talk therapy—it was understanding how their brains actually work. Now I combine neuroscience assessment with targeted interventions that create lasting change, not just coping skills.",
    "Another about paragraph...",
    ...
  ],
  "feature_descriptions": [
    "Bottom-Up Neuroscience Assessment: Unlike traditional therapy questionnaires, we measure your actual nervous system responses through interoceptive awareness evaluation, sensory processing analysis, and executive function testing to understand how your brain uniquely processes the world.",
    ...
  ],
  "benefit_statements": [
    "Reduce anxiety attacks by understanding and regulating your nervous system responses rather than just managing symptoms.",
    ...
  ],
  "faq_content": [
    "Q: How is this different from regular therapy? A: Traditional therapy focuses on thoughts and behaviors. We start with your neurobiology—how your brain actually processes information—and build interventions that work with your unique wiring, not against it.",
    ...
  ],
  "service_descriptions": [
    "ADHD Executive Function Support: Comprehensive neuroscience-based assessment of working memory, attention systems, and dopamine regulation, followed by targeted interventions including environmental modifications, cognitive strategies, and nervous system regulation techniques tailored to your specific executive function profile.",
    ...
  ]
}}

NOTE: Testimonials field REMOVED for FTC compliance (16 CFR Part 465 - no AI-generated testimonials).

Use Liz's direct, authentic voice. Ground everything in neuroscience. Address real St. Louis client pain points.

RETURN ONLY THE JSON OBJECT. NO OTHER TEXT.
"""

            response = await self._structured_generation(
                prompt=prompt,
                response_model=WebsiteCopy,
                context=context,
                suffix="founder_website",
                temperature=0.6,  # Research optimal for creative content (lowered from 0.7)
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
            summary = self._summarize_research(document_text, max_chars=1500)

            prompt = f"""
You are Liz Wooten writing blog content that positions Enlitens as the neuroscience therapy leader in St. Louis.
Your blog should educate, challenge traditional approaches, and drive inquiries.

RESEARCH CONTEXT:
{summary}

CLINICAL INSIGHTS:
{clinical_data}

Blog Strategy:
- Establish expertise in neuroscience-based therapy
- Challenge traditional mental health narratives
- Provide immediate value to readers
- Drive calls-to-action naturally
- Target specific St. Louis mental health concerns

Content Focus:
- ADHD and executive function challenges
- Anxiety and emotional regulation
- Trauma and PTSD recovery
- Autism and neurodiversity
- Treatment-resistant conditions

CRITICAL: You MUST return ONLY valid JSON. NO markdown, NO headers, NO formatting.

Generate 5-10 strings for each field below:

{{
  "article_ideas": [
    "Why Your ADHD Meds Stopped Working (And What Neuroscience Says to Do Next)",
    "The Real Reason Your Anxiety Spikes at Night (Brain Science Explains)",
    ...
  ],
  "blog_outlines": [
    "Intro explaining ADHD executive function challenges, three evidence-based strategies (working memory support, dopamine regulation, environmental modifications), practical implementation steps, conclusion with next actions.",
    ...
  ],
  "talking_points": [
    "Your brain isn't broken—it's responding to an environment it wasn't designed for.",
    "Neuroplasticity means change is always possible, no matter your age or diagnosis.",
    ...
  ],
  "expert_quotes": [
    "I started Enlitens because I was tired of watching brilliant, capable people told they were 'broken' when their brains just work differently.",
    ...
  ],
  "statistics": [
    {{"claim": "According to Smith et al. (2023), 67% of participants showed improved executive function after 8 weeks of neurofeedback training", "citation": {{"quote": "Sixty-seven percent of participants demonstrated statistically significant improvements in executive function measures", "source_id": "doc_id", "source_title": "Study Name", "page_or_section": "pg 12"}}}},
    ...
  ],
  "case_studies": [
    "[HYPOTHETICAL EXAMPLE] Client with treatment-resistant anxiety learned their symptoms were sensory processing differences. Using interoceptive awareness training and bottom-up regulation, they reduced panic attacks by 80% in 8 weeks.",
    ...
  ],
  "how_to_guides": [
    "5-Minute Brain Reset for Anxiety: Quick bilateral stimulation exercise using alternate nostril breathing to activate parasympathetic nervous system, reducing cortisol and restoring executive function.",
    ...
  ],
  "myth_busting": [
    "Myth: ADHD is caused by poor parenting. Reality: Neuroscience shows ADHD involves differences in dopamine regulation and prefrontal cortex development. What This Means: Parents aren't to blame; brains just work differently and need different support strategies.",
    ...
  ]
}}

CRITICAL STATISTICS RULES:
- Statistics MUST include proper citations with exact quotes from research papers
- Format: {{"claim": "According to [Author] ([Year]), [statistic]", "citation": {{"quote": "exact quote", "source_id": "doc_id", ...}}}}
- NEVER generate Enlitens practice statistics or client outcome data
- Only cite published research findings from provided documents

CASE STUDY RULES:
- ALL case studies must be marked as "[HYPOTHETICAL EXAMPLE]"
- NO real client names, testimonials, or specific practice data
- Base examples on research findings, not fabricated scenarios

Use Liz's rebellious, direct voice. Ground everything in neuroscience. Make it shareable and valuable.

RETURN ONLY THE JSON OBJECT. NO OTHER TEXT.
"""

            response = await self._structured_generation(
                prompt=prompt,
                response_model=BlogContent,
                context=context,
                suffix="founder_blog",
                temperature=0.6,  # LOWERED from 0.65: Research optimal for creative content
                max_retries=3,
                use_cot_prompt=False,
                validation_context={'source_text': document_text},  # CRITICAL: Enable citation verification
            )

            return response or BlogContent()

        except Exception as e:
            logger.error(f"Blog content generation failed: {e}")
            return BlogContent()

    async def _generate_social_media_content(self, clinical_data: Dict[str, Any],
                                           context: Dict[str, Any]) -> SocialMediaContent:
        """Generate social media content that builds community."""
        try:
            prompt = f"""
You are Liz Wooten creating social media content that builds Enlitens' community in St. Louis.
Your social media should feel like a conversation with a trusted friend who really gets it.

Social Media Goals:
- Build trust and authenticity
- Show understanding of daily struggles
- Share hope through neuroscience
- Drive engagement and inquiries
- Position as the "rebel" in mental health

Clinical Data:
{clinical_data}

Content Style:
- Short, punchy, and relatable
- Mix of education and inspiration
- Questions that encourage comments
- Real talk about real challenges
- Hopeful without being cheesy

CRITICAL: You MUST return ONLY valid JSON. NO markdown, NO headers, NO formatting.

Generate 5-10 strings for each field below:

{{
  "post_ideas": [
    "Share your morning brain fog story—explain it's not laziness, it's your prefrontal cortex taking 90 minutes to fully wake up. Include 3 neuroscience-based wake-up strategies.",
    ...
  ],
  "captions": [
    "Your ADHD brain isn't broken. It's wired for novelty-seeking and responds differently to dopamine. That's not a disorder—that's a different operating system. Learn to work WITH your brain, not against it.",
    ...
  ],
  "quotes": [
    "Traditional therapy asks 'what's wrong with you?' Neuroscience asks 'what happened TO you, and how did your brain adapt?'",
    ...
  ],
  "hashtags": [
    "#NeuroscienceTherapy #STLTherapist #ADHDSupport #TraumaInformed #BrainBasedHealing #StLouisMentalHealth",
    ...
  ],
  "story_ideas": [
    "Film morning routine showing sensory regulation techniques—demonstrate bilateral stimulation, explain vagus nerve activation in voiceover.",
    ...
  ],
  "reel_ideas": [
    "30-second explanation of why your anxiety spikes at bedtime—show how cortisol rhythm works, demonstrate 3-minute reset technique.",
    ...
  ],
  "carousel_content": [
    "5 signs your anxiety is actually sensory processing differences: 1-overwhelmed by noise 2-need alone time to recharge 3-difficulty in crowds 4-sensitive to textures 5-emotional after busy days.",
    ...
  ],
  "poll_questions": [
    "What time of day is your ADHD brain sharpest? A) Morning (7-10am) B) Midday (10am-2pm) C) Afternoon (2-6pm) D) Evening/Night (after 6pm)",
    ...
  ]
}}

Write as Liz - conversational, direct, and caring. Balance education with empathy.

RETURN ONLY THE JSON OBJECT. NO OTHER TEXT.
"""

            response = await self._structured_generation(
                prompt=prompt,
                response_model=SocialMediaContent,
                context=context,
                suffix="founder_social",
                temperature=0.6,  # Higher creativity for social media
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
            prompt = f"""
You are Liz Wooten brainstorming content ideas that will establish Enlitens as St. Louis's neuroscience therapy leader.
Your content should drive inquiries while building long-term trust and authority.

Content Strategy:
- Address real pain points from client intakes
- Challenge traditional therapy narratives
- Showcase neuroscience as the solution
- Build email list and inquiries
- Create shareable, valuable content

Clinical Data:
{clinical_data}

Content Goals:
- Weekly blog posts and social media
- Monthly webinars and workshops
- Quarterly educational events
- Ongoing email nurturing sequences

CRITICAL: You MUST return ONLY valid JSON. NO markdown, NO headers, NO formatting.

Generate 5-10 strings for each field below:

{{
  "topic_ideas": [
    "How Neuroscience Explains Why Your Anxiety Gets Worse at Night",
    "The ADHD-Trauma Connection Most Therapists Miss",
    ...
  ],
  "angle_ideas": [
    "Your brain isn't broken—it's responding to an environment it wasn't designed for",
    "Why medication alone won't fix executive dysfunction (and what will)",
    ...
  ],
  "hook_ideas": [
    "If you've tried 5+ therapists and nothing worked, this is why...",
    "Your ADHD diagnosis might be missing the real problem",
    ...
  ],
  "series_ideas": [
    "The Neuroscience of Self-Regulation: 4-part series on brain-body connection",
    "Breaking the ADHD Cycle: Weekly tips for executive function",
    ...
  ],
  "collaboration_ideas": [
    "Partner with Washington University on epigenetics webinar",
    "Co-host workshop with St. Louis Parks on nature and neuroplasticity",
    ...
  ],
  "trend_ideas": [
    "Back-to-school stress and brain development",
    "Post-pandemic mental health and neural networks",
    ...
  ],
  "seasonal_ideas": [
    "Spring allergies and brain inflammation connection",
    "Summer outdoor activities for nervous system regulation",
    ...
  ]
}}

Focus on content that directly addresses ADHD, anxiety, trauma challenges and drives inquiries.

RETURN ONLY THE JSON OBJECT. NO OTHER TEXT.
"""

            response = await self._structured_generation(
                prompt=prompt,
                response_model=ContentCreationIdeas,
                context=context,
                suffix="founder_ideas",
                temperature=0.6,
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
