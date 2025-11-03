"""
Enhanced Complete Enlitens Agent with advanced prompt engineering and schema enforcement.

This agent uses sophisticated prompt engineering, JSON repair, and schema enforcement
to generate high-quality, structured content for the Enlitens knowledge base.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from src.synthesis.ollama_client import OllamaClient
from src.models.enlitens_schemas import (
    EnlitensKnowledgeEntry, DocumentMetadata, ExtractedEntities,
    RebellionFramework, MarketingContent, SEOContent, WebsiteCopy,
    BlogContent, SocialMediaContent, EducationalContent, ClinicalContent,
    ResearchContent, ContentCreationIdeas
)
from src.extraction.enhanced_extraction_tools import EnhancedExtractionTools
from src.monitoring.error_telemetry import TelemetrySeverity, log_with_telemetry

logger = logging.getLogger(__name__)
TELEMETRY_AGENT = "enhanced_complete_enlitens_agent"

class EnhancedCompleteEnlitensAgent:
    """
    Enhanced Complete Enlitens Agent with advanced prompt engineering and schema enforcement.
    """
    
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.extraction_tools = EnhancedExtractionTools()
        logger.info("Enhanced Complete Enlitens Agent initialized")

    async def extract_complete_content(self, text: str, document_id: str, 
                                     client_insights: Optional[Dict[str, Any]] = None,
                                     founder_insights: Optional[Dict[str, Any]] = None) -> EnlitensKnowledgeEntry:
        """
        Extract complete content using enhanced prompts and schema enforcement.
        
        Args:
            text: Research paper text
            document_id: Unique document identifier
            client_insights: Optional client pain point analysis
            founder_insights: Optional founder voice pattern analysis
            
        Returns:
            Complete EnlitensKnowledgeEntry with all content types
        """
        try:
            logger.info(f"Starting enhanced content extraction for document: {document_id}")
            
            # Create metadata
            metadata = DocumentMetadata(
                document_id=document_id,
                filename=f"{document_id}.pdf",
                processing_timestamp=datetime.now()
            )
            
            # Extract entities using enhanced tools
            entities = await self._extract_entities(text)
            
            client_payload: Dict[str, Any] = dict(client_insights or {})
            founder_payload: Dict[str, Any] = dict(founder_insights or {})

            raw_client_context = client_payload.get("raw_context")
            if not raw_client_context:
                enhanced_client = client_payload.get("enhanced_analysis")
                if isinstance(enhanced_client, dict):
                    raw_client_context = enhanced_client.get("raw_content")
            if raw_client_context:
                client_payload["raw_context"] = raw_client_context

            raw_founder_context = founder_payload.get("raw_context")
            if not raw_founder_context:
                enhanced_founder = founder_payload.get("enhanced_analysis")
                if isinstance(enhanced_founder, dict):
                    raw_founder_context = enhanced_founder.get("raw_content")
            if raw_founder_context:
                founder_payload["raw_context"] = raw_founder_context

            if raw_client_context and not client_payload.get("pain_point_keywords"):
                try:
                    fallback_keywords = [
                        keyword
                        for keyword, _ in self.extraction_tools.extract_semantic_keywords(
                            raw_client_context,
                            keyphrase_ngram_range=(1, 3),
                            top_n=5,
                        )
                    ]
                    if fallback_keywords:
                        client_payload["pain_point_keywords"] = fallback_keywords
                except Exception as exc:
                    log_with_telemetry(
                        logger.warning,
                        "Failed to derive client pain point keywords from raw context: %s",
                        exc,
                        agent=TELEMETRY_AGENT,
                        severity=TelemetrySeverity.MINOR,
                        impact="Client keyword fallback failed",
                        doc_id=document_id,
                        details={"error": str(exc)},
                    )

            if raw_founder_context and not founder_payload.get("founder_keywords"):
                try:
                    fallback_keywords = [
                        keyword
                        for keyword, _ in self.extraction_tools.extract_semantic_keywords(
                            raw_founder_context,
                            keyphrase_ngram_range=(1, 3),
                            top_n=8,
                        )
                    ]
                    if fallback_keywords:
                        founder_payload["founder_keywords"] = fallback_keywords
                except Exception as exc:
                    log_with_telemetry(
                        logger.warning,
                        "Failed to derive founder keywords from raw context: %s",
                        exc,
                        agent=TELEMETRY_AGENT,
                        severity=TelemetrySeverity.MINOR,
                        impact="Founder keyword fallback failed",
                        doc_id=document_id,
                        details={"error": str(exc)},
                    )

            content_insights: Dict[str, Any] = {}
            if client_payload or founder_payload:
                content_insights = self.extraction_tools.generate_content_insights(
                    text, client_payload, founder_payload
                )
            
            # Extract all content types with enhanced prompts
            rebellion_content = await self._extract_rebellion_framework(text, content_insights)
            marketing_content = await self._extract_marketing_content(text, content_insights)
            seo_content = await self._extract_seo_content(text, content_insights)
            website_copy = await self._extract_website_copy(text, content_insights)
            blog_content = await self._extract_blog_content(text, content_insights)
            social_media_content = await self._extract_social_media_content(text, content_insights)
            educational_content = await self._extract_educational_content(text, content_insights)
            clinical_content = await self._extract_clinical_content(text, content_insights)
            research_content = await self._extract_research_content(text, content_insights)
            content_creation_ideas = await self._extract_content_creation_ideas(text, content_insights)
            
            # Create complete knowledge entry
            knowledge_entry = EnlitensKnowledgeEntry(
                metadata=metadata,
                extracted_entities=entities,
                rebellion_framework=rebellion_content,
                marketing_content=marketing_content,
                seo_content=seo_content,
                website_copy=website_copy,
                blog_content=blog_content,
                social_media_content=social_media_content,
                educational_content=educational_content,
                clinical_content=clinical_content,
                research_content=research_content,
                content_creation_ideas=content_creation_ideas
            )
            
            logger.info(f"Successfully extracted complete content for document: {document_id}")
            return knowledge_entry
            
        except Exception as e:
            log_with_telemetry(
                logger.error,
                "Error extracting complete content for %s: %s",
                document_id,
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.CRITICAL,
                impact="Complete content extraction failed",
                doc_id=document_id,
                details={"error": str(e)},
            )
            raise

    async def _extract_entities(self, text: str) -> ExtractedEntities:
        """Extract entities using enhanced tools."""
        try:
            # Extract semantic keywords
            keywords = self.extraction_tools.extract_semantic_keywords(
                text, keyphrase_ngram_range=(1, 3), top_n=20
            )
            
            # Categorize keywords (simplified approach)
            biomedical_entities = []
            neuroscience_entities = []
            clinical_entities = []
            statistical_entities = []
            
            for keyword, score in keywords:
                keyword_lower = keyword.lower()
                if any(term in keyword_lower for term in ['brain', 'neural', 'synaptic', 'neuro']):
                    neuroscience_entities.append(keyword)
                elif any(term in keyword_lower for term in ['therapy', 'treatment', 'intervention', 'clinical']):
                    clinical_entities.append(keyword)
                elif any(term in keyword_lower for term in ['p <', 'effect size', 'correlation', 'significant']):
                    statistical_entities.append(keyword)
                else:
                    biomedical_entities.append(keyword)
            
            return ExtractedEntities(
                biomedical_entities=biomedical_entities,
                neuroscience_entities=neuroscience_entities,
                clinical_entities=clinical_entities,
                statistical_entities=statistical_entities,
                total_entities=len(keywords)
            )
            
        except Exception as e:
            log_with_telemetry(
                logger.error,
                "Error extracting entities: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Entity extraction failed",
                details={"error": str(e)},
            )
            return ExtractedEntities()

    async def _extract_rebellion_framework(self, text: str, content_insights: Dict[str, Any]) -> RebellionFramework:
        """Extract rebellion framework content with enhanced prompts."""
        prompt = f"""
# ROLE
You are an expert in the Enlitens Rebellion Framework, a revolutionary approach to mental health that challenges traditional clinical practices through neuroscience-based care.

# CONTEXT
Enlitens is building a rebellion against the mental health system by:
- Replacing shame with science (neurobiological truth)
- Deconstructing systemic failures in traditional clinical practices
- Empowering clients through strengths-based, neurodiversity-affirming approaches
- Using cutting-edge neuroscience research to inform treatment

# PURPOSE
Extract content that supports our rebellion against traditional mental health approaches and empowers clients through neurobiological understanding.

# RESEARCH TEXT
{text[:3000]}

# CONTENT INSIGHTS
{json.dumps(content_insights, indent=2) if content_insights else "No additional insights available"}

# INSTRUCTIONS
Analyze the research text and extract content that aligns with our rebellion framework:

1. **Narrative Deconstruction**: Identify how this research challenges conventional mental health narratives
2. **Sensory Profiling**: Extract insights about sensory processing and interoception
3. **Executive Function**: Find research about cognitive control and regulation
4. **Social Processing**: Identify social cognition and relationship insights
5. **Strengths Synthesis**: Find research that highlights neurodiversity as strength
6. **Rebellion Themes**: Extract themes that challenge traditional approaches
7. **Aha Moments**: Identify insights that could create powerful "aha!" moments for clients

Focus on content that:
- Challenges conventional wisdom about mental health
- Emphasizes neurobiological truth over pathology
- Empowers clients through scientific understanding
- Supports strengths-based approaches
- Deconstructs traditional mental health narratives

# OUTPUT FORMAT
Provide your analysis as a JSON object with the specified schema. 

CRITICAL: Each field must be a simple list of strings, not nested objects. For example:
- "narrative_deconstruction": ["insight 1", "insight 2", "insight 3"]
- NOT: "narrative_deconstruction": {"content": ["insight 1", "insight 2"]}

EXAMPLE OUTPUT FORMAT:
{
  "narrative_deconstruction": ["Traditional therapy focuses on symptoms rather than neurobiology", "Research challenges the pathology model"],
  "sensory_profiling": ["Interoceptive awareness training", "Sensory processing insights"],
  "executive_function": ["Prefrontal cortex regulation", "Cognitive control strategies"],
  "social_processing": ["Social cognition research", "Relationship insights"],
  "strengths_synthesis": ["Neurodiversity as strength", "Individual differences as assets"],
  "rebellion_themes": ["Science over shame", "Neurobiological truth"],
  "aha_moments": ["Your brain isn't broken", "Neuroplasticity insights"]
}

Ensure all fields are present and properly formatted as simple lists.
"""
        
        try:
            result = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=RebellionFramework,
                temperature=0.7
            )
            return result if result else RebellionFramework()
        except Exception as e:
            log_with_telemetry(
                logger.error,
                "Error extracting rebellion framework: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Rebellion framework extraction failed",
                details={"error": str(e)},
            )
            return RebellionFramework()

    async def _extract_marketing_content(self, text: str, content_insights: Dict[str, Any]) -> MarketingContent:
        """Extract marketing content with enhanced prompts."""
        prompt = f"""
# ROLE
You are a marketing strategist for Enlitens, a revolutionary mental health practice that challenges traditional clinical approaches through neuroscience-based care.

# CONTEXT
Enlitens is building a rebellion against the mental health system by:
- Replacing shame with science (neurobiological truth)
- Deconstructing systemic failures in traditional clinical practices
- Empowering clients through strengths-based, neurodiversity-affirming approaches
- Using cutting-edge neuroscience research to inform treatment

# PURPOSE
Craft high-impact marketing assets using the research insights, St. Louis context, and our rebellion framework.

# RESEARCH HIGHLIGHTS (trimmed to 3K characters)
{self._summarize_research(text)}

# ST. LOUIS AUDIENCE INSIGHTS
- Top Client Frustrations: long waitlists, surface-level talk therapy, lack of neuroscience.
- Desired Outcomes: faster relief, science-backed care, therapists who “get” their lived experience.
- Voice Style: bold, rebellious, data-grounded, hopeful.

# OUTPUT SPECIFICATIONS
Produce thoughtful, non-generic copy with vivid language and concrete evidence.
- Each list must contain 3 to 5 items.
- Keep items under 18 words; punchy, plain-English phrases.
- Avoid duplicate wording across items.
- Reference neuroscience or rebel positioning wherever natural.

Return JSON with fields:
- headlines: electrifying headlines.
- taglines: short slogans.
- value_propositions: clear differentiators rooted in neuroscience.
- benefits: client outcomes framed in emotional + practical terms.
- pain_points: pains we solve, echoing client voice.
- social_proof: mix of stats, testimonials, media mentions.

Polish everything to sound like a confident rebel strategist speaking to frustrated but hopeful clients.
"""
        
        try:
            logger.debug("Marketing content input length=%d characters", len(text))
            llama_client = self.ollama_client.clone_with_model("llama3.1:8b")

            # Always generate rich notes first, then structure them
            raw_notes = await llama_client.generate_text(
                prompt=prompt,
                temperature=0.8,
                num_predict=1024
            )
            logger.debug(
                "Fallback marketing notes length=%d sample=%s",
                len(raw_notes),
                raw_notes[:500]
            )

            normalized_notes = self._normalize_marketing_notes(raw_notes)

            structuring_prompt = f"""
You are a JSON formatter. Convert the following marketing notes into valid JSON with schema:
{{{{
  "headlines": [string],
  "taglines": [string],
  "value_propositions": [string],
  "benefits": [string],
  "pain_points": [string],
  "social_proof": [string]
}}}}

Rules:
- Each list must have 3-5 plain strings (no inner objects, no bullets).
- If you see patterns like "statistic: ..." or "testimonial: ...", convert them into concise sentences.
- Remove labels such as "statistic:"/"quote:" and return polished standalone lines.

CONTENT
---
{normalized_notes}
---
Return JSON only.
"""
            try:
                structured = await self.ollama_client.generate_structured_response(
                    prompt=structuring_prompt,
                    response_model=MarketingContent,
                    temperature=0.2,
                    max_retries=3
                )
            except Exception as err:
                log_with_telemetry(
                    logger.error,
                    "Structured marketing formatting failed: %s",
                    err,
                    agent=TELEMETRY_AGENT,
                    severity=TelemetrySeverity.MAJOR,
                    impact="Marketing structuring failed",
                    details={"error": str(err)},
                )
                structured = None

            if structured:
                logger.debug(
                    "Structured fallback marketing content: %s",
                    {k: structured.model_dump().get(k) for k in ("headlines", "value_propositions", "benefits")}
                )
                return structured

            log_with_telemetry(
                logger.warning,
                "Marketing content structuring failed; returning empty model",
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MINOR,
                impact="Marketing content fallback",
            )
            return MarketingContent()
        except Exception as e:
            log_with_telemetry(
                logger.error,
                "Error extracting marketing content: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Marketing content extraction failed",
                details={"error": str(e)},
            )
            return MarketingContent()

    async def _extract_seo_content(self, text: str, content_insights: Dict[str, Any]) -> SEOContent:
        """Extract SEO content with enhanced prompts."""
        prompt = f"""
# ROLE
You are an SEO strategist for Enlitens, a revolutionary mental health practice that challenges traditional clinical approaches through neuroscience-based care.

# CONTEXT
Enlitens is building a rebellion against the mental health system by:
- Replacing shame with science (neurobiological truth)
- Deconstructing systemic failures in traditional clinical practices
- Empowering clients through strengths-based, neurodiversity-affirming approaches
- Using cutting-edge neuroscience research to inform treatment

# PURPOSE
Extract SEO content that will help us:
- Rank for keywords that people use when searching for alternatives to traditional therapy
- Attract clients who are frustrated with conventional mental health approaches
- Build authority in neuroscience-based mental health care
- Create content that challenges mainstream mental health narratives
- Target people searching for "neurodiversity," "strengths-based therapy," "neuroscience therapy"

# RESEARCH TEXT
{text[:3000]}

# CONTENT INSIGHTS
{json.dumps(content_insights, indent=2) if content_insights else "No additional insights available"}

# INSTRUCTIONS
Extract SEO content that helps us reach people who are:
- Frustrated with traditional therapy approaches
- Interested in neuroscience-based mental health care
- Looking for neurodiversity-affirming practices
- Seeking evidence-based alternatives to conventional treatment
- Interested in understanding the neurobiological basis of mental health

Focus on keywords and topics that align with our rebellion against traditional mental health approaches.

# OUTPUT FORMAT
Provide your analysis as a JSON object with the specified schema. 

CRITICAL: Each field must be a simple list of strings, not nested objects. For example:
- "headlines": ["headline 1", "headline 2", "headline 3"]
- NOT: "headlines": {"content": ["headline 1", "headline 2"]}

Ensure all fields are present and properly formatted as simple lists.
"""
        
        try:
            result = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=SEOContent,
                temperature=0.7
            )
            return result if result else SEOContent()
        except Exception as e:
            log_with_telemetry(
                logger.error,
                "Error extracting SEO content: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="SEO content extraction failed",
                details={"error": str(e)},
            )
            return SEOContent()

    async def _extract_website_copy(self, text: str, content_insights: Dict[str, Any]) -> WebsiteCopy:
        """Extract website copy with enhanced prompts."""
        prompt = f"""
# ROLE
You are a copywriter for Enlitens, a revolutionary mental health practice that challenges traditional clinical approaches through neuroscience-based care.

# CONTEXT
Enlitens is building a rebellion against the mental health system by:
- Replacing shame with science (neurobiological truth)
- Deconstructing systemic failures in traditional clinical practices
- Empowering clients through strengths-based, neurodiversity-affirming approaches
- Using cutting-edge neuroscience research to inform treatment

# PURPOSE
Extract website copy that will help us:
- Convert visitors who are frustrated with traditional mental health approaches
- Clearly communicate our unique, science-backed approach
- Build trust through evidence-based messaging
- Create compelling content for our website pages (About, Services, FAQ, etc.)
- Position Enlitens as the alternative to outdated mental health practices

# RESEARCH TEXT
{text[:3000]}

# CONTENT INSIGHTS
{json.dumps(content_insights, indent=2) if content_insights else "No additional insights available"}

# INSTRUCTIONS
Extract website copy that:
- Challenges conventional wisdom about mental health
- Emphasizes our neuroscience-based approach
- Speaks to people who feel failed by traditional therapy
- Builds trust through scientific evidence
- Clearly explains how we're different from typical mental health practices
- Addresses common concerns and objections

Focus on creating copy that converts visitors into clients by showing them there's a better way.

# OUTPUT FORMAT
Provide your analysis as a JSON object with the specified schema. 

CRITICAL: Each field must be a simple list of strings, not nested objects. For example:
- "headlines": ["headline 1", "headline 2", "headline 3"]
- NOT: "headlines": {"content": ["headline 1", "headline 2"]}

Ensure all fields are present and properly formatted as simple lists.
"""
        
        try:
            result = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=WebsiteCopy,
                temperature=0.7
            )
            return result if result else WebsiteCopy()
        except Exception as e:
            log_with_telemetry(
                logger.error,
                "Error extracting website copy: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Website copy extraction failed",
                details={"error": str(e)},
            )
            return WebsiteCopy()

    async def _extract_blog_content(self, text: str, content_insights: Dict[str, Any]) -> BlogContent:
        """Extract blog content with enhanced prompts."""
        prompt = f"""
# ROLE
You are a content strategist for Enlitens, a revolutionary mental health practice that challenges traditional clinical approaches through neuroscience-based care.

# CONTEXT
Enlitens is building a rebellion against the mental health system by:
- Replacing shame with science (neurobiological truth)
- Deconstructing systemic failures in traditional clinical practices
- Empowering clients through strengths-based, neurodiversity-affirming approaches
- Using cutting-edge neuroscience research to inform treatment

# PURPOSE
Extract blog content that will help us:
- Educate people about the neurobiological basis of mental health
- Challenge conventional wisdom about mental health and therapy
- Build authority as neuroscience-based mental health experts
- Attract clients who are frustrated with traditional approaches
- Create shareable content that positions us as thought leaders
- Generate content for our blog, social media, and educational materials

# RESEARCH TEXT
{text[:3000]}

# CONTENT INSIGHTS
{json.dumps(content_insights, indent=2) if content_insights else "No additional insights available"}

# INSTRUCTIONS
Extract blog content that:
- Educates readers about neuroscience and mental health
- Challenges mainstream mental health narratives
- Provides evidence-based alternatives to conventional wisdom
- Helps people understand their neurobiology
- Builds trust through scientific evidence
- Creates "aha!" moments about mental health
- Debunks myths about mental health and therapy
- Shows how neuroscience research applies to real life

Focus on creating content that educates, challenges, and empowers readers.

# OUTPUT FORMAT
Provide your analysis as a JSON object with the specified schema. 

CRITICAL: Each field must be a simple list of strings, not nested objects. For example:
- "headlines": ["headline 1", "headline 2", "headline 3"]
- NOT: "headlines": {"content": ["headline 1", "headline 2"]}

Ensure all fields are present and properly formatted as simple lists.
"""
        
        try:
            result = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=BlogContent,
                temperature=0.7
            )
            return result if result else BlogContent()
        except Exception as e:
            log_with_telemetry(
                logger.error,
                "Error extracting blog content: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Blog content extraction failed",
                details={"error": str(e)},
            )
            return BlogContent()

    async def _extract_social_media_content(self, text: str, content_insights: Dict[str, Any]) -> SocialMediaContent:
        """Extract social media content with enhanced prompts."""
        prompt = f"""
# ROLE
You are a social media strategist for Enlitens, a revolutionary mental health practice that challenges traditional clinical approaches through neuroscience-based care.

# CONTEXT
Enlitens is building a rebellion against the mental health system by:
- Replacing shame with science (neurobiological truth)
- Deconstructing systemic failures in traditional clinical practices
- Empowering clients through strengths-based, neurodiversity-affirming approaches
- Using cutting-edge neuroscience research to inform treatment

# PURPOSE
Extract social media content that will help us:
- Build a community of people frustrated with traditional mental health approaches
- Educate followers about neuroscience and mental health
- Challenge conventional wisdom about mental health
- Build authority as neuroscience-based mental health experts
- Create engaging, shareable content that drives traffic to our website
- Attract clients who are looking for alternatives to traditional therapy

# RESEARCH TEXT
{text[:3000]}

# CONTENT INSIGHTS
{json.dumps(content_insights, indent=2) if content_insights else "No additional insights available"}

# INSTRUCTIONS
Extract social media content that:
- Educates followers about neurobiology and mental health
- Challenges mainstream mental health narratives
- Creates "aha!" moments about mental health
- Builds trust through scientific evidence
- Engages people who feel failed by traditional therapy
- Positions us as thought leaders in neuroscience-based mental health
- Creates shareable content that drives engagement
- Uses hashtags that reach our target audience

Focus on creating content that educates, challenges, and builds community.

# OUTPUT FORMAT
Provide your analysis as a JSON object with the specified schema. 

CRITICAL: Each field must be a simple list of strings, not nested objects. For example:
- "headlines": ["headline 1", "headline 2", "headline 3"]
- NOT: "headlines": {"content": ["headline 1", "headline 2"]}

Ensure all fields are present and properly formatted as simple lists.
"""
        
        try:
            result = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=SocialMediaContent,
                temperature=0.7
            )
            return result if result else SocialMediaContent()
        except Exception as e:
            log_with_telemetry(
                logger.error,
                "Error extracting social media content: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Social media content extraction failed",
                details={"error": str(e)},
            )
            return SocialMediaContent()

    async def _extract_educational_content(self, text: str, content_insights: Dict[str, Any]) -> EducationalContent:
        """Extract educational content with enhanced prompts."""
        prompt = f"""
# ROLE
You are an educational content creator for Enlitens, a revolutionary mental health practice that challenges traditional clinical approaches through neuroscience-based care.

# CONTEXT
Enlitens is building a rebellion against the mental health system by:
- Replacing shame with science (neurobiological truth)
- Deconstructing systemic failures in traditional clinical practices
- Empowering clients through strengths-based, neurodiversity-affirming approaches
- Using cutting-edge neuroscience research to inform treatment

# PURPOSE
Extract educational content that will help us:
- Educate clients about the neurobiological basis of their mental health
- Help people understand their neurobiology in accessible terms
- Create "aha!" moments that empower clients
- Build trust through scientific education
- Provide tools for clients to understand themselves better
- Create educational materials for our practice and website
- Help clients see their strengths and neurodiversity as assets

# RESEARCH TEXT
{text[:3000]}

# CONTENT INSIGHTS
{json.dumps(content_insights, indent=2) if content_insights else "No additional insights available"}

# INSTRUCTIONS
Extract educational content that:
- Explains complex neuroscience in simple, accessible terms
- Helps people understand their neurobiology
- Creates analogies that make neuroscience relatable
- Provides examples that people can relate to
- Defines terms in ways that empower rather than pathologize
- Shows processes that help people understand themselves
- Creates comparisons that challenge conventional wisdom
- Provides visual aids that make concepts clear
- Sets learning objectives that empower clients

Focus on creating educational content that empowers and educates rather than pathologizes.

# OUTPUT FORMAT
Provide your analysis as a JSON object with the specified schema. 

CRITICAL: Each field must be a simple list of strings, not nested objects. For example:
- "headlines": ["headline 1", "headline 2", "headline 3"]
- NOT: "headlines": {"content": ["headline 1", "headline 2"]}

Ensure all fields are present and properly formatted as simple lists.
"""
        
        try:
            result = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=EducationalContent,
                temperature=0.7
            )
            return result if result else EducationalContent()
        except Exception as e:
            log_with_telemetry(
                logger.error,
                "Error extracting educational content: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Educational content extraction failed",
                details={"error": str(e)},
            )
            return EducationalContent()

    async def _extract_clinical_content(self, text: str, content_insights: Dict[str, Any]) -> ClinicalContent:
        """Extract clinical content with enhanced prompts."""
        prompt = f"""
# ROLE
You are a clinical content specialist for Enlitens, a revolutionary mental health practice that challenges traditional clinical approaches through neuroscience-based care.

# CONTEXT
Enlitens is building a rebellion against the mental health system by:
- Replacing shame with science (neurobiological truth)
- Deconstructing systemic failures in traditional clinical practices
- Empowering clients through strengths-based, neurodiversity-affirming approaches
- Using cutting-edge neuroscience research to inform treatment

# PURPOSE
Extract clinical content that will help us:
- Develop evidence-based interventions grounded in neuroscience
- Create assessments that understand neurodiversity and strengths
- Design protocols that challenge traditional clinical approaches
- Build guidelines that are neurodiversity-affirming
- Develop monitoring approaches that focus on strengths and growth
- Create clinical tools that empower rather than pathologize
- Build a clinical framework that challenges conventional mental health practices

# RESEARCH TEXT
{text[:3000]}

# CONTENT INSIGHTS
{json.dumps(content_insights, indent=2) if content_insights else "No additional insights available"}

# INSTRUCTIONS
Extract clinical content that:
- Develops interventions based on neuroscience research
- Creates assessments that understand neurodiversity
- Focuses on outcomes that empower clients
- Develops protocols that challenge traditional approaches
- Creates guidelines that are strengths-based
- Identifies contraindications that respect neurodiversity
- Monitors side effects in ways that support clients
- Develops monitoring approaches that focus on growth

Focus on creating clinical content that empowers clients and challenges conventional mental health practices.

# OUTPUT FORMAT
Provide your analysis as a JSON object with the specified schema. 

CRITICAL: Each field must be a simple list of strings, not nested objects. For example:
- "headlines": ["headline 1", "headline 2", "headline 3"]
- NOT: "headlines": {"content": ["headline 1", "headline 2"]}

Ensure all fields are present and properly formatted as simple lists.
"""
        
        try:
            result = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=ClinicalContent,
                temperature=0.7
            )
            return result if result else ClinicalContent()
        except Exception as e:
            log_with_telemetry(
                logger.error,
                "Error extracting clinical content: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Clinical content extraction failed",
                details={"error": str(e)},
            )
            return ClinicalContent()

    async def _extract_research_content(self, text: str, content_insights: Dict[str, Any]) -> ResearchContent:
        """Extract research content with enhanced prompts."""
        prompt = f"""
# ROLE
You are a research content specialist for Enlitens, a revolutionary mental health practice that challenges traditional clinical approaches through neuroscience-based care.

# CONTEXT
Enlitens is building a rebellion against the mental health system by:
- Replacing shame with science (neurobiological truth)
- Deconstructing systemic failures in traditional clinical practices
- Empowering clients through strengths-based, neurodiversity-affirming approaches
- Using cutting-edge neuroscience research to inform treatment

# PURPOSE
Extract research content that will help us:
- Build authority as neuroscience-based mental health experts
- Support our clinical approaches with scientific evidence
- Challenge conventional mental health practices with research
- Educate clients about the science behind their mental health
- Create evidence-based content for our practice and website
- Build credibility through scientific research
- Support our rebellion against traditional mental health approaches

# RESEARCH TEXT
{text[:3000]}

# CONTENT INSIGHTS
{json.dumps(content_insights, indent=2) if content_insights else "No additional insights available"}

# INSTRUCTIONS
Extract research content that:
- Supports our neuroscience-based approach to mental health
- Challenges conventional mental health practices with evidence
- Provides statistics that support our clinical approaches
- Identifies methodologies that align with our values
- Highlights limitations of traditional approaches
- Suggests future directions that support our mission
- Provides implications that empower clients
- Creates citations that build our authority

Focus on extracting research content that supports our rebellion against traditional mental health approaches and empowers our clients.

# OUTPUT FORMAT
Provide your analysis as a JSON object with the specified schema. 

CRITICAL: Each field must be a simple list of strings, not nested objects. For example:
- "headlines": ["headline 1", "headline 2", "headline 3"]
- NOT: "headlines": {"content": ["headline 1", "headline 2"]}

Ensure all fields are present and properly formatted as simple lists.
"""
        
        try:
            result = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=ResearchContent,
                temperature=0.7
            )
            return result if result else ResearchContent()
        except Exception as e:
            log_with_telemetry(
                logger.error,
                "Error extracting research content: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Research content extraction failed",
                details={"error": str(e)},
            )
            return ResearchContent()

    async def _extract_content_creation_ideas(self, text: str, content_insights: Dict[str, Any]) -> ContentCreationIdeas:
        """Extract content creation ideas with enhanced prompts."""
        prompt = f"""
# ROLE
You are a content creation strategist for Enlitens, a revolutionary mental health practice that challenges traditional clinical approaches through neuroscience-based care.

# CONTEXT
Enlitens is building a rebellion against the mental health system by:
- Replacing shame with science (neurobiological truth)
- Deconstructing systemic failures in traditional clinical practices
- Empowering clients through strengths-based, neurodiversity-affirming approaches
- Using cutting-edge neuroscience research to inform treatment

# PURPOSE
Extract content creation ideas that will help us:
- Generate innovative content ideas that challenge conventional mental health narratives
- Create content that educates and empowers clients
- Develop content that positions us as thought leaders
- Create content that attracts clients frustrated with traditional approaches
- Generate ideas for various content formats and platforms
- Develop content that supports our rebellion against traditional mental health

# RESEARCH TEXT
{text[:3000]}

# CONTENT INSIGHTS
{json.dumps(content_insights, indent=2) if content_insights else "No additional insights available"}

# INSTRUCTIONS
Extract content creation ideas that:
- Challenge conventional wisdom about mental health
- Educate people about neuroscience and mental health
- Empower clients through scientific understanding
- Support our rebellion against traditional approaches
- Create engaging, shareable content
- Build authority and trust
- Attract the right clients

Focus on creating innovative, educational, and empowering content ideas.

# OUTPUT FORMAT
Provide your analysis as a JSON object with the specified schema. 

CRITICAL: Each field must be a simple list of strings, not nested objects. For example:
- "headlines": ["headline 1", "headline 2", "headline 3"]
- NOT: "headlines": {"content": ["headline 1", "headline 2"]}

Ensure all fields are present and properly formatted as simple lists.
"""
        
        try:
            result = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=ContentCreationIdeas,
                temperature=0.7
            )
            return result if result else ContentCreationIdeas()
        except Exception as e:
            log_with_telemetry(
                logger.error,
                "Error extracting content creation ideas: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Content ideas extraction failed",
                details={"error": str(e)},
            )
            return ContentCreationIdeas()

    async def close(self):
        """Close the agent and clean up resources."""
        try:
            await self.ollama_client.close()
            self.extraction_tools.cleanup()
            logger.info("Enhanced Complete Enlitens Agent closed")
        except Exception as e:
            log_with_telemetry(
                logger.error,
                "Error closing agent: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MINOR,
                impact="Agent cleanup encountered errors",
                details={"error": str(e)},
            )

    def _summarize_research(self, text: str, max_chars: int = 1200) -> str:
        snippet = text.strip().replace("\n", " ")
        if len(snippet) <= max_chars:
            return snippet
        return snippet[:max_chars - 3].rstrip() + "..."

    def _normalize_marketing_notes(self, notes: str) -> str:
        """Normalize llama output into plain-text bullet lines."""
        try:
            data = json.loads(notes)
            if isinstance(data, dict):
                flattened = []
                for key, value in data.items():
                    if isinstance(value, list):
                        flattened.extend(self._coerce_to_string(item) for item in value)
                    else:
                        flattened.append(self._coerce_to_string(value))
                return "\n".join(flattened)
        except Exception:
            pass

        # Fallback: strip excessive bullet markers and flatten label:value lines
        cleaned_lines = []
        for line in notes.splitlines():
            line = line.strip("- •\t ")
            if not line:
                continue
            # Collapse "label: value" pairs
            if ":" in line:
                key, value = line.split(":", 1)
                cleaned_lines.append(value.strip())
            else:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _coerce_to_string(self, item: Any) -> str:
        if isinstance(item, dict):
            parts = []
            for k, v in item.items():
                if isinstance(v, (dict, list)):
                    parts.append(self._coerce_to_string(v))
                else:
                    parts.append(str(v))
            return " ".join(parts)
        if isinstance(item, list):
            return " ".join(self._coerce_to_string(v) for v in item)
        return str(item)
