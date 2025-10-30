"""
Pydantic models for Enlitens Knowledge Base schema enforcement.
These models ensure structured, validated JSON output from the LLM.

HALLUCINATION PREVENTION:
- All statistics require citations from source documents
- Testimonials, credentials, and social proof fields REMOVED (FTC compliance)
- Validators block practice statistics and fabricated content
"""

from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import List, Optional, Dict, Any
from datetime import datetime
import re


class Citation(BaseModel):
    """Citation for verifiable claims - REQUIRED for all statistics"""
    quote: str = Field(description="EXACT verbatim quote from source document")
    source_id: str = Field(description="Document ID from knowledge base")
    source_title: str = Field(default="", description="Title of source document")
    page_or_section: str = Field(default="", description="Page number or section name")

    @field_validator('quote')
    @classmethod
    def quote_not_empty(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Citation quote must be at least 10 characters")
        return v


class VerifiedStatistic(BaseModel):
    """Statistics MUST cite research sources - NO practice statistics allowed"""
    claim: str = Field(description="The statistical claim with proper attribution")
    citation: Citation = Field(description="REQUIRED: Source citation for this statistic")

    @field_validator('claim')
    @classmethod
    def block_practice_stats(cls, v):
        """Block any statistics about Enlitens practice"""
        blocked_patterns = [
            r'enlitens\s+(clients?|patients?|practice)',
            r'\d+%\s+of\s+(our|my)\s+clients?',
            r'(our|my)\s+(clients?|patients?|practice)\s+report',
            r'enlitens\s+data',
            r'clients?\s+at\s+enlitens'
        ]

        for pattern in blocked_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(
                    f"BLOCKED: Cannot generate Enlitens practice statistics. "
                    f"Only cite published research. Blocked pattern: {pattern}"
                )

        # Must include attribution - flexible patterns
        attribution_phrases = [
            'according to', 'research shows', 'study found', 'found that',
            'suggests that', 'indicates that', 'demonstrates that', 'reports that',
            'showed that', 'revealed that', 'data show', 'evidence suggests'
        ]

        # Also accept year patterns like (2023) or et al. (2023)
        has_year_pattern = bool(re.search(r'\(\d{4}\)', v))
        has_attribution = any(phrase in v.lower() for phrase in attribution_phrases)

        if not (has_attribution or has_year_pattern):
            raise ValueError(
                "Statistics must include attribution like 'According to [Author] (Year)' "
                "or phrases like 'Research shows', 'Study found', etc."
            )

        return v

    @field_validator('citation')
    @classmethod
    def validate_citation_exists(cls, v, info: ValidationInfo):
        """Verify citation quote appears in source text"""
        context = info.context
        if context and 'source_text' in context:
            source = context['source_text']
            # Check if quote appears in source
            if v.quote not in source:
                # Try fuzzy match (allow minor variations)
                import difflib
                # Split source into sentences
                sentences = source.split('. ')
                best_match = difflib.get_close_matches(v.quote, sentences, n=1, cutoff=0.8)
                if not best_match:
                    raise ValueError(
                        f"HALLUCINATION DETECTED: Citation not found in source text. "
                        f"Quote: '{v.quote[:100]}...'"
                    )
        return v


class RebellionFramework(BaseModel):
    """Rebellion framework content extracted from research papers."""
    narrative_deconstruction: List[str] = Field(default_factory=list, description="Narrative deconstruction insights")
    sensory_profiling: List[str] = Field(default_factory=list, description="Sensory profiling insights")
    executive_function: List[str] = Field(default_factory=list, description="Executive function insights")
    social_processing: List[str] = Field(default_factory=list, description="Social processing insights")
    strengths_synthesis: List[str] = Field(default_factory=list, description="Strengths synthesis insights")
    rebellion_themes: List[str] = Field(default_factory=list, description="Rebellion themes")
    aha_moments: List[str] = Field(default_factory=list, description="Aha moments for clients")


class MarketingContent(BaseModel):
    """Marketing content extracted from research papers.

    NOTE: social_proof field REMOVED for FTC compliance (no fake testimonials)
    """
    headlines: List[str] = Field(default_factory=list, description="Marketing headlines")
    taglines: List[str] = Field(default_factory=list, description="Marketing taglines")
    value_propositions: List[str] = Field(default_factory=list, description="Value propositions")
    benefits: List[str] = Field(default_factory=list, description="Client benefits")
    pain_points: List[str] = Field(default_factory=list, description="Client pain points addressed")
    # social_proof: REMOVED - FTC violation (16 CFR Part 465)


class SEOContent(BaseModel):
    """SEO content extracted from research papers."""
    primary_keywords: List[str] = Field(default_factory=list, description="Primary SEO keywords")
    secondary_keywords: List[str] = Field(default_factory=list, description="Secondary SEO keywords")
    long_tail_keywords: List[str] = Field(default_factory=list, description="Long-tail SEO keywords")
    meta_descriptions: List[str] = Field(default_factory=list, description="Meta descriptions")
    title_tags: List[str] = Field(default_factory=list, description="Title tags")
    content_topics: List[str] = Field(default_factory=list, description="Content topics")


class WebsiteCopy(BaseModel):
    """Website copy extracted from research papers.

    NOTE: testimonials field REMOVED for FTC compliance (no fake testimonials)
    """
    about_sections: List[str] = Field(default_factory=list, description="About page content")
    feature_descriptions: List[str] = Field(default_factory=list, description="Feature descriptions")
    benefit_statements: List[str] = Field(default_factory=list, description="Benefit statements")
    # testimonials: REMOVED - FTC violation (16 CFR Part 465)
    faq_content: List[str] = Field(default_factory=list, description="FAQ content")
    service_descriptions: List[str] = Field(default_factory=list, description="Service descriptions")


class BlogContent(BaseModel):
    """Blog content extracted from research papers.

    IMPORTANT: statistics field now uses VerifiedStatistic with required citations
    """
    article_ideas: List[str] = Field(default_factory=list, description="Blog article ideas")
    blog_outlines: List[str] = Field(default_factory=list, description="Blog post outlines")
    talking_points: List[str] = Field(default_factory=list, description="Talking points")
    expert_quotes: List[str] = Field(default_factory=list, description="Expert quotes")

    # Statistics now REQUIRE citations
    statistics: List[VerifiedStatistic] = Field(
        default_factory=list,
        description="ONLY cite statistics from research papers with proper attribution"
    )

    case_studies: List[str] = Field(
        default_factory=list,
        description="Hypothetical case study templates - must be marked as examples"
    )
    how_to_guides: List[str] = Field(default_factory=list, description="How-to guide ideas")
    myth_busting: List[str] = Field(default_factory=list, description="Myth-busting content")

    @field_validator('case_studies')
    @classmethod
    def mark_as_templates(cls, v):
        """Ensure case studies are clearly marked as hypothetical examples"""
        return [
            f"[HYPOTHETICAL EXAMPLE] {case}" if not case.startswith('[') else case
            for case in v
        ]


class SocialMediaContent(BaseModel):
    """Social media content extracted from research papers."""
    post_ideas: List[str] = Field(default_factory=list, description="Social media post ideas")
    captions: List[str] = Field(default_factory=list, description="Social media captions")
    quotes: List[str] = Field(default_factory=list, description="Quote content")
    hashtags: List[str] = Field(default_factory=list, description="Hashtag suggestions")
    story_ideas: List[str] = Field(default_factory=list, description="Story ideas")
    reel_ideas: List[str] = Field(default_factory=list, description="Reel ideas")
    carousel_content: List[str] = Field(default_factory=list, description="Carousel content")
    poll_questions: List[str] = Field(default_factory=list, description="Poll questions")


class EducationalContent(BaseModel):
    """Educational content extracted from research papers."""
    explanations: List[str] = Field(default_factory=list, description="Educational explanations")
    examples: List[str] = Field(default_factory=list, description="Educational examples")
    analogies: List[str] = Field(default_factory=list, description="Educational analogies")
    definitions: List[str] = Field(default_factory=list, description="Educational definitions")
    processes: List[str] = Field(default_factory=list, description="Educational processes")
    comparisons: List[str] = Field(default_factory=list, description="Educational comparisons")
    visual_aids: List[str] = Field(default_factory=list, description="Visual aid suggestions")
    learning_objectives: List[str] = Field(default_factory=list, description="Learning objectives")


class ClinicalContent(BaseModel):
    """Clinical content extracted from research papers."""
    interventions: List[str] = Field(default_factory=list, description="Clinical interventions")
    assessments: List[str] = Field(default_factory=list, description="Assessment tools")
    outcomes: List[str] = Field(default_factory=list, description="Clinical outcomes")
    protocols: List[str] = Field(default_factory=list, description="Clinical protocols")
    guidelines: List[str] = Field(default_factory=list, description="Clinical guidelines")
    contraindications: List[str] = Field(default_factory=list, description="Contraindications")
    side_effects: List[str] = Field(default_factory=list, description="Side effects")
    monitoring: List[str] = Field(default_factory=list, description="Monitoring approaches")


class ResearchContent(BaseModel):
    """Research content extracted from research papers."""
    findings: List[str] = Field(default_factory=list, description="Research findings")
    statistics: List[str] = Field(default_factory=list, description="Statistical data from research")
    methodologies: List[str] = Field(default_factory=list, description="Research methodologies")
    limitations: List[str] = Field(default_factory=list, description="Research limitations")
    future_directions: List[str] = Field(default_factory=list, description="Future research directions")
    implications: List[str] = Field(default_factory=list, description="Clinical implications")
    citations: List[str] = Field(default_factory=list, description="Citation information")
    references: List[str] = Field(default_factory=list, description="Reference information")


class ContentCreationIdeas(BaseModel):
    """Content creation ideas extracted from research papers."""
    topic_ideas: List[str] = Field(default_factory=list, description="Topic ideas")
    angle_ideas: List[str] = Field(default_factory=list, description="Angle ideas")
    hook_ideas: List[str] = Field(default_factory=list, description="Hook ideas")
    series_ideas: List[str] = Field(default_factory=list, description="Series ideas")
    collaboration_ideas: List[str] = Field(default_factory=list, description="Collaboration ideas")
    trend_ideas: List[str] = Field(default_factory=list, description="Trend ideas")
    seasonal_ideas: List[str] = Field(default_factory=list, description="Seasonal ideas")


class DocumentMetadata(BaseModel):
    """Metadata for each processed document."""
    document_id: str = Field(description="Unique document identifier")
    filename: str = Field(description="Original filename")
    processing_timestamp: datetime = Field(default_factory=datetime.now, description="Processing timestamp")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    page_count: Optional[int] = Field(None, description="Number of pages")
    word_count: Optional[int] = Field(None, description="Word count")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    # New field for hallucination prevention
    full_text_stored: bool = Field(default=False, description="Whether full document text is stored for verification")


class ExtractedEntities(BaseModel):
    """Entities extracted from the document."""
    biomedical_entities: List[str] = Field(default_factory=list, description="Biomedical entities")
    neuroscience_entities: List[str] = Field(default_factory=list, description="Neuroscience entities")
    clinical_entities: List[str] = Field(default_factory=list, description="Clinical entities")
    statistical_entities: List[str] = Field(default_factory=list, description="Statistical entities")
    total_entities: int = Field(default=0, description="Total number of entities extracted")


class EnlitensKnowledgeEntry(BaseModel):
    """Complete knowledge base entry for a single document."""
    metadata: DocumentMetadata = Field(description="Document metadata")
    extracted_entities: ExtractedEntities = Field(description="Extracted entities")
    rebellion_framework: RebellionFramework = Field(description="Rebellion framework content")
    marketing_content: MarketingContent = Field(description="Marketing content")
    seo_content: SEOContent = Field(description="SEO content")
    website_copy: WebsiteCopy = Field(description="Website copy")
    blog_content: BlogContent = Field(description="Blog content")
    social_media_content: SocialMediaContent = Field(description="Social media content")
    educational_content: EducationalContent = Field(description="Educational content")
    clinical_content: ClinicalContent = Field(description="Clinical content")
    research_content: ResearchContent = Field(description="Research content")
    content_creation_ideas: ContentCreationIdeas = Field(description="Content creation ideas")

    # New field for verification
    full_document_text: Optional[str] = Field(
        None,
        description="Full document text for citation verification - NOT included in JSON output"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "metadata": {
                    "document_id": "2023-67353-007",
                    "filename": "2023-67353-007.pdf",
                    "processing_timestamp": "2025-01-25T21:30:00Z",
                    "file_size": 1024000,
                    "page_count": 15,
                    "word_count": 5000,
                    "processing_time": 180.5,
                    "full_text_stored": True
                },
                "extracted_entities": {
                    "biomedical_entities": ["neuroplasticity", "synaptic plasticity"],
                    "neuroscience_entities": ["prefrontal cortex", "amygdala"],
                    "clinical_entities": ["anxiety", "depression"],
                    "statistical_entities": ["p < 0.05", "effect size 0.8"],
                    "total_entities": 25
                },
                "rebellion_framework": {
                    "narrative_deconstruction": ["Traditional therapy focuses on symptoms rather than neurobiology"],
                    "sensory_profiling": ["Interoceptive awareness training"],
                    "executive_function": ["Prefrontal cortex regulation"],
                    "social_processing": ["Social anxiety neurobiology"],
                    "strengths_synthesis": ["Neurodiversity as strength"],
                    "rebellion_themes": ["Science over shame"],
                    "aha_moments": ["Your brain isn't broken, it's adapting"]
                }
            }
        }


class EnlitensKnowledgeBase(BaseModel):
    """Complete knowledge base containing all processed documents."""
    version: str = Field(default="1.0", description="Knowledge base version")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    total_documents: int = Field(default=0, description="Total number of documents")
    documents: List[EnlitensKnowledgeEntry] = Field(default_factory=list, description="List of processed documents")

    class Config:
        json_schema_extra = {
            "example": {
                "version": "1.0",
                "created_at": "2025-01-25T21:30:00Z",
                "total_documents": 344,
                "documents": []
            }
        }
