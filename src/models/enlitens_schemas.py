"""
Pydantic models for Enlitens Knowledge Base schema enforcement.
These models ensure structured, validated JSON output from the LLM.

HALLUCINATION PREVENTION:
- All statistics require citations from source documents
- Testimonials, credentials, and social proof fields REMOVED (FTC compliance)
- Validators block practice statistics and fabricated content
"""

from pydantic import BaseModel, Field, field_validator, ValidationInfo, HttpUrl
from typing import ClassVar, List, Optional, Dict, Any, Literal, Tuple
from datetime import datetime
import difflib
import logging
import re

from src.monitoring.error_telemetry import TelemetrySeverity, log_with_telemetry


logger = logging.getLogger(__name__)


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
    narrative_deconstruction: List[str] = Field(
        default_factory=list,
        description="Narrative deconstruction insights",
        json_schema_extra={"minItems": 3, "maxItems": 10},
    )
    sensory_profiling: List[str] = Field(
        default_factory=list,
        description="Sensory profiling insights",
        json_schema_extra={"minItems": 3, "maxItems": 10},
    )
    executive_function: List[str] = Field(
        default_factory=list,
        description="Executive function insights",
        json_schema_extra={"minItems": 3, "maxItems": 10},
    )
    social_processing: List[str] = Field(
        default_factory=list,
        description="Social processing insights",
        json_schema_extra={"minItems": 3, "maxItems": 10},
    )
    strengths_synthesis: List[str] = Field(
        default_factory=list,
        description="Strengths synthesis insights",
        json_schema_extra={"minItems": 3, "maxItems": 10},
    )
    rebellion_themes: List[str] = Field(
        default_factory=list,
        description="Rebellion themes",
        json_schema_extra={"minItems": 3, "maxItems": 10},
    )
    aha_moments: List[str] = Field(
        default_factory=list,
        description="Aha moments for clients",
        json_schema_extra={"minItems": 3, "maxItems": 10},
    )

    @staticmethod
    def _coerce_list(values: List[Any]) -> List[str]:
        coerced: List[str] = []
        for v in values or []:
            if isinstance(v, list):
                coerced.append(" ".join(str(x) for x in v))
            else:
                coerced.append(str(v))
        return coerced

    @field_validator(
        "narrative_deconstruction",
        "sensory_profiling",
        "executive_function",
        "social_processing",
        "strengths_synthesis",
        "rebellion_themes",
        "aha_moments",
        mode="before",
    )
    @classmethod
    def _flatten_nested_lists(cls, v):
        if isinstance(v, list) and any(isinstance(x, list) for x in v):
            return cls._coerce_list(v)
        return v


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
    content_topics: List[str] = Field(default_factory=list, description="Content topics")


class WebsiteCopy(BaseModel):
    """Website copy extracted from research papers.

    NOTE: testimonials field REMOVED for FTC compliance (no fake testimonials)
    """
    about_sections: List[str] = Field(default_factory=list, description="About page content")
    # testimonials: REMOVED - FTC violation (16 CFR Part 465)
    faq_content: List[str] = Field(default_factory=list, description="FAQ content")
    topic_ideas: List[str] = Field(
        default_factory=list,
        description="High-level topic ideas for future website updates",
    )


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
    poll_questions: List[str] = Field(default_factory=list, description="Poll questions")

    @staticmethod
    def _extract_quote_body(value: str) -> str:
        if not isinstance(value, str):
            return ""
        match = re.search(r'"([^"]+)"', value)
        if match:
            return match.group(1).strip()
        # Remove trailing citation blocks like [Source 1]
        cleaned = re.sub(r"\s*\[Source[^\]]+\]\s*$", "", value).strip()
        return cleaned

    @staticmethod
    def _quote_in_text(quote: str, text: str) -> bool:
        if not quote or not text:
            return False
        if quote in text:
            return True
        sentences = [
            sentence.strip()
            for sentence in re.split(r'(?<=[.!?])\s+', text)
            if sentence.strip()
        ]
        if not sentences:
            return False
        best_match = difflib.get_close_matches(quote, sentences, n=1, cutoff=0.85)
        return bool(best_match)

    @field_validator("quotes")
    @classmethod
    def validate_quotes(cls, values: List[str], info: ValidationInfo) -> List[str]:
        if not values:
            return []

        context = info.context or {}
        source_text = (context.get("source_text") or "").strip()
        raw_segments = context.get("source_segments") or []
        citation_map = dict(context.get("source_citation_map") or {})
        telemetry_meta = context.get("quote_validation_telemetry") or {}
        missing_note = (context.get("quote_missing_note") or "").strip()

        normalized_segments: List[str] = []
        for segment in raw_segments:
            if isinstance(segment, str):
                stripped = segment.strip()
                if stripped:
                    normalized_segments.append(stripped)
            elif isinstance(segment, dict):
                text = str(segment.get("text", "")).strip()
                tag = str(segment.get("tag", "")).strip()
                if text:
                    normalized_segments.append(text)
                    if tag:
                        citation_map.setdefault(tag, text)

        combined_sources = "\n".join(
            [part for part in [source_text, *normalized_segments] if part]
        ).strip()

        sentences = [
            sentence.strip()
            for sentence in re.split(r'(?<=[.!?])\s+', combined_sources)
            if sentence.strip()
        ]

        valid_quotes: List[str] = []
        missing_quotes: List[str] = []

        for value in values:
            quote_body = cls._extract_quote_body(value)
            if not quote_body:
                raise ValueError("Quote entries must include extractable quoted text")

            matched = False
            citation_tags = [tag.strip() for tag in re.findall(r"\[([^\]]+)\]", value)]
            for tag in citation_tags:
                if not tag:
                    continue
                normalized_tag = " ".join(tag.split())
                source_payload = citation_map.get(normalized_tag) or citation_map.get(tag)
                if source_payload and cls._quote_in_text(quote_body, source_payload):
                    matched = True
                    break

            if not matched and combined_sources:
                if quote_body in combined_sources or difflib.get_close_matches(quote_body, sentences, n=1, cutoff=0.85):
                    matched = True

            if matched:
                valid_quotes.append(value)
            else:
                missing_quotes.append(value)

        if valid_quotes:
            if missing_quotes and telemetry_meta:
                log_with_telemetry(
                    logger.warning,
                    "Filtered social media quotes without matching evidence",
                    agent=telemetry_meta.get("agent", "unknown"),
                    severity=TelemetrySeverity.MINOR,
                    impact="quote_validation_partial_mismatch",
                    doc_id=telemetry_meta.get("doc_id"),
                    details={
                        "dropped_quotes": missing_quotes,
                        "available_citations": list(citation_map.keys()),
                    },
                )
            return valid_quotes

        if telemetry_meta:
            log_with_telemetry(
                logger.warning,
                "No verifiable quotes matched provided sources",
                agent=telemetry_meta.get("agent", "unknown"),
                severity=TelemetrySeverity.MINOR,
                impact="quote_validation_missing_evidence",
                doc_id=telemetry_meta.get("doc_id"),
                details={
                    "requested_quotes": values,
                    "available_citations": list(citation_map.keys()),
                },
            )

        fallback_note = missing_note or "Evidence unavailable: no verified quotes could be matched to the provided sources."
        return [fallback_note]


class EducationalContent(BaseModel):
    """Educational content extracted from research papers."""
    explanations: List[str] = Field(
        default_factory=list,
        description="Educational explanations",
        json_schema_extra={"minItems": 5, "maxItems": 10},
    )
    examples: List[str] = Field(
        default_factory=list,
        description="Educational examples",
        json_schema_extra={"minItems": 5, "maxItems": 10},
    )
    analogies: List[str] = Field(
        default_factory=list,
        description="Educational analogies",
        json_schema_extra={"minItems": 5, "maxItems": 10},
    )
    definitions: List[str] = Field(
        default_factory=list,
        description="Educational definitions",
        json_schema_extra={"minItems": 5, "maxItems": 10},
    )
    processes: List[str] = Field(
        default_factory=list,
        description="Educational processes",
        json_schema_extra={"minItems": 5, "maxItems": 10},
    )
    comparisons: List[str] = Field(
        default_factory=list,
        description="Educational comparisons",
        json_schema_extra={"minItems": 5, "maxItems": 10},
    )
    visual_aids: List[str] = Field(
        default_factory=list,
        description="Visual aid suggestions",
        json_schema_extra={"minItems": 5, "maxItems": 10},
    )
    learning_objectives: List[str] = Field(
        default_factory=list,
        description="Learning objectives",
        json_schema_extra={"minItems": 5, "maxItems": 10},
    )


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

    @field_validator("citations", "references", mode="before")
    @classmethod
    def _normalize_reference_items(cls, value):
        if value is None:
            return []
        if not isinstance(value, list):
            return value
        normalized = []
        for item in value:
            if isinstance(item, str):
                normalized.append(item.strip())
            elif isinstance(item, dict):
                author = item.get("author") or item.get("authors") or "Unknown author"
                year = item.get("year")
                title = item.get("title")
                journal = item.get("journal")
                doi = item.get("doi")
                pages = item.get("pages")
                parts = []
                if author:
                    parts.append(str(author).strip())
                if year:
                    parts.append(f"({year})")
                if title:
                    parts.append(str(title).strip())
                if journal:
                    parts.append(str(journal).strip())
                if pages:
                    parts.append(f"pp. {pages}")
                if doi:
                    parts.append(f"doi:{doi}")
                normalized.append(" ".join(parts))
            else:
                normalized.append(str(item))
        return normalized


class ContentCreationIdeas(BaseModel):
    """Content creation ideas extracted from research papers."""
    topic_ideas: List[str] = Field(default_factory=list, description="Topic ideas")
    angle_ideas: List[str] = Field(default_factory=list, description="Angle ideas")
    hook_ideas: List[str] = Field(default_factory=list, description="Hook ideas")
    series_ideas: List[str] = Field(default_factory=list, description="Series ideas")
    collaboration_ideas: List[str] = Field(default_factory=list, description="Collaboration ideas")
    trend_ideas: List[str] = Field(default_factory=list, description="Trend ideas")
    seasonal_ideas: List[str] = Field(default_factory=list, description="Seasonal ideas")


class ClientProfile(BaseModel):
    """Client profile linking intake language to specific research citations."""

    profile_name: str = Field(
        ...,
        description="Short, descriptive label for the client persona",
    )
    fictional_disclaimer: str = Field(
        ...,
        description="Explicit indicator that this scenario is fictional",
    )
    intake_reference: str = Field(
        ...,
        description="Direct intake phrasing, quoted exactly as the client or intake form stated it",
    )
    persona_overview: str = Field(
        ...,
        description="Narrative snapshot of the fictional scenario (no clinical outcomes)",
    )
    research_reference: str = Field(
        ...,
        description="Sentence that cites a retrieved research passage using [Source #] tags",
        json_schema_extra={"pattern": r"\[Source [^\]]+\]"},
    )
    benefit_explanation: str = Field(
        ...,
        description="Why the cited research benefits this client, explicitly citing [Source #]",
        json_schema_extra={"pattern": r"\[Source [^\]]+\]"},
    )
    st_louis_alignment: Optional[str] = Field(
        None,
        description="Optional tie-in to St. Louis context or community realities, also citing [Source #] when applicable",
        json_schema_extra={"pattern": r"\[Source [^\]]+\]"},
    )
    regional_touchpoints: List[str] = Field(
        default_factory=list,
        description="Named neighbourhoods, schools, or community sites that shape the client's week",
        json_schema_extra={"minItems": 3, "maxItems": 8},
    )
    masking_signals: List[str] = Field(
        default_factory=list,
        description="Behavioural or autobiographical signs of masking/burnout",
        json_schema_extra={"minItems": 2, "maxItems": 8},
    )
    unmet_needs: List[str] = Field(
        default_factory=list,
        description="Core needs or system barriers that intensify dysregulation",
        json_schema_extra={"minItems": 3, "maxItems": 8},
    )
    support_recommendations: List[str] = Field(
        default_factory=list,
        description="Evidence-informed focus areas (no promises of outcomes)",
        json_schema_extra={"minItems": 3, "maxItems": 8},
    )
    cautionary_flags: List[str] = Field(
        default_factory=list,
        description="Boundary or ethical cautions for the care team",
        json_schema_extra={"minItems": 2, "maxItems": 6},
    )

    _FICTIONAL_KEYWORD: ClassVar[str] = "FICTIONAL"
    _OUTCOME_BLOCKLIST: ClassVar[Tuple[str, ...]] = (
        "cure",
        "guarantee",
        "100%",
        "certain outcome",
        "promised result",
        "clinically proven",
    )

    @field_validator("profile_name")
    @classmethod
    def profile_name_not_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Profile name cannot be empty")
        return cleaned

    @field_validator("fictional_disclaimer")
    @classmethod
    def ensure_fictional_label(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Fictional disclaimer cannot be empty")
        if cls._FICTIONAL_KEYWORD.lower() not in cleaned.lower():
            raise ValueError("Fictional disclaimer must clearly state the scenario is FICTIONAL")
        return cleaned

    @field_validator("intake_reference")
    @classmethod
    def ensure_intake_quote(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Intake reference cannot be empty")
        quote_markers = ('"', "'", "“", "”")
        if not any(marker in cleaned for marker in quote_markers):
            raise ValueError(
                "Intake reference must include the client's exact phrasing wrapped in quotes"
            )
        return cleaned

    @field_validator("persona_overview")
    @classmethod
    def block_outcome_language(cls, value: str) -> str:
        cleaned = value.strip()
        lowered = cleaned.lower()
        if any(term in lowered for term in cls._OUTCOME_BLOCKLIST):
            raise ValueError("Persona overview must not promise outcomes or cures")
        return cleaned

    @field_validator("research_reference", "benefit_explanation", "st_louis_alignment")
    @classmethod
    def ensure_source_citation(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        cleaned = value.strip()
        if cleaned and "[Source" not in cleaned:
            raise ValueError("Research-connected fields must include a [Source #] citation tag")
        lowered = cleaned.lower()
        if any(term in lowered for term in cls._OUTCOME_BLOCKLIST):
            raise ValueError("Research fields must not promise clinical outcomes")
        return cleaned

    @field_validator(
        "regional_touchpoints",
        "masking_signals",
        "unmet_needs",
        "support_recommendations",
        "cautionary_flags",
    )
    @classmethod
    def enforce_list_quality(cls, values: List[str], info: ValidationInfo) -> List[str]:
        values = [str(item).strip() for item in values if str(item).strip()]
        if not values:
            raise ValueError(f"{info.field_name} cannot be empty")
        unique_values = []
        seen = set()
        for item in values:
            lowered = item.lower()
            if lowered not in seen:
                seen.add(lowered)
                unique_values.append(item)
        field_meta = ClientProfile.model_fields.get(info.field_name) if info.field_name else None
        extras = field_meta.json_schema_extra if field_meta and field_meta.json_schema_extra else {}
        min_items = extras.get("minItems")
        max_items = extras.get("maxItems")
        if min_items and len(unique_values) < min_items:
            raise ValueError(f"{info.field_name} requires at least {min_items} entries")
        if max_items and len(unique_values) > max_items:
            unique_values = unique_values[:max_items]
        if info.field_name == "support_recommendations":
            for item in unique_values:
                lowered = item.lower()
                if any(term in lowered for term in cls._OUTCOME_BLOCKLIST):
                    raise ValueError("Support recommendations must not promise clinical outcomes")
        return unique_values


class ExternalResearchSource(BaseModel):
    """External research or regional data used to enrich fictional scenarios."""

    label: str = Field(
        ...,
        description="Bracketed label that should be cited in text (e.g., [Ext 1])",
        json_schema_extra={"pattern": r"\[Ext [^\]]+\]"},
    )
    title: str = Field(..., description="Title or headline of the external resource")
    url: HttpUrl = Field(..., description="Resolvable URL for verification")
    publisher: Optional[str] = Field(
        None,
        description="Publisher, newsroom, or organisation providing the data",
    )
    published_at: Optional[str] = Field(
        None,
        description="Publication date or retrieved date for the resource",
    )
    summary: str = Field(
        ...,
        description="Short factual summary of the relevant finding",
    )
    verification_status: Literal["verified", "needs_review", "flagged"] = Field(
        "verified",
        description="Confidence in the source after automated checks",
    )

    @field_validator("summary")
    @classmethod
    def summary_not_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("External research summary cannot be empty")
        return cleaned


class ClientProfileSet(BaseModel):
    """Container for exactly three client profiles derived from a research paper."""

    profiles: List[ClientProfile] = Field(
        ..., description="Exactly three tailored client profiles", min_length=3, max_length=3
    )
    shared_thread: Optional[str] = Field(
        None,
        description="Optional summary describing the common thread across the three profiles",
    )
    external_sources: List[ExternalResearchSource] = Field(
        default_factory=list,
        description="Externally verified references used across fictional personas",
        json_schema_extra={"maxItems": 12},
    )

    @field_validator("external_sources")
    @classmethod
    def ensure_unique_labels(cls, sources: List[ExternalResearchSource]) -> List[ExternalResearchSource]:
        seen = set()
        unique_sources: List[ExternalResearchSource] = []
        for source in sources:
            label = source.label.strip().lower()
            if label in seen:
                continue
            seen.add(label)
            unique_sources.append(source)
        max_items = cls.model_fields["external_sources"].json_schema_extra.get("maxItems")
        if max_items and len(unique_sources) > max_items:
            unique_sources = unique_sources[:max_items]
        return unique_sources

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
    client_profiles: Optional[ClientProfileSet] = Field(
        None, description="Three intake-grounded client profiles aligned to retrieved research"
    )

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
                },
                "client_profiles": {
                    "profiles": [
                        {
                            "profile_name": "Transit-triggered shutdowns",
                            "intake_reference": "\"I lose words after the evening Metro ride\"",
                            "research_reference": "[Source 1] documents sensory gating strain during urban commutes.",
                            "benefit_explanation": "[Source 1] shows vestibular supports lower shutdown risk for this pattern.",
                            "st_louis_alignment": "[Source 1] plus STL Metro noise complaints justify adapting transit routines.",
                        }
                    ],
                    "shared_thread": "Clients ride St. Louis transit systems under chronic sensory load.",
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
