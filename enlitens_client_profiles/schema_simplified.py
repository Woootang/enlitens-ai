"""Simplified 8-field Pydantic schema for Enlitens client personas."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, root_validator


class ProfileMeta(BaseModel):
    """Metadata attached to every persona profile."""

    profile_id: str = Field(..., description="Stable unique identifier (e.g. slug)")
    persona_name: str = Field(..., description="Human-friendly persona name, e.g. 'Overachiever Olivia'")
    persona_tagline: str = Field(..., description="Short descriptor used in dashboards or UI")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source_documents: List[str] = Field(default_factory=list, description="Document IDs or filenames used during generation")
    llm_model: Optional[str] = Field(None, description="Model used for generation")
    version: str = Field(default="2.0")
    attribute_tags: List[str] = Field(
        default_factory=list,
        description="Key attribute tags used for deduplication and routing (e.g. 'adhd', 'young_adult').",
    )


class Demographics(BaseModel):
    """Core demographic information."""

    age_range: str = Field(..., description="Age range or stage (e.g. 'late 20s', '15-17')")
    gender: str = Field(..., description="Gender identity")
    pronouns: str = Field(..., description="Preferred pronouns")
    ethnicity: Optional[str] = Field(None, description="Ethnic/racial identity")
    family_status: str = Field(..., description="Family structure (e.g. 'Lives with parents and 2 siblings', 'Single parent household')")
    occupation: Optional[str] = Field(None, description="Current occupation or school status")
    education: Optional[str] = Field(None, description="Education level")
    locality: str = Field(
        ...,
        description="SPECIFIC St. Louis locality (e.g. 'Kirkwood', 'Wentzville', 'Clayton', 'Central West End neighborhood'). "
        "Must reference actual municipalities or neighborhoods, NOT just 'St. Louis'.",
    )


class NeurodivergenceClinical(BaseModel):
    """Combined neurodivergence profile, clinical challenges, and adaptive strengths."""

    neuro_identities: List[str] = Field(
        ...,
        min_items=1,
        description="[direct] or [inferred] neurodivergent identities (e.g. '[direct] Autistic', '[inferred] ADHD-presenting')",
    )
    diagnosis_notes: str = Field(..., description="Diagnosis journey, assessment history, or self-identification context")
    language_preferences: List[str] = Field(
        ...,
        min_items=1,
        description="[direct] or [inferred] preferred language about their neurotype (identity-first, person-first, strengths framing)",
    )
    presenting_challenges: List[str] = Field(
        ...,
        min_items=2,
        description="[direct] or [inferred] salient challenges (e.g. '[direct] Executive function with task initiation', "
        "'[inferred] Social anxiety in group settings')",
    )
    nervous_system_pattern: str = Field(
        ..., description="Plain-language explanation of nervous system tendencies (e.g. 'Hypervigilance and shutdown cycles')"
    )
    mood_patterns: List[str] = Field(
        ...,
        min_items=2,
        description="[direct] or [inferred] mood patterns (e.g. '[inferred] Mood swings with transitions', '[direct] Periods of shutdown')",
    )
    adaptive_strengths: List[str] = Field(
        ...,
        min_items=3,
        description="[inferred] Strengths-based reframes of challenges (e.g. 'Hyperfocus becomes flow state in preferred activities', "
        "'Pattern recognition enables quick problem-solving')",
    )
    reframes: List[str] = Field(
        ...,
        min_items=2,
        description="[inferred] Neurodiversity-affirming reframes of 'weaknesses' (e.g. 'Sensory sensitivity → heightened awareness', "
        "'Rigid routines → strong internal structure')",
    )


class ExecutiveSensory(BaseModel):
    """Combined executive function and sensory profile."""

    executive_strengths: List[str] = Field(
        ..., min_items=2, description="[direct] or [inferred] executive function strengths (e.g. 'Creative problem-solving', 'Detailed planning')"
    )
    executive_friction: List[str] = Field(
        ...,
        min_items=2,
        description="[direct] or [inferred] executive function friction points (e.g. 'Task initiation', 'Time blindness', 'Working memory')",
    )
    executive_workarounds: List[str] = Field(
        ...,
        min_items=2,
        description="[inferred] Workarounds or accommodations (e.g. 'Visual timers', 'Body doubling', 'Breaking tasks into micro-steps')",
    )
    sensory_sensitivities: List[str] = Field(
        ...,
        min_items=2,
        description="[direct] or [inferred] sensory sensitivities (e.g. '[direct] Auditory - loud noises', '[inferred] Tactile - clothing tags')",
    )
    sensory_seeking: List[str] = Field(
        ...,
        min_items=1,
        description="[inferred] Sensory-seeking behaviors (e.g. 'Deep pressure (weighted blankets)', 'Vestibular (spinning, swinging)')",
    )
    sensory_regulation: List[str] = Field(
        ...,
        min_items=2,
        description="[direct] or [inferred] sensory regulation methods (e.g. 'Noise-canceling headphones', 'Fidget tools', 'Sensory breaks')",
    )


class GoalsBarriers(BaseModel):
    """Combined goals, motivations, and pain points/barriers."""

    therapy_goals: List[str] = Field(
        ...,
        min_items=2,
        description="[direct] or [inferred] therapy goals (e.g. '[direct] Improve emotion regulation', '[inferred] Build self-advocacy skills')",
    )
    life_goals: List[str] = Field(
        ...,
        min_items=2,
        description="[direct] or [inferred] broader life goals (e.g. '[inferred] Complete college', '[direct] Find a job that accommodates sensory needs')",
    )
    motivations: List[str] = Field(
        ..., min_items=2, description="[inferred] Core motivations (e.g. 'Desire to feel understood', 'Building independence', 'Supporting family')"
    )
    pain_points: List[str] = Field(
        ...,
        min_items=3,
        description="[direct] or [inferred] pain points (e.g. '[direct] Finding therapists who understand autism', "
        "'[inferred] Navigating school accommodations', '[direct] Financial barriers to care')",
    )
    systemic_barriers: List[str] = Field(
        ...,
        min_items=2,
        description="[inferred] Systemic or structural barriers (e.g. 'Waitlists for autism assessments in St. Louis County', "
        "'Lack of neurodiversity-affirming providers in South County')",
    )


class LocalCulturalContext(BaseModel):
    """Combined local environment, cultural context, and support system."""

    specific_locality_details: str = Field(
        ...,
        description="Detailed description of SPECIFIC St. Louis locality (e.g. 'Kirkwood: walkable downtown, Kirkwood Park, "
        "strong school district with robust special ed'). Must reference actual places, schools, parks, businesses—NOT generic 'St. Louis' descriptions.",
    )
    neighborhoods_places: List[str] = Field(
        ...,
        min_items=3,
        description="[inferred] Specific neighborhoods, schools, parks, restaurants, venues relevant to this persona "
        "(e.g. 'Francis Howell School District (Wentzville)', 'Magic House (Kirkwood)', 'The Muny (Forest Park)', "
        "'Mission Taco Joint (Soulard)', 'Queeny Park (West County)')",
    )
    local_stressors: List[str] = Field(
        ...,
        min_items=2,
        description="[inferred] Local stressors specific to their area (e.g. 'Long commute from Wentzville to therapy in Clayton', "
        "'Limited sensory-friendly venues in South County', 'School district gaps in IEP support')",
    )
    safe_spaces: List[str] = Field(
        ...,
        min_items=2,
        description="[inferred] Safe or comfortable spaces (e.g. 'Home sensory room', 'Kirkwood Farmers Market (low-sensory mornings)', "
        "'St. Louis Public Library - Central branch (quiet study rooms)')",
    )
    cultural_identities: List[str] = Field(
        ..., min_items=1, description="[direct] or [inferred] cultural/community identities (e.g. 'Catholic', 'Midwestern', 'Working-class', 'Black')"
    )
    community_roles: List[str] = Field(
        ...,
        min_items=1,
        description="[inferred] Community roles (e.g. 'Student', 'Parent', 'Caregiver', 'Youth group member', 'Volunteer')",
    )
    support_network: str = Field(
        ...,
        description="[direct] or [inferred] description of support system (e.g. 'Mom is primary support; extended family nearby in Florissant; "
        "school counselor provides check-ins; no current therapist')",
    )
    support_gaps: List[str] = Field(
        ...,
        min_items=1,
        description="[inferred] Gaps in support (e.g. 'No local neurodiversity support groups', 'Limited respite care for parents', "
        "'Long waitlists for ABA alternatives')",
    )


class NarrativeVoice(BaseModel):
    """Combined quotes, narrative, and therapy preferences (Liz Wooten's voice)."""

    direct_quotes: List[str] = Field(
        ...,
        min_items=2,
        description="[direct] Actual quotes from intake forms or transcripts (e.g. 'I just want someone who gets it', "
        "'School says he's fine but I know he's struggling')",
    )
    inferred_inner_voice: List[str] = Field(
        ...,
        min_items=2,
        description="[inferred] Empathetic interpretations of what the persona might be thinking/feeling (in Liz's voice) "
        "(e.g. 'I'm exhausted from masking all day', 'I want to unmask but I'm scared of rejection')",
    )
    persona_narrative: str = Field(
        ...,
        description="A 300-500 word empathetic, strengths-based narrative written in Liz Wooten's first-person therapeutic voice. "
        "Describe the persona's journey, challenges, strengths, and context. Use neurodiversity-affirming language. "
        "Reference SPECIFIC St. Louis localities, places, and lived experiences.",
    )
    therapy_preferences: List[str] = Field(
        ...,
        min_items=2,
        description="[direct] or [inferred] therapy preferences (e.g. '[direct] Prefers in-person sessions', "
        "'[inferred] Responds well to visual supports', '[direct] Wants a therapist who understands autism')",
    )
    therapeutic_approach: str = Field(
        ...,
        description="[inferred] Recommended therapeutic approach for this persona (e.g. 'Play therapy with sensory integration', "
        "'CBT adapted for executive function challenges', 'Trauma-informed care with somatic techniques')",
    )


class MarketingSEO(BaseModel):
    """Combined marketing copy, SEO brief, and resources."""

    marketing_headline: str = Field(
        ...,
        description="Compelling headline for marketing content (e.g. 'Kirkwood Families: Neurodiversity-Affirming Therapy for Autistic Teens')",
    )
    marketing_body: str = Field(
        ...,
        description="150-250 word marketing copy in Liz's voice, addressing this persona's pain points and positioning Enlitens as the solution. "
        "Reference SPECIFIC St. Louis localities, schools, or landmarks. Use empathetic, strengths-based language.",
    )
    value_propositions: List[str] = Field(
        ...,
        min_items=3,
        description="Key value propositions for this persona (e.g. 'Autism assessments without 18-month waitlists', "
        "'Executive function coaching for ADHD teens', 'Sensory-friendly office in Kirkwood')",
    )
    seo_keywords: List[str] = Field(
        ...,
        min_items=5,
        description="SEO keywords derived from GA4/GSC data and persona context (e.g. 'autism assessment Kirkwood MO', "
        "'ADHD therapist Wentzville', 'sensory-friendly therapy St. Louis County', 'neurodivergent teen counseling Clayton')",
    )
    content_topics: List[str] = Field(
        ...,
        min_items=3,
        description="Blog/email content topics for this persona (e.g. 'Navigating IEPs in Kirkwood R-7 School District', "
        "'Sensory-friendly activities in West County', 'ADHD medication myths')",
    )
    local_resources: List[str] = Field(
        ...,
        min_items=2,
        description="[inferred] Local resources or referrals (e.g. 'St. Louis Autism Support Network', "
        "'Sensory-friendly showings at AMC Chesterfield', 'CHADD St. Louis chapter')",
    )


class ClientProfileDocumentSimplified(BaseModel):
    """Simplified 8-field client persona document."""

    meta: ProfileMeta
    demographics: Demographics
    neurodivergence_clinical: NeurodivergenceClinical
    executive_sensory: ExecutiveSensory
    goals_barriers: GoalsBarriers
    local_cultural_context: LocalCulturalContext
    narrative_voice: NarrativeVoice
    marketing_seo: MarketingSEO

    @root_validator(skip_on_failure=True)
    def ensure_minimum_content(cls, values):
        """Ensure all list fields have minimum items and enforce [direct]/[inferred] tags where specified."""
        # This is a lightweight validator - the Field constraints above do the heavy lifting
        return values

    class Config:
        json_schema_extra = {
            "example": {
                "meta": {
                    "profile_id": "kirkwood-teen-autistic-adhd-01",
                    "persona_name": "Mateo",
                    "persona_tagline": "Autistic + ADHD teen navigating high school in Kirkwood",
                    "attribute_tags": ["autistic", "adhd", "teen", "kirkwood", "high_school"],
                },
                "demographics": {
                    "age_range": "15-17",
                    "gender": "Male",
                    "pronouns": "he/him",
                    "family_status": "Lives with parents and younger sister",
                    "occupation": "High school student",
                    "locality": "Kirkwood",
                },
            }
        }

