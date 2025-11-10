from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ProfileMeta(BaseModel):
    profile_id: str = Field(..., description="Stable unique identifier (e.g. persona-001)")
    persona_name: str = Field(..., description="Human-friendly persona name, e.g. 'Late-Diagnosed Allison'")
    persona_tagline: Optional[str] = Field(None, description="Short descriptor used in dashboards or UI")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source_documents: List[str] = Field(default_factory=list, description="Document IDs or filenames used during generation")
    llm_model: Optional[str] = Field(None, description="Model used for generation")
    version: str = Field(default="2.0")
    attribute_tags: List[str] = Field(
        default_factory=list,
        description="Key attribute tags used for deduplication and routing (e.g. 'adhd', 'late_diagnosed', 'parent').",
    )


class IdentityDemographics(BaseModel):
    age_range: Optional[str] = Field(None, description="Age range or stage (e.g. 'late 30s', 'mid 20s')")
    gender: Optional[str] = None
    pronouns: Optional[str] = None
    orientation: Optional[str] = None
    ethnicity: Optional[str] = None
    current_life_situation: Optional[str] = Field(None, description="Current life situation (e.g. 'single mom of 4', 'grad student', 'recently divorced')")
    occupation: Optional[str] = None
    education: Optional[str] = None
    locality: Optional[str] = Field(None, description="High-level locale reference (e.g. 'South City', 'North County')")
    cultural_faith_identities: List[str] = Field(default_factory=list, description="Cultural, faith, or community identities that matter to them")


class DevelopmentalStory(BaseModel):
    childhood_environment: Optional[str] = Field(None, description="Childhood environment and experiences (0-12 years)")
    adolescence: Optional[str] = Field(None, description="Adolescent experiences (13-18 years)")
    early_adulthood: Optional[str] = Field(None, description="Early adulthood experiences (19-25 years)")
    family_structure: Optional[str] = Field(None, description="Who raised them, siblings, family dynamics, losses, separations")
    formative_adversities: List[str] = Field(default_factory=list, description="Significant adversities: poverty, abuse, instability, loss, trauma, displacement")
    educational_journey: Optional[str] = Field(None, description="Educational path, school changes, disruptions, achievements, struggles")
    pivotal_moments: List[str] = Field(default_factory=list, description="Pivotal moments or experiences that shaped who they are")
    intergenerational_patterns: Optional[str] = Field(None, description="Patterns inherited from family (e.g. 'mom also neurodivergent', 'generational trauma')")


class NeurodivergenceMentalHealth(BaseModel):
    identities: List[str] = Field(default_factory=list, description="Formal or self-identified neurodivergent/mental health identities")
    diagnosis_journey: Optional[str] = Field(None, description="How and when they were diagnosed, what led to it")
    how_it_shows_up: Optional[str] = Field(None, description="How neurodivergence/mental health shows up in their daily life")
    nervous_system_pattern: Optional[str] = Field(None, description="Plain-language explanation of nervous system tendencies")
    strengths_superpowers: List[str] = Field(default_factory=list, description="Adaptive strengths framed as 'superpowers'")
    current_coping_strategies: List[str] = Field(default_factory=list, description="Coping strategies they currently use")


class ExecutiveFunctionSensory(BaseModel):
    ef_strengths: List[str] = Field(default_factory=list, description="Executive function strengths")
    ef_friction_points: List[str] = Field(default_factory=list, description="Executive function friction points")
    ef_workarounds: List[str] = Field(default_factory=list, description="Executive function workarounds they've developed")
    sensory_profile: Optional[str] = Field(None, description="Sensory sensitivities, seeking behaviors, regulation methods")
    food_sensory_details: Optional[str] = Field(None, description="DETAILED food sensory profile: safe foods (specific brands/preparations), texture aversions, taste preferences, food-related challenges, how food impacts functioning. NOT generic 'pizza and nuggets' - be SPECIFIC about textures, temperatures, preparations.")


class CurrentLifeContext(BaseModel):
    where_they_live: Optional[str] = Field(None, description="Where they live and why (neighborhood, housing situation)")
    work_school_situation: Optional[str] = Field(None, description="Current work or school situation, environment, challenges")
    commute_daily_rhythms: Optional[str] = Field(None, description="Commute details, daily rhythms, routines")
    local_stressors: List[str] = Field(default_factory=list, description="St. Louis-specific or local stressors")
    safe_spaces: List[str] = Field(default_factory=list, description="Physical places they feel safe (can include specific St. Louis places if relevant, but NO tourist traps)")
    support_system: Optional[str] = Field(None, description="Who is actually there for them - friends, family, partners, community. Include gaps in support.")


class GoalsBarriers(BaseModel):
    why_therapy_now: Optional[str] = Field(None, description="Why they're seeking therapy NOW - the catalyst, urgency")
    what_they_want_to_change: List[str] = Field(default_factory=list, description="What they want to change or work on")
    whats_in_the_way: List[str] = Field(default_factory=list, description="Barriers - internal AND external combined (e.g. perfectionism, cost, stigma, time)")


class NarrativeVoice(BaseModel):
    quote_struggle: str = Field(..., description="Verbatim or inferred quote expressing their struggle")
    quote_hope: str = Field(..., description="Verbatim or inferred quote expressing hope or motivation")
    quotes_additional: List[str] = Field(default_factory=list, description="Additional quotes that capture their voice")
    liz_clinical_narrative: str = Field(..., description="150-250 word clinical narrative in Liz Wooten's compassionate, neurodivergent-affirming tone")
    therapy_preferences: Optional[str] = Field(None, description="Preferred and disliked therapy approaches, past experiences")


class MarketingSEO(BaseModel):
    website_copy_snippets: List[str] = Field(default_factory=list, description="2-3 website copy snippets that would resonate")
    primary_keywords: List[str] = Field(default_factory=list, description="Primary SEO keywords")
    local_entities: List[str] = Field(default_factory=list, description="Local entities for SEO (neighborhoods, institutions, landmarks - NO tourist traps)")
    content_angles: List[str] = Field(default_factory=list, description="2-3 content angles for blogs/articles")


class ClientProfileV2RealStories(BaseModel):
    meta: ProfileMeta
    identity_demographics: IdentityDemographics
    developmental_story: DevelopmentalStory
    neurodivergence_mental_health: NeurodivergenceMentalHealth
    executive_function_sensory: ExecutiveFunctionSensory
    current_life_context: CurrentLifeContext
    goals_barriers: GoalsBarriers
    narrative_voice: NarrativeVoice
    marketing_seo: MarketingSEO

