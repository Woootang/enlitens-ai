"""Pydantic schema definitions for Enlitens client personas."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, root_validator, validator


class ProfileMeta(BaseModel):
    """Metadata attached to every persona profile."""

    profile_id: str = Field(..., description="Stable unique identifier (e.g. slug)")
    persona_name: str = Field(..., description="Human-friendly persona name, e.g. 'Overachiever Olivia'")
    persona_tagline: Optional[str] = Field(None, description="Short descriptor used in dashboards or UI")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source_documents: List[str] = Field(default_factory=list, description="Document IDs or filenames used during generation")
    llm_model: Optional[str] = Field(None, description="Model used for generation")
    version: str = Field(default="1.0")
    attribute_tags: List[str] = Field(
        default_factory=list,
        description="Key attribute tags used for deduplication and routing (e.g. 'adhd', 'young_adult').",
    )


class Demographics(BaseModel):
    age_range: Optional[str] = Field(None, description="Age range or stage (e.g. 'late 20s')")
    gender: Optional[str] = None
    pronouns: Optional[str] = None
    orientation: Optional[str] = None
    ethnicity: Optional[str] = None
    family_status: Optional[str] = None
    occupation: Optional[str] = None
    education: Optional[str] = None
    locality: Optional[str] = Field(None, description="High-level locale reference (e.g. 'North County - Florissant')")


class NeurodivergenceProfile(BaseModel):
    identities: List[str] = Field(default_factory=list, description="Formal or self-identified neurodivergent identities")
    diagnosis_notes: Optional[str] = Field(None, description="Diagnosis journey or context")
    language_preferences: List[str] = Field(
        default_factory=list, description="Preferred language about their neurotype (identity-first, strengths framing, etc.)"
    )


class ClinicalChallenges(BaseModel):
    presenting_issues: List[str] = Field(default_factory=list, description="Salient challenges described by the persona")
    nervous_system_pattern: str = Field(..., description="Plain-language explanation of nervous system tendencies")
    mood_patterns: List[str] = Field(default_factory=list, description="Observed mood patterns (e.g. shutdown, hypervigilance)")
    trauma_history: Optional[str] = Field(None, description="Notable trauma context or notes if disclosed")


class AdaptiveStrengths(BaseModel):
    strengths: List[str] = Field(default_factory=list, description="Reframed adaptive strengths and superpowers")
    coping_skills: List[str] = Field(default_factory=list, description="Existing coping skills that can be leveraged")


class ExecutiveFunctionProfile(BaseModel):
    strengths: List[str] = Field(default_factory=list)
    friction_points: List[str] = Field(default_factory=list)
    workarounds: List[str] = Field(default_factory=list, description="Known strategies the persona already uses")


class SensoryProfile(BaseModel):
    sensitivities: List[str] = Field(default_factory=list)
    seeking_behaviors: List[str] = Field(default_factory=list)
    regulation_methods: List[str] = Field(default_factory=list, description="Preferred regulation practices or tools")


class GoalsMotivations(BaseModel):
    therapy_goals: List[str] = Field(default_factory=list)
    life_goals: List[str] = Field(default_factory=list)
    motivations: List[str] = Field(default_factory=list, description="Underlying values or motivations")
    why_now: Optional[str] = Field(None, description="Trigger that led them to seek support now")


class PainPointsBarriers(BaseModel):
    internal: List[str] = Field(default_factory=list, description="Internal narratives or blockers")
    systemic: List[str] = Field(default_factory=list, description="Systemic or cultural barriers")
    access: List[str] = Field(default_factory=list, description="Access barriers (time, cost, transport)")


class CulturalCommunityContext(BaseModel):
    identities: List[str] = Field(default_factory=list, description="Cultural, faith, or community identities")
    community_roles: List[str] = Field(default_factory=list, description="Roles they play in community or family")
    notes: Optional[str] = Field(None, description="Additional cultural considerations")


class LocalEnvironment(BaseModel):
    home_environment: Optional[str] = Field(None, description="Summary of home setting dynamics")
    work_environment: Optional[str] = Field(None, description="Summary of work or school context")
    commute: Optional[str] = Field(None, description="Typical commute or mobility pattern")
    local_stressors: List[str] = Field(default_factory=list, description="Local or environmental stressors")
    safe_spaces: List[str] = Field(default_factory=list, description="Places they go for regulation or community")


class SupportSystem(BaseModel):
    household: Optional[str] = Field(None, description="Who they live with or primary household structure")
    supportive_allies: List[str] = Field(default_factory=list)
    caregiving_roles: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list, description="Missing supports or relational ruptures")


class TechMediaHabits(BaseModel):
    platforms: List[str] = Field(default_factory=list, description="Digital platforms they frequent")
    communication_preferences: List[str] = Field(default_factory=list, description="Preferred communication modes")
    content_styles: List[str] = Field(default_factory=list, description="Content styles that resonate (memes, longform, etc.)")


class TherapyPreferences(BaseModel):
    preferred_styles: List[str] = Field(default_factory=list, description="Therapy styles or modalities they gravitate toward")
    disliked_approaches: List[str] = Field(default_factory=list, description="Approaches that have not worked or feel invalidating")
    past_experiences: Optional[str] = Field(None, description="Summary of prior therapy experiences")


class QuoteSet(BaseModel):
    struggle: str = Field(..., description="Direct quote capturing their struggle")
    hope: str = Field(..., description="Direct quote capturing their hope or goal")
    additional: List[str] = Field(default_factory=list, description="Any additional supporting quotes")


class NarrativeSummary(BaseModel):
    liz_voice: str = Field(..., description="Narrative written in Liz Wooten's tone summarising the persona")
    highlights: List[str] = Field(default_factory=list, description="Key beats or moments to emphasize in storytelling")


class MarketingCopy(BaseModel):
    website_about: List[str] = Field(default_factory=list)
    landing_page_intro: List[str] = Field(default_factory=list)
    email_nurture: List[str] = Field(default_factory=list)
    social_snippets: List[str] = Field(default_factory=list)


class SEOBrief(BaseModel):
    primary_keywords: List[str] = Field(default_factory=list)
    long_tail_keywords: List[str] = Field(default_factory=list)
    local_entities: List[str] = Field(default_factory=list)
    content_angles: List[str] = Field(default_factory=list)


class PersonaResources(BaseModel):
    recommended_offers: List[str] = Field(default_factory=list, description="Services, groups, or products to recommend")
    referral_needs: List[str] = Field(default_factory=list, description="External referrals that commonly support this persona")


class PersonaAnalytics(BaseModel):
    coverage_notes: Optional[str] = Field(None, description="Notes about representation/coverage for dashboards")
    similarity_fingerprint: Optional[str] = Field(
        None, description="Hash or fingerprint generated by similarity checker for deduplication"
    )


class ClientProfileDocument(BaseModel):
    """Complete persona document used across Enlitens workflows."""

    meta: ProfileMeta
    demographics: Demographics
    neurodivergence_profile: NeurodivergenceProfile
    clinical_challenges: ClinicalChallenges
    adaptive_strengths: AdaptiveStrengths
    executive_function: ExecutiveFunctionProfile
    sensory_profile: SensoryProfile
    goals_motivations: GoalsMotivations
    pain_points_barriers: PainPointsBarriers
    cultural_context: CulturalCommunityContext
    local_environment: LocalEnvironment
    support_system: SupportSystem
    tech_media_habits: Optional[TechMediaHabits] = None
    therapy_preferences: Optional[TherapyPreferences] = None
    quotes: QuoteSet
    narrative: NarrativeSummary
    marketing_copy: MarketingCopy
    seo_brief: SEOBrief
    resources: PersonaResources = Field(default_factory=PersonaResources)
    analytics: PersonaAnalytics = Field(default_factory=PersonaAnalytics)

    @root_validator(skip_on_failure=True)
    def _ensure_minimum_content(cls, values: Dict[str, object]) -> Dict[str, object]:
        """Guard against empty critical sections."""

        strengths: AdaptiveStrengths = values.get("adaptive_strengths")  # type: ignore[assignment]
        quotes: QuoteSet = values.get("quotes")  # type: ignore[assignment]

        if strengths and not strengths.strengths:
            raise ValueError("adaptive_strengths.strengths must include at least one item")

        if not quotes.struggle or not quotes.hope:
            raise ValueError("quotes must include both struggle and hope entries")

        return values

    def slug(self) -> str:
        return self.meta.profile_id

    def attribute_set(self) -> List[str]:
        """Return a deduplication-friendly set of attributes."""

        tags = set(self.meta.attribute_tags)

        if self.demographics.age_range:
            tags.add(f"age:{self.demographics.age_range}")
        if self.demographics.gender:
            tags.add(f"gender:{self.demographics.gender}")
        if self.demographics.locality:
            tags.add(f"locality:{self.demographics.locality}")

        for identity in self.neurodivergence_profile.identities:
            tags.add(f"identity:{identity.lower()}")

        for issue in self.clinical_challenges.presenting_issues:
            tags.add(f"challenge:{issue.lower()}")

        return sorted(tags)

