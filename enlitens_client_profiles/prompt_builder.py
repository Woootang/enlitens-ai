"""Prompt templates for generating rich client personas."""

from __future__ import annotations

import textwrap
from typing import Dict, List, Optional, Sequence


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an ethnographic analyst preparing richly detailed therapy client personas for Enlitens, a
    neuroscience-forward, trauma-informed therapy practice in the Greater St. Louis region. Every profile must:

    - Anchor to specific municipalities, neighborhoods, and corridors in and around St. Louis City, St. Louis County,
      metro-east Illinois, Jefferson County, and St. Charles County.
    - Use the authentic language and emotional register clients share in their intake inquiries and phone transcripts.
    - Reflect socioeconomic texture (industry, commute, family system, cultural and faith influences) while staying
      empathetic and strengths-oriented.
    - Surface nervous-system patterns, executive function dynamics, sensory environments, and barriers to access.
    - Produce actionable marketing and SEO signals that map to real human search intent (no generic business jargon).

    Avoid repetition across profiles. Lean into nuance, dialect, and situational detail. If data is missing, infer
    plausibly from similar cases and public health context but clearly mark inferred vs verbatim signals. Cite source
    snippets when using direct quotes.
    """
).strip()


def build_profile_prompt(
    *,
    intake_samples: Sequence[str],
    transcript_samples: Sequence[str],
    health_insights: str,
    knowledge_assets: Dict[str, str],
    geo_reference: Dict[str, List[str]],
    site_map_context: str,
    brand_site_context: str,
    brand_mentions_context: str,
    locality_counts: Dict[str, int],
    analytics_summary: str = "",
    analytics_lookback_days: Optional[int] = None,
    foundation_summary: str = "",
    research_summary: str = "",
) -> str:
    """Compose the LLM brief for generating a persona document."""

    intake_block = "\n\n".join(intake_samples)
    transcript_block = "\n\n".join(transcript_samples)
    knowledge_block = "\n".join(
        f"### {name}\n{content[:600]}" for name, content in list(knowledge_assets.items())[:2]
    )
    geo_summary = "\n".join(
        f"- {bucket.replace('_', ' ').title()}: {', '.join(values[:12])}" for bucket, values in geo_reference.items()
    )
    locality_lines = "\n".join(
        f"- {municipality}: {count}" for municipality, count in sorted(locality_counts.items(), key=lambda item: -item[1])[:10]
    )

    analytics_label = "analytics snapshot unavailable"
    analytics_block = analytics_summary.strip()
    if analytics_block and analytics_lookback_days:
        analytics_label = f"{analytics_lookback_days}-day lookback"
    elif analytics_block:
        analytics_label = "recent analytics snapshot"
    else:
        analytics_block = "Analytics snapshot unavailable"

    foundation_block = foundation_summary.strip() if foundation_summary else "Foundation scaffold unavailable"
    research_block = research_summary.strip() if research_summary else "Research agent did not return additional findings"

    prompt = f"""
    === ENLITENS CLIENT PERSONA BRIEF ===

    ## Intake Snippets (verbatim client language)
    {intake_block}

    ## Liz Wooten Voice Samples
    {transcript_block}

    ## Regional Public Health Insights (excerpt)
    {health_insights[:1400]}

    ## St. Louis Locality Mentions (frequency in recent intakes)
    {locality_lines if locality_lines else 'N/A'}

    ## St. Louis Regional Geography Anchors
    {geo_summary}

    ## Enlitens Website Snapshot (tone reference)
    {site_map_context}

    ## Enlitens Voice & Brand Anchors (site corpus synthesis)
    {brand_site_context}

    ## External Enlitens Mentions & Signals
    {brand_mentions_context if brand_mentions_context else 'No external mentions captured this cycle'}

    ## Additional Knowledge Assets
    {knowledge_block}

    ## Website Analytics + Search Intent Signals ({analytics_label})
    {analytics_block}

    ## Persona Foundation Skeleton & Known Gaps
    {foundation_block}

    ## Deep Research Findings (external sources)
    {research_block}

    TASK: Produce a single JSON object matching the following schema keys. Every field must be filled with
    detailed, human-specific content, balancing direct quotes (prefix entries with "[direct]") and
    inferred synthesis (prefix entries with "[inferred]").

    - meta: profile_id (slug), persona_name, persona_tagline, attribute_tags, and source_documents referencing
      which snippets informed the profile.
    - demographics: age_range, gender, pronouns, orientation, ethnicity, family_status, occupation, education, locality.
    - neurodivergence_profile: identities (list), diagnosis_notes (story of discovery), language_preferences.
    - clinical_challenges: presenting_issues (list), nervous_system_pattern (Liz-style explanation), mood_patterns,
      trauma_history (if relevant).
    - adaptive_strengths: strengths (list framed as superpowers) and coping_skills already in use.
    - executive_function: strengths, friction_points, workarounds tied to daily routines.
    - sensory_profile: sensitivities, seeking_behaviors, regulation_methods grounded in lived detail.
    - goals_motivations: therapy_goals, life_goals, motivations, why_now.
    - pain_points_barriers: internal narratives, systemic barriers, access constraints (financial/scheduling/transport).
    - cultural_context: identities, community_roles, notes around culture/faith/local identity.
    - local_environment: home_environment, work_environment, commute, local_stressors, safe_spaces (include STL-specific references).
    - support_system: household summary, supportive_allies, caregiving_roles, gaps.
    - tech_media_habits: platforms, communication_preferences, content_styles (omit or keep empty arrays if unknown).
    - therapy_preferences: preferred_styles, disliked_approaches, past_experiences (if unknown, supply thoughtful inference and tag with [inferred]).
    - quotes: struggle and hope quotes (verbatim client language) plus additional quotes, including [direct] or [inferred] prefix.
    - narrative: liz_voice 150-220 word narrative written in Liz Wooten's tone, highlighting nervous-system reframes,
      sensory context, and local anchors; include 3-5 bullet highlights.
    - marketing_copy: website_about, landing_page_intro, email_nurture, social_snippets (each list must contain
      2-3 pieces of copy, with local and neurobiological framing).
    - seo_brief: primary_keywords, long_tail_keywords, local_entities, content_angles (minimum 5 each, grounded in persona language).
    - resources: recommended_offers (Enlitens services/groups) and referral_needs (external supports they often need).
    - analytics: coverage_notes (how this persona expands reach) and similarity_fingerprint placeholder (string hash or descriptor).

    STRICT REQUIREMENTS:
    - Use specific municipalities/neighborhoods at least twice across narrative and marketing_copy.
    - Keep tone neurodiversity-affirming, strengths-forward, slightly irreverent in Liz's style.
    - Avoid generic corporate jargon; root phrases in the intake samples.
    - Where data is inferred, include the "[inferred]" prefix in the string; where verbatim, use "[direct]".
    - Every list-type field must be a JSON array with at least two thoughtfully written entries (unless the schema specifies a single string field).
    - Output must be valid UTF-8 JSON with double-quoted keys, no comments, and no trailing commas.
    """

    return textwrap.dedent(prompt).strip()

