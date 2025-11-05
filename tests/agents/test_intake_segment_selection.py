from src.agents.client_profile_agent import ClientProfileAgent


def test_select_relevant_intake_segments_prioritizes_demographic_facets():
    agent = ClientProfileAgent()

    intake_registry = {
        "entries": [
            {
                "snippet": "As a mother balancing overnight hospital shifts, I lean on our church pantry and feel exhausted caring for my son.",
                "themes": ["caregiver fatigue"],
            },
            {
                "snippet": "My teenager lights up at the library coding club and weekend gaming meetups, which keeps us hopeful.",
                "themes": ["youth engagement"],
            },
            {
                "snippet": "Our rural route lacks transit and neighbors check in at the feed store after tending the pasture.",
                "themes": ["transport barriers"],
            },
            {
                "snippet": "Downtown loft life means constant sirens outside the coworking hub and I stay anxious around coworkers.",
                "themes": ["sensory stress"],
            },
        ]
    }

    segments = agent._select_relevant_intake_segments(
        intake_registry=intake_registry,
        document_localities=[("St. Louis", 4)],
        retrieved_passages=[],
        fallback_quotes=[],
        max_segments=4,
    )

    assert len(segments) == 4
    coverage_union = {tag for segment in segments for tag in segment.get("coverage_tags", [])}
    for expected in (
        "family_role",
        "employment_role",
        "community_institution",
        "hobby_interest",
        "emotional_tone",
    ):
        assert expected in coverage_union
    assert any("rural_cue" in segment.get("coverage_tags", []) for segment in segments)
    assert any("urban_cue" in segment.get("coverage_tags", []) for segment in segments)


def test_select_relevant_intake_segments_preserves_locale_diversity_with_limited_slots():
    agent = ClientProfileAgent()

    intake_registry = {
        "entries": [
            {
                "snippet": "Long-form background about generalized paperwork that barely references people but stretches many sentences to add length for scoring.",
                "themes": ["paperwork"],
            },
            {
                "snippet": "Rural caregivers on the county road trade childcare while tending barns.",
                "themes": ["mutual aid"],
            },
            {
                "snippet": "Urban apartment roommates dodge sirens after late-night shifts downtown.",
                "themes": ["night work"],
            },
            {
                "snippet": "Families gather at the community garden to share recipes and art projects together.",
                "themes": ["community"],
            },
        ]
    }

    segments = agent._select_relevant_intake_segments(
        intake_registry=intake_registry,
        document_localities=[("St. Louis", 2)],
        retrieved_passages=[],
        fallback_quotes=[],
        max_segments=3,
    )

    assert len(segments) == 3
    assert any(segment.get("contextual_cues", {}).get("rural") for segment in segments)
    assert any(segment.get("contextual_cues", {}).get("urban") for segment in segments)
    tags_union = {tag for segment in segments for tag in segment.get("coverage_tags", [])}
    assert "community_institution" in tags_union or "family_role" in tags_union
