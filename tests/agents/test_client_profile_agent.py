import asyncio

import pytest

from src.agents.client_profile_agent import ClientProfileAgent
from src.models.enlitens_schemas import ClientProfile, ClientProfileSet
from src.models.prediction_error import PredictionErrorEntry


class _RecordingOllamaClient:
    def __init__(self, responses):
        self.responses = responses
        self.prompts = []

    async def generate_structured_response(self, *, prompt, response_model, **kwargs):  # type: ignore[override]
        self.prompts.append(prompt)
        factory = self.responses.get(response_model)
        if callable(factory):
            return factory()
        return factory


def _profiles_payload(source_tag: str = "[Source 1]") -> ClientProfileSet:
    return ClientProfileSet(
        profiles=[
            ClientProfile(
                profile_name="Sensory rail commuter",
                intake_reference='"The red line makes my skin buzz"',
                research_reference=f"{source_tag} outlines how chronic transit noise overloads sensory gating.",
                benefit_explanation=f"{source_tag} shows vestibular priming calms their commute shutdown spiral.",
                st_louis_alignment=f"{source_tag} plus STL Metro complaints show we adapt local transit routines.",
                prediction_errors=[
                    PredictionErrorEntry(
                        trigger_context="Assumes the Metro commute is only depleting.",
                        surprising_pivot=f"{source_tag} cites riders who recharge by stopping at the quiet Riverfront Trail lookout.",
                        intended_cognitive_effect="Invite the team to scout calming micro-routines within public transit.",
                    ),
                    PredictionErrorEntry(
                        trigger_context="Believes noise-cancelling headphones are the only option.",
                        surprising_pivot=f"{source_tag} plus [Ext 1] show library sensory hours align with commute schedules.",
                        intended_cognitive_effect="Spark curiosity about civic spaces that extend the commute decompression arc.",
                    ),
                ],
            ),
            ClientProfile(
                profile_name="After-school melter",
                intake_reference='"By pickup time I am empty and shaking"',
                research_reference=f"{source_tag} links cortisol spikes to afternoon classroom sensory debt.",
                benefit_explanation=f"{source_tag} backs co-regulation stops before homework to prevent crashes.",
                st_louis_alignment=f"{source_tag} + northside school overstimulation data justify flexible pickups.",
                prediction_errors=[
                    PredictionErrorEntry(
                        trigger_context="Assumes after-school meltdown relief has to happen at home.",
                        surprising_pivot=f"{source_tag} cites playground sensory diets at Fairground Park that defuse cortisol spikes.",
                        intended_cognitive_effect="Encourage designers to see the pickup route as an intervention zone.",
                    ),
                    PredictionErrorEntry(
                        trigger_context="Believes teachers control the entire afternoon regulation arc.",
                        surprising_pivot=f"{source_tag} plus [Ext 2] highlight mutual-aid carpool pods sharing decompression roles.",
                        intended_cognitive_effect="Shift the frame toward distributed caregiver ecosystems.",
                    ),
                ],
            ),
            ClientProfile(
                profile_name="Shift worker hypervigilance",
                intake_reference='"Night sirens keep my jaw locked"',
                research_reference=f"{source_tag} documents auditory hypervigilance during overnight shifts.",
                benefit_explanation=f"{source_tag} validates low-light decompression pods after each shift.",
                st_louis_alignment=f"{source_tag} and STL hospital siren density explain the neighbourhood plan.",
                prediction_errors=[
                    PredictionErrorEntry(
                        trigger_context="Assumes post-shift shutdown must happen at home.",
                        surprising_pivot=f"{source_tag} references hospital chapels offering sensory dimming before the drive home.",
                        intended_cognitive_effect="Open the reader to on-site decompression rituals.",
                    ),
                    PredictionErrorEntry(
                        trigger_context="Believes siren exposure is unavoidable in South City nights.",
                        surprising_pivot=f"{source_tag} plus [Ext 3] map quieter river routes for night-shift commutes.",
                        intended_cognitive_effect="Prompt planning for alternate sensory corridors.",
                    ),
                ],
            ),
        ],
        shared_thread="Research-backed sensory relief for St. Louis nervous systems",
    )


def test_client_profile_agent_prompt_references_intake_and_sources():
    agent = ClientProfileAgent()
    client = _RecordingOllamaClient({ClientProfileSet: _profiles_payload})
    agent.ollama_client = client

    context = {
        "retrieved_passages": [
            {
                "text": "[Source 1] Elevated siren exposure in St. Louis correlates with autonomic dysregulation.",
                "chunk_id": "chunk-1",
                "document_id": "doc-ctx",
            }
        ],
        "client_insights": {"direct_quotes": ['"Night sirens keep my jaw locked"']},
        "raw_client_context": 'Client said "The red line makes my skin buzz" during intake.',
        "st_louis_context": {"neighbourhoods": ["The Ville"], "stressors": ["sirens"]},
    }

    result = asyncio.run(agent.process(context))

    assert client.prompts, "Expected prompt to be recorded"
    prompt = client.prompts[0]
    assert "sirens" in prompt
    assert "[Source 1]" in prompt

    payload = result["client_profiles"]
    assert len(payload["profiles"]) == 3
    for profile in payload["profiles"]:
        assert "[Source 1]" in profile["research_reference"]
        assert "[Source 1]" in profile["benefit_explanation"]
        assert '"' in profile["intake_reference"]
        assert len(profile["prediction_errors"]) >= 2


def test_client_profile_agent_validate_output_enforces_citations():
    agent = ClientProfileAgent()

    good_payload = {"client_profiles": _profiles_payload().model_dump()}
    assert asyncio.run(agent.validate_output(good_payload))

    bad_profiles = _profiles_payload().model_dump()
    bad_profiles["profiles"][0]["benefit_explanation"] = "Missing citation"

    with pytest.raises(ValueError):
        asyncio.run(agent.validate_output({"client_profiles": bad_profiles}))


def test_client_profile_agent_normalizes_fragmented_partial_payload():
    agent = ClientProfileAgent()

    partial_payload = {
        "shared_thread": "Threaded",
        "profiles": [
            {
                "profile_name": "Sensory rail commuter",
                "intake_reference": '"The red line makes my skin buzz"',
                "research_reference": "[Source 1] Transit overload evidence.",
                "benefit_explanation": "[Source 1] Tailored commute regulation.",
                "st_louis_alignment": "[Source 1] Metro tie-in.",
            },
            "profile_name",
        ],
        "profiles.1.profile_name": "After-school melter",
        "profiles.1.intake_reference": '"By pickup time I am empty and shaking"',
        "profiles.1.research_reference": "[Source 1] Afternoon cortisol guidance.",
        "profiles.1.benefit_explanation": "[Source 1] Homework decompression plan.",
        "profiles.1.st_louis_alignment": "[Source 1] Northside school realities.",
        "profiles[2].profile_name": "Shift worker hypervigilance",
        "profiles[2].intake_reference": '"Night sirens keep my jaw locked"',
        "profiles[2].research_reference": "[Source 1] Overnight siren data.",
        "profiles[2].benefit_explanation": "[Source 1] Low-light decompression pods.",
        "profiles[2].st_louis_alignment": "[Source 1] STL hospital siren map.",
    }

    client = _RecordingOllamaClient({ClientProfileSet: lambda: partial_payload})
    agent.ollama_client = client

    context = {
        "retrieved_passages": [
            {
                "text": "[Source 1] Elevated siren exposure in St. Louis correlates with autonomic dysregulation.",
                "chunk_id": "chunk-1",
                "document_id": "doc-ctx",
            }
        ],
        "client_insights": {"direct_quotes": ['"Night sirens keep my jaw locked"']},
        "raw_client_context": 'Client said "The red line makes my skin buzz" during intake.',
        "st_louis_context": {"neighbourhoods": ["The Ville"], "stressors": ["sirens"]},
    }

    result = asyncio.run(agent.process(context))
    payload = result["client_profiles"]

    assert len(payload["profiles"]) == 3
    assert payload["profiles"][1]["profile_name"] == "After-school melter"
    assert payload["profiles"][2]["profile_name"] == "Shift worker hypervigilance"

    assert asyncio.run(agent.validate_output(result))


def test_client_profile_agent_backfills_missing_citations_from_retrieval():
    agent = ClientProfileAgent()

    citation_free_payload = {
        "shared_thread": "Threaded",
        "profiles": [
            {
                "profile_name": "Sensory rail commuter",
                "intake_reference": '"The red line makes my skin buzz"',
                "research_reference": "Transit overload evidence.",
                "benefit_explanation": "Tailored commute regulation.",
                "st_louis_alignment": "Metro tie-in for STL riders.",
            },
            {
                "profile_name": "After-school melter",
                "intake_reference": '"By pickup time I am empty and shaking"',
                "research_reference": "Afternoon cortisol guidance.",
                "benefit_explanation": "Homework decompression plan.",
                "st_louis_alignment": "Northside school realities.",
            },
            {
                "profile_name": "Shift worker hypervigilance",
                "intake_reference": '"Night sirens keep my jaw locked"',
                "research_reference": "Overnight siren data.",
                "benefit_explanation": "Low-light decompression pods.",
                "st_louis_alignment": "STL hospital siren map.",
            },
        ],
    }

    client = _RecordingOllamaClient({ClientProfileSet: lambda: citation_free_payload})
    agent.ollama_client = client

    context = {
        "retrieved_passages": [
            {"text": "Passage A", "chunk_id": "chunk-1", "document_id": "doc-1"},
            {"text": "Passage B", "chunk_id": "chunk-2", "document_id": "doc-2"},
            {"text": "Passage C", "chunk_id": "chunk-3", "document_id": "doc-3"},
        ],
        "client_insights": {"direct_quotes": ['"Night sirens keep my jaw locked"']},
        "raw_client_context": 'Client said "The red line makes my skin buzz" during intake.',
        "st_louis_context": {"neighbourhoods": ["The Ville"], "stressors": ["sirens"]},
    }

    result = asyncio.run(agent.process(context))
    payload = result["client_profiles"]

    validated = ClientProfileSet.model_validate(payload)
    for idx, profile in enumerate(validated.profiles, start=1):
        expected_tag = f"[Source {idx}]"
        assert expected_tag in profile.research_reference
        assert expected_tag in profile.benefit_explanation
        if profile.st_louis_alignment:
            assert expected_tag in profile.st_louis_alignment


def test_client_profile_agent_falls_back_when_partial_payload_invalid():
    agent = ClientProfileAgent()

    malformed_payload = {
        "profiles": ["profile_name", "intake_reference", "research_reference"],
        "shared_thread": "garbled",
    }

    client = _RecordingOllamaClient({ClientProfileSet: lambda: malformed_payload})
    agent.ollama_client = client

    context = {
        "retrieved_passages": [
            {
                "text": "[Source 1] Elevated siren exposure in St. Louis correlates with autonomic dysregulation.",
                "chunk_id": "chunk-1",
                "document_id": "doc-ctx",
            }
        ],
        "client_insights": {"direct_quotes": ['"Night sirens keep my jaw locked"']},
        "raw_client_context": 'Client said "The red line makes my skin buzz" during intake.',
        "st_louis_context": {"neighbourhoods": ["The Ville"], "stressors": ["sirens"]},
    }

    result = asyncio.run(agent.process(context))
    payload = result["client_profiles"]

    assert len(payload["profiles"]) == 3
    assert payload["profiles"][0]["profile_name"] == "Transit sensory overload"

    assert asyncio.run(agent.validate_output(result))
