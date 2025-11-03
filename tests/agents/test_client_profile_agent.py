import asyncio

import pytest

from src.agents.client_profile_agent import ClientProfileAgent
from src.models.enlitens_schemas import ClientProfile, ClientProfileSet


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
            ),
            ClientProfile(
                profile_name="After-school melter",
                intake_reference='"By pickup time I am empty and shaking"',
                research_reference=f"{source_tag} links cortisol spikes to afternoon classroom sensory debt.",
                benefit_explanation=f"{source_tag} backs co-regulation stops before homework to prevent crashes.",
                st_louis_alignment=f"{source_tag} + northside school overstimulation data justify flexible pickups.",
            ),
            ClientProfile(
                profile_name="Shift worker hypervigilance",
                intake_reference='"Night sirens keep my jaw locked"',
                research_reference=f"{source_tag} documents auditory hypervigilance during overnight shifts.",
                benefit_explanation=f"{source_tag} validates low-light decompression pods after each shift.",
                st_louis_alignment=f"{source_tag} and STL hospital siren density explain the neighbourhood plan.",
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


def test_client_profile_agent_validate_output_enforces_citations():
    agent = ClientProfileAgent()

    good_payload = {"client_profiles": _profiles_payload().model_dump()}
    assert asyncio.run(agent.validate_output(good_payload))

    bad_profiles = _profiles_payload().model_dump()
    bad_profiles["profiles"][0]["benefit_explanation"] = "Missing citation"

    with pytest.raises(ValueError):
        asyncio.run(agent.validate_output({"client_profiles": bad_profiles}))
