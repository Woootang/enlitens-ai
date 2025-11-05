import asyncio

import pytest
from typing import Optional

from src.agents.client_profile_agent import ClientProfileAgent
from src.orchestration.research_orchestrator import (
    ExternalResearchOrchestrator,
    NullConnector,
    StaticConnector,
)
from src.models.enlitens_schemas import ClientProfile, ClientProfileSet
from src.models.prediction_error import PredictionErrorEntry


COMMUTER_LOCALITIES = ["Central West End", "Tower Grove South", "Delmar Loop"]
COMMUTER_CONNECTIONS = ["Carondelet YMCA", "International Institute", "Pageant Community Room"]
CAREGIVER_LOCALITIES = ["Kirkwood", "Webster Groves", "Ferguson"]
CAREGIVER_CONNECTIONS = [
    "Kirkwood Community Center",
    "Webster Groves Recreation Complex",
    "Ferguson Community Empowerment Center",
]
SHIFT_LOCALITIES = ["Florissant", "Clayton", "Maplewood Richmond Heights"]
SHIFT_CONNECTIONS = [
    "James J. Eagan Center",
    "The Center of Clayton",
    "THE HEIGHTS Community Center",
]

_DEFAULT_MASKING = ["Commute shutdown", "Hypervigilant jaw clench"]
_DEFAULT_UNMET = [
    "Predictable decompression",
    "Transit sensory scouting",
    "Community accountability for rest",
]
_DEFAULT_SUPPORT = [
    "Schedule river overlook pauses",
    "Coordinate with Metro sensory hours",
    "Share scripts for requesting quiet cars",
]
_DEFAULT_FLAGS = ["Do not promise outcomes", "Fictional composite reminder"]


def _profile_template(
    *,
    name: str,
    intake: str,
    research: str,
    benefit: str,
    alignment: str,
    localities: list[str],
    connections: list[str],
    prediction_errors: list[PredictionErrorEntry],
    masking: Optional[list[str]] = None,
    unmet: Optional[list[str]] = None,
    support: Optional[list[str]] = None,
    flags: Optional[list[str]] = None,
) -> ClientProfile:
    persona_overview = (
        "Neighborhood & Daily Geography: "
        f"{localities[0]} anchors this FICTIONAL composite's week. "
        "Family & Intergenerational History: Fictional caregivers process legacy stress. "
        "Economic Context & Access Gaps: Highlights budget trade-offs without promising outcomes. "
        "Sensory & Community Experiences: Maps overstimulation across civic spaces. "
        "Local Supports (schools, leagues, churches, eateries): Draws on referenced community assets."
    )
    return ClientProfile(
        profile_name=name,
        fictional_disclaimer="FICTIONAL composite for internal research translation.",
        intake_reference=intake,
        persona_overview=persona_overview,
        research_reference=research,
        benefit_explanation=benefit,
        st_louis_alignment=alignment,
        local_geography=list(localities),
        community_connections=list(connections),
        regional_touchpoints=list(localities)[:3],
        masking_signals=list(masking or _DEFAULT_MASKING),
        unmet_needs=list(unmet or _DEFAULT_UNMET),
        support_recommendations=list(support or _DEFAULT_SUPPORT),
        cautionary_flags=list(flags or _DEFAULT_FLAGS),
        prediction_errors=prediction_errors,
    )


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


class _HealthyOllamaClient:
    def __init__(self, *, default_model):
        self.default_model = default_model

    async def check_connection(self):
        return True


@pytest.mark.asyncio
async def test_initialize_raises_when_only_null_research_connectors(monkeypatch):
    agent = ClientProfileAgent()

    monkeypatch.setattr(
        "src.agents.client_profile_agent.OllamaClient",
        _HealthyOllamaClient,
    )

    monkeypatch.setattr(
        ExternalResearchOrchestrator,
        "from_settings",
        classmethod(lambda cls: ExternalResearchOrchestrator(connectors=[NullConnector()])),
    )

    with pytest.raises(RuntimeError, match="NullConnector"):
        await agent.initialize()


@pytest.mark.asyncio
async def test_initialize_succeeds_with_real_research_connector(monkeypatch):
    agent = ClientProfileAgent()

    monkeypatch.setattr(
        "src.agents.client_profile_agent.OllamaClient",
        _HealthyOllamaClient,
    )

    monkeypatch.setattr(
        ExternalResearchOrchestrator,
        "from_settings",
        classmethod(lambda cls: ExternalResearchOrchestrator(connectors=[StaticConnector([])])),
    )

    assert await agent.initialize()


def _profiles_payload(source_tag: str = "[Source 1]") -> ClientProfileSet:
    return ClientProfileSet(
        profiles=[
            _profile_template(
                name="Sensory rail commuter",
                intake='"The red line makes my skin buzz"',
                research=f"{source_tag} outlines how chronic transit noise overloads sensory gating.",
                benefit=f"{source_tag} shows vestibular priming calms their commute shutdown spiral.",
                alignment=f"{source_tag} plus STL Metro complaints show we adapt local transit routines.",
                localities=COMMUTER_LOCALITIES,
                connections=COMMUTER_CONNECTIONS,
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
            _profile_template(
                name="After-school melter",
                intake='"By pickup time I am empty and shaking"',
                research=f"{source_tag} links cortisol spikes to afternoon classroom sensory debt.",
                benefit=f"{source_tag} backs co-regulation stops before homework to prevent crashes.",
                alignment=f"{source_tag} + northside school overstimulation data justify flexible pickups.",
                localities=CAREGIVER_LOCALITIES,
                connections=CAREGIVER_CONNECTIONS,
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
            _profile_template(
                name="Shift worker hypervigilance",
                intake='"Night sirens keep my jaw locked"',
                research=f"{source_tag} documents auditory hypervigilance during overnight shifts.",
                benefit=f"{source_tag} validates low-light decompression pods after each shift.",
                alignment=f"{source_tag} and STL hospital siren density explain the neighbourhood plan.",
                localities=SHIFT_LOCALITIES,
                connections=SHIFT_CONNECTIONS,
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


def test_ensure_minimum_items_replenishes_after_outcome_stripping():
    agent = ClientProfileAgent()

    duplicate_items = [
        "Clinically proven restful nights",
        "100% restful nights",
    ]
    duplicate_fallbacks = [
        "Cure restful mornings",
        "cure restful mornings",
        "CURE restful mornings",
    ]

    requirements = [
        ("regional_touchpoints", 3, 8),
        ("masking_signals", 2, 8),
        ("unmet_needs", 3, 8),
        ("support_recommendations", 3, 8),
        ("cautionary_flags", 2, 6),
    ]

    for field, minimum, maximum in requirements:
        result = agent._ensure_minimum_items(
            duplicate_items,
            fallback=duplicate_fallbacks,
            min_items=minimum,
            max_items=maximum,
        )
        assert len(result) >= minimum, f"{field} fell below minimum {minimum}"


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
    for profile in payload["profiles"]:
        assert "FICTIONAL" in profile["fictional_disclaimer"].upper()
        assert len(profile["local_geography"]) >= 3
        assert len(profile["community_connections"]) >= 3

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
        expected_primary = f"[Source {idx}]"
        expected_alt = f"[Source {idx % 3 + 1}]"
        assert expected_primary in profile.research_reference
        assert expected_alt in profile.benefit_explanation
        if profile.st_louis_alignment:
            assert expected_primary in profile.st_louis_alignment


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
    for profile in payload["profiles"]:
        assert "FICTIONAL" in profile["fictional_disclaimer"].upper()
        assert len(profile["local_geography"]) >= 3
        assert len(profile["community_connections"]) >= 3

    assert asyncio.run(agent.validate_output(result))


def test_post_process_profiles_backfills_missing_locality_data():
    agent = ClientProfileAgent()
    base_profiles = [
        _profile_template(
            name="Locality gap",
            intake='"I never know which neighborhood supports me"',
            research="[Source 1] explores local support gaps.",
            benefit="[Source 1] highlights neighborhood-based regulation pivots.",
            alignment="[Source 1] plus city data show where supports are thin.",
            localities=COMMUTER_LOCALITIES,
            connections=COMMUTER_CONNECTIONS,
            prediction_errors=[
                PredictionErrorEntry(
                    trigger_context="Assumes support only exists downtown.",
                    surprising_pivot="[Source 1] lifts up pocket resources in mixed-income corridors.",
                    intended_cognitive_effect="Invite the reader to map overlooked anchors.",
                ),
                PredictionErrorEntry(
                    trigger_context="Believes community hubs close after work hours.",
                    surprising_pivot="[Source 1] plus [Ext 2] show late-evening sensory rooms in libraries.",
                    intended_cognitive_effect="Spark curiosity about extended-hour assets.",
                ),
            ],
        ),
        _profile_template(
            name="Caretaker baseline",
            intake='"I juggle caregiver roles across the metro"',
            research="[Source 2] examines distributed caregiving networks.",
            benefit="[Source 2] validates rotating decompression rituals.",
            alignment="[Source 2] plus carpool data map mutual-aid corridors.",
            localities=CAREGIVER_LOCALITIES,
            connections=CAREGIVER_CONNECTIONS,
            prediction_errors=[
                PredictionErrorEntry(
                    trigger_context="Assumes regulation can only happen at home.",
                    surprising_pivot="[Source 2] documents park-based sensory resets on the drive home.",
                    intended_cognitive_effect="Invite the team to treat the commute as a co-regulation asset.",
                ),
                PredictionErrorEntry(
                    trigger_context="Believes caregivers must shoulder the load solo.",
                    surprising_pivot="[Source 2] plus [Ext 3] highlight neighborhood carpool circles sharing decompression.",
                    intended_cognitive_effect="Encourage distributed networks for after-school support.",
                ),
            ],
        ),
        _profile_template(
            name="Shift worker baseline",
            intake='"Night sirens keep my jaw locked"',
            research="[Source 3] documents auditory hypervigilance for night workers.",
            benefit="[Source 3] shows decompression pods reduce sympathetic spikes.",
            alignment="[Source 3] paired with hospital siren counts justify hospital pods.",
            localities=SHIFT_LOCALITIES,
            connections=SHIFT_CONNECTIONS,
            prediction_errors=[
                PredictionErrorEntry(
                    trigger_context="Assumes decompression can't happen on site.",
                    surprising_pivot="[Source 3] references chapels and dim rooms staff already use before commuting.",
                    intended_cognitive_effect="Encourage immediate decompression rituals before leaving work.",
                ),
                PredictionErrorEntry(
                    trigger_context="Believes every route home is equally loud.",
                    surprising_pivot="[Source 3] plus [Ext 4] map river-adjacent drives with fewer sirens.",
                    intended_cognitive_effect="Promote experimenting with quieter corridors.",
                ),
            ],
        ),
    ]

    missing_data = base_profiles[0].model_dump()
    missing_data.update(
        {
            "local_geography": [],
            "community_connections": [],
            "regional_touchpoints": [],
        }
    )
    base_profiles[0] = ClientProfile.model_construct(**missing_data)

    profile_set = ClientProfileSet(profiles=base_profiles)
    processed = agent._post_process_profiles(
        response=profile_set,
        document_id="doc-1",
        document_localities=[("Kirkwood", 4), ("The Ville", 2)],
        external_sources=[],
        regional_atlas={"stl_city_neighborhoods": ["The Ville", "Benton Park"]},
    )

    first = processed.profiles[0]
    assert len(first.local_geography) >= 3
    assert len(first.community_connections) >= 3
    assert len(set(first.local_geography)) >= 3
    assert len({conn.lower() for conn in first.community_connections}) >= 3


def test_post_process_profiles_expands_sparse_locality_data():
    agent = ClientProfileAgent()
    sparse_profile = _profile_template(
        name="Sparse locality",
        intake='"We only visit Kirkwood"',
        research="[Source 1] covers west county sensory hubs.",
        benefit="[Source 1] maps rotating decompression stops.",
        alignment="[Source 1] plus municipal data chart underused supports.",
        localities=CAREGIVER_LOCALITIES,
        connections=CAREGIVER_CONNECTIONS,
        prediction_errors=[
            PredictionErrorEntry(
                trigger_context="Assumes only one neighborhood feels accessible.",
                surprising_pivot="[Source 1] reveals overlapping supports in adjacent municipalities.",
                intended_cognitive_effect="Encourage venturing beyond the default routine.",
            ),
            PredictionErrorEntry(
                trigger_context="Believes bilingual support groups are rare.",
                surprising_pivot="[Ext 2] documents Spanish-language circles at local community centers.",
                intended_cognitive_effect="Highlight community stewardship in nearby towns.",
            ),
        ],
    )
    sparse_data = sparse_profile.model_dump()
    sparse_data.update(
        {
            "local_geography": ["Kirkwood"],
            "community_connections": ["Kirkwood Community Center"],
        }
    )
    profile = ClientProfile.model_construct(**sparse_data)
    profile_set = ClientProfileSet(
        profiles=[profile, _profiles_payload().profiles[1], _profiles_payload().profiles[2]]
    )

    processed = agent._post_process_profiles(
        response=profile_set,
        document_id="doc-2",
        document_localities=[("Kirkwood", 5), ("Webster Groves", 3)],
        external_sources=[],
        regional_atlas={"stl_city_neighborhoods": ["Maplewood"], "stl_county_municipalities": ["Clayton"]},
    )

    updated = processed.profiles[0]
    assert "Kirkwood" in updated.local_geography
    assert len(updated.local_geography) >= 3
    assert "Kirkwood Community Center" in updated.community_connections
    assert len(updated.community_connections) >= 3


def test_post_process_profiles_preserves_rich_locality_data():
    agent = ClientProfileAgent()
    rich_profile = _profile_template(
        name="Rich data",
        intake='"Our week spans multiple neighborhoods"',
        research="[Source 1] analyses multi-site sensory strategies.",
        benefit="[Source 1] backs rotational support planning.",
        alignment="[Source 1] with census data shows cross-municipality routines.",
        localities=["Central West End", "Tower Grove South", "Delmar Loop", "Ferguson"],
        connections=[
            "Carondelet YMCA",
            "International Institute",
            "Pageant Community Room",
            "Ferguson Community Empowerment Center",
        ],
        prediction_errors=[
            PredictionErrorEntry(
                trigger_context="Assumes the same three locales must carry all support.",
                surprising_pivot="[Source 1] highlights rotating micro-rituals across cultural districts.",
                intended_cognitive_effect="Encourage diversifying weekly anchors.",
            ),
            PredictionErrorEntry(
                trigger_context="Believes commuting between neighborhoods adds stress.",
                surprising_pivot="[Ext 3] maps quiet routes that turn travel into decompression time.",
                intended_cognitive_effect="Reframe travel as restorative sequencing.",
            ),
        ],
    )

    profile_set = ClientProfileSet(
        profiles=[rich_profile, _profiles_payload().profiles[0], _profiles_payload().profiles[1]]
    )

    processed = agent._post_process_profiles(
        response=profile_set,
        document_id="doc-3",
        document_localities=[("Central West End", 4)],
        external_sources=[],
        regional_atlas={"stl_city_neighborhoods": ["The Ville"]},
    )

    enriched = processed.profiles[0]
    assert enriched.local_geography[:4] == [
        "Central West End",
        "Tower Grove South",
        "Delmar Loop",
        "Ferguson",
    ]
    assert enriched.community_connections[:4] == [
        "Carondelet YMCA",
        "International Institute",
        "Pageant Community Room",
        "Ferguson Community Empowerment Center",
    ]
