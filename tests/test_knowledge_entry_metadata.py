import sys
import types
from pathlib import Path

import asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Optional

from src.models.enlitens_schemas import ClientProfile, ClientProfileSet
from src.models.prediction_error import PredictionErrorEntry


# Provide minimal stubs for optional dependencies used during module import
langgraph_module = types.ModuleType("langgraph")
langgraph_graph_module = types.ModuleType("langgraph.graph")


class _StubStateGraph:
    def __init__(self, *_args, **_kwargs):
        pass

    def add_node(self, *_args, **_kwargs):
        return None

    def set_entry_point(self, *_args, **_kwargs):
        return None

    def add_edge(self, *_args, **_kwargs):
        return None

    def compile(self):
        return self


langgraph_graph_module.StateGraph = _StubStateGraph
langgraph_graph_module.END = object()
langgraph_module.graph = langgraph_graph_module

sys.modules.setdefault("langgraph", langgraph_module)
sys.modules.setdefault("langgraph.graph", langgraph_graph_module)


def _register_stub(path: str, **attributes):
    module = types.ModuleType(path)
    for name, value in attributes.items():
        setattr(module, name, value)
    sys.modules.setdefault(path, module)


class _StubSupervisorAgent:
    async def process_document(self, *_args, **_kwargs):
        return {}


class _StubExtractionTeam:
    async def extract_entities(self, *_args, **_kwargs):
        return {}


class _StubPDFExtractor:
    def extract(self, *_args, **_kwargs):  # pragma: no cover - not used in tests
        return {}


class _StubExtractionTools:
    pass


class _StubEmbeddingIngestionPipeline:
    def ingest_document(self, *_args, **_kwargs):
        return None


_register_stub("src.agents.supervisor_agent", SupervisorAgent=_StubSupervisorAgent)
_register_stub("src.agents.extraction_team", ExtractionTeam=_StubExtractionTeam)
_register_stub("src.extraction.enhanced_pdf_extractor", EnhancedPDFExtractor=_StubPDFExtractor)
_register_stub("src.extraction.enhanced_extraction_tools", EnhancedExtractionTools=_StubExtractionTools)
_register_stub(
    "src.retrieval.embedding_ingestion",
    EmbeddingIngestionPipeline=_StubEmbeddingIngestionPipeline,
    EmbeddingIngestion=_StubEmbeddingIngestionPipeline,
)

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

from process_multi_agent_corpus import MultiAgentProcessor


def test_convert_to_knowledge_entry_populates_metadata_fields():
    processor = MultiAgentProcessor.__new__(MultiAgentProcessor)

    result = {
        "document_text": "one two three four",
        "agent_outputs": {},
    }

    entry = asyncio.run(
        processor._convert_to_knowledge_entry(
            result,
            document_id="doc-123",
            processing_time=1.5,
            file_size=4096,
            page_count=12,
            filename="doc-123.pdf",
            fallback_full_text="",
        )
    )

    metadata = entry.metadata
    assert metadata.file_size == 4096
    assert metadata.page_count == 12
    assert metadata.full_text_stored is True
    assert metadata.word_count == 4
    assert metadata.filename == "doc-123.pdf"


def test_convert_to_knowledge_entry_uses_fallback_text_for_metadata():
    processor = MultiAgentProcessor.__new__(MultiAgentProcessor)

    result = {
        "agent_outputs": {},
    }

    entry = asyncio.run(
        processor._convert_to_knowledge_entry(
            result,
            document_id="doc-456",
            processing_time=2.0,
            file_size=2048,
            page_count=8,
            filename="custom-name.pdf",
            fallback_full_text="fallback text only",
        )
    )

    metadata = entry.metadata
    assert metadata.file_size == 2048
    assert metadata.page_count == 8
    assert metadata.full_text_stored is True
    assert metadata.word_count == 3
    assert metadata.filename == "custom-name.pdf"
    assert entry.full_document_text == "fallback text only"


def test_convert_to_knowledge_entry_includes_client_profiles():
    processor = MultiAgentProcessor.__new__(MultiAgentProcessor)

    profile_set = ClientProfileSet(
        profiles=[
            _profile_template(
                name="Transit sensory overload",
                intake='"The red line makes my skin buzz"',
                research="[Source 1] explains auditory overload on commuters.",
                benefit="[Source 1] backs vestibular regulation for this pattern.",
                alignment="[Source 1] plus STL Metro noise logs demand local adjustments.",
                localities=COMMUTER_LOCALITIES,
                connections=COMMUTER_CONNECTIONS,
                prediction_errors=[
                    PredictionErrorEntry(
                        trigger_context="Assumes Metro commutes only drain.",
                        surprising_pivot="[Source 1] highlights riders scripting sensory resets at the Arch overlook.",
                        intended_cognitive_effect="Encourage scouting for restorative pauses mid-commute.",
                    ),
                    PredictionErrorEntry(
                        trigger_context="Believes quiet spaces aren't accessible downtown.",
                        surprising_pivot="[Source 1] plus [Ext 1] surface library sensory-friendly hours near the platform.",
                        intended_cognitive_effect="Shift mindset toward civic supports hiding in plain sight.",
                    ),
                ],
            ),
            _profile_template(
                name="After-school crash",
                intake='"By pickup time I am empty and shaking"',
                research="[Source 1] connects cortisol spikes to sensory debt.",
                benefit="[Source 1] validates co-reg stops before homework.",
                alignment="[Source 1] and STL school density data show needed buffers.",
                localities=CAREGIVER_LOCALITIES,
                connections=CAREGIVER_CONNECTIONS,
                prediction_errors=[
                    PredictionErrorEntry(
                        trigger_context="Assumes regulation can only happen at home.",
                        surprising_pivot="[Source 1] documents park-based sensory resets on the drive home.",
                        intended_cognitive_effect="Invite the team to treat the commute as a co-regulation asset.",
                    ),
                    PredictionErrorEntry(
                        trigger_context="Believes caregivers must shoulder the load solo.",
                        surprising_pivot="[Source 1] plus [Ext 2] highlight neighborhood carpool circles sharing decompression.",
                        intended_cognitive_effect="Encourage distributed networks for after-school support.",
                    ),
                ],
            ),
            _profile_template(
                name="Night shift hypervigilance",
                intake='"Night sirens keep my jaw locked"',
                research="[Source 1] documents auditory hypervigilance for night workers.",
                benefit="[Source 1] shows decompression pods reduce sympathetic spikes.",
                alignment="[Source 1] paired with hospital siren counts justify hospital pods.",
                localities=SHIFT_LOCALITIES,
                connections=SHIFT_CONNECTIONS,
                prediction_errors=[
                    PredictionErrorEntry(
                        trigger_context="Assumes decompression can't happen on site.",
                        surprising_pivot="[Source 1] references chapels and dim rooms staff already use before commuting.",
                        intended_cognitive_effect="Encourage immediate decompression rituals before leaving work.",
                    ),
                    PredictionErrorEntry(
                        trigger_context="Believes every route home is equally loud.",
                        surprising_pivot="[Source 1] plus [Ext 3] map river-adjacent drives with fewer sirens.",
                        intended_cognitive_effect="Promote experimenting with quieter corridors.",
                    ),
                ],
            ),
        ]
    )

    result = {
        "document_text": "sample body text",
        "agent_outputs": {"client_profiles": profile_set.model_dump()},
    }

    entry = asyncio.run(
        processor._convert_to_knowledge_entry(
            result,
            document_id="doc-789",
            processing_time=1.2,
            file_size=1024,
            page_count=4,
            filename="doc-789.pdf",
            fallback_full_text="",
        )
    )

    assert entry.client_profiles is not None
    assert len(entry.client_profiles.profiles) == 3
    first = entry.client_profiles.profiles[0]
    assert first.research_reference.startswith("[Source 1]")
