from datetime import datetime
import json
from pathlib import Path
from typing import Optional

import pytest

from src.knowledge_base.prediction_errors import (
    collect_prediction_error_index,
    filter_prediction_errors_by_locality,
    get_prediction_errors_for_profile,
    load_prediction_error_records,
)
from src.models.enlitens_schemas import (
    BlogContent,
    ClinicalContent,
    ClientProfile,
    ClientProfileSet,
    ContentCreationIdeas,
    DocumentMetadata,
    EducationalContent,
    EnlitensKnowledgeBase,
    EnlitensKnowledgeEntry,
    ExtractedEntities,
    MarketingContent,
    RebellionFramework,
    ResearchContent,
    SEOContent,
    SocialMediaContent,
    WebsiteCopy,
)
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


@pytest.fixture()
def sample_knowledge_base(tmp_path: Path) -> Path:
    document_id = "kb-001"
    metadata = DocumentMetadata(
        document_id=document_id,
        filename="kb-001.pdf",
        processing_timestamp=datetime.utcnow(),
        full_text_stored=True,
    )
    entry = EnlitensKnowledgeEntry(
        metadata=metadata,
        extracted_entities=ExtractedEntities(total_entities=0),
        rebellion_framework=RebellionFramework(),
        marketing_content=MarketingContent(),
        seo_content=SEOContent(),
        website_copy=WebsiteCopy(),
        blog_content=BlogContent(),
        social_media_content=SocialMediaContent(),
        educational_content=EducationalContent(),
        clinical_content=ClinicalContent(),
        research_content=ResearchContent(),
        content_creation_ideas=ContentCreationIdeas(),
        client_profiles=ClientProfileSet(
            profiles=[
                _profile_template(
                    name="South City commuter",
                    intake='"The red line makes my skin buzz"',
                    research="[Source 1] discusses transit sensory overload.",
                    benefit="[Source 1] highlights vestibular resets for commuters.",
                    alignment="[Source 1] plus STL Metro noise logs show localized action.",
                    localities=COMMUTER_LOCALITIES,
                    connections=COMMUTER_CONNECTIONS,
                    prediction_errors=[
                        PredictionErrorEntry(
                            trigger_context="Assumes every South City commute drains regulation.",
                            surprising_pivot="[Source 1] cites riders pausing at the Arch grounds overlook.",
                            intended_cognitive_effect="Encourage mapping restorative micro-stops.",
                        ),
                        PredictionErrorEntry(
                            trigger_context="Believes quiet routes don't exist downtown.",
                            surprising_pivot="[Ext 2] maps Market Street sensory-friendly corridors.",
                            intended_cognitive_effect="Spark curiosity about civic assets hiding in plain sight.",
                        ),
                    ],
                ),
                _profile_template(
                    name="MetroEast caregiver",
                    intake='"By pickup time I am empty and shaking"',
                    research="[Source 2] links cortisol spikes to afternoon sensory debt.",
                    benefit="[Source 2] shows carpool decompression rituals reduce crashes.",
                    alignment="[Source 2] plus [Ext 3] uplift Metro East after-school supports.",
                    localities=CAREGIVER_LOCALITIES,
                    connections=CAREGIVER_CONNECTIONS,
                    prediction_errors=[
                        PredictionErrorEntry(
                            trigger_context="Assumes regulation can only happen at home.",
                            surprising_pivot="[Source 2] documents Fairmont City park decompression walks.",
                            intended_cognitive_effect="Invite the reader to frame the commute as an intervention zone.",
                        ),
                        PredictionErrorEntry(
                            trigger_context="Believes caregivers must carry the load alone.",
                            surprising_pivot="[Ext 3] highlights bilingual carpool pods sharing sensory prep.",
                            intended_cognitive_effect="Encourage community-distributed support plans.",
                        ),
                    ],
                ),
                _profile_template(
                    name="North county connector",
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
                            surprising_pivot="[Ext 4] map river-adjacent drives with fewer sirens.",
                            intended_cognitive_effect="Promote experimenting with quieter corridors.",
                        ),
                    ],
                ),
            ],
            shared_thread="Fictional personas translating research into transit and after-school pivots.",
        ),
    )
    knowledge_base = EnlitensKnowledgeBase(
        version="test",
        created_at=datetime.utcnow(),
        total_documents=1,
        documents=[entry],
    )
    kb_path = tmp_path / "enlitens_knowledge_base_latest.json"
    with open(kb_path, "w", encoding="utf-8") as handle:
        json.dump(knowledge_base.model_dump(), handle, indent=2, default=str)
    return kb_path


def test_load_prediction_error_records_reads_file(sample_knowledge_base: Path):
    records = load_prediction_error_records(path=sample_knowledge_base)
    assert len(records) == 6
    assert any("[Source 1]" in record.entry.surprising_pivot for record in records)
    assert all(record.source_tags for record in records)


def test_get_prediction_errors_for_profile_filters(sample_knowledge_base: Path):
    entries = get_prediction_errors_for_profile(
        document_id="kb-001",
        profile_name="South City commuter",
        path=sample_knowledge_base,
    )
    assert len(entries) == 2
    assert entries[0].trigger_context.startswith("Assumes")


def test_filter_prediction_errors_by_locality_matches_case_insensitive(sample_knowledge_base: Path):
    results = filter_prediction_errors_by_locality(
        "fairmont city",
        path=sample_knowledge_base,
    )
    assert len(results) == 1
    assert results[0].profile_name == "MetroEast caregiver"


def test_collect_prediction_error_index_groups_by_document(sample_knowledge_base: Path):
    index = collect_prediction_error_index(path=sample_knowledge_base)
    assert set(index.keys()) == {"kb-001"}
    assert len(index["kb-001"]) == 6
