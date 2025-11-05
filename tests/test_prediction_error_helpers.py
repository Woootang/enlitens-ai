from datetime import datetime
from pathlib import Path
import json

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
                ClientProfile(
                    profile_name="South City commuter",
                    intake_reference='"The red line makes my skin buzz"',
                    research_reference="[Source 1] discusses transit sensory overload.",
                    benefit_explanation="[Source 1] highlights vestibular resets for commuters.",
                    st_louis_alignment="[Source 1] plus STL Metro noise logs show localized action.",
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
                    regional_touchpoints=["South City", "Gateway Arch", "Central Library"],
                    masking_signals=["Commute shutdown", "Hypervigilant jaw clench"],
                    unmet_needs=[
                        "Predictable decompression",
                        "Transit sensory scouting",
                        "Community accountability for rest",
                    ],
                    support_recommendations=[
                        "Schedule river overlook pauses",
                        "Coordinate with Metro sensory hours",
                        "Share scripts for requesting quiet cars",
                    ],
                    cautionary_flags=["Do not promise outcomes", "Fictional composite reminder"],
                ),
                ClientProfile(
                    profile_name="MetroEast caregiver",
                    intake_reference='"By pickup time I am empty and shaking"',
                    research_reference="[Source 2] links cortisol spikes to afternoon sensory debt.",
                    benefit_explanation="[Source 2] shows carpool decompression rituals reduce crashes.",
                    st_louis_alignment="[Source 2] plus [Ext 3] uplift Metro East after-school supports.",
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
                    regional_touchpoints=["Fairmont City", "Granite City", "Bi-State carpool"],
                    masking_signals=["After-school crash", "Sensory exhaustion"],
                    unmet_needs=[
                        "Shared regulation roles",
                        "Bilingual caregiver scripts",
                        "Predictable decompression rituals",
                    ],
                    support_recommendations=[
                        "Rotate decompression leads",
                        "Use bilingual sensory prompts",
                        "Plan restorative stops before homework",
                    ],
                    cautionary_flags=["Flag fictional composite", "Escalate acute risk language"],
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
    assert len(records) == 4
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
    assert len(index["kb-001"]) == 4
