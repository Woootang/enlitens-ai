import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytest

pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.agents.context_rag_agent import ContextRAGAgent
from src.models.enlitens_schemas import (
    BlogContent,
    ClinicalContent,
    ClientProfile,
    ClientProfileSet,
    ContentCreationIdeas,
    DocumentMetadata,
    EducationalContent,
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
from src.retrieval.embedding_ingestion import EmbeddingIngestionPipeline
from src.retrieval.vector_store import QdrantVectorStore


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
        f"{localities[0]} anchors this FICTIONAL composite's weekly rhythm. "
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


@pytest.fixture(scope="module", autouse=True)
def use_small_embedding_model():
    os.environ["ENLITENS_EMBED_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
    yield
    os.environ.pop("ENLITENS_EMBED_MODEL", None)


def build_sample_entry(document_id: str = "doc-001") -> EnlitensKnowledgeEntry:
    metadata = DocumentMetadata(
        document_id=document_id,
        filename=f"{document_id}.pdf",
        processing_timestamp=datetime.utcnow(),
        full_text_stored=True,
    )
    return EnlitensKnowledgeEntry(
        metadata=metadata,
        extracted_entities=ExtractedEntities(
            biomedical_entities=["neuroplasticity"],
            neuroscience_entities=["prefrontal cortex"],
            clinical_entities=["anxiety"],
            statistical_entities=["effect size"],
            total_entities=4,
        ),
        rebellion_framework=RebellionFramework(narrative_deconstruction=["Challenges old models"]),
        marketing_content=MarketingContent(headlines=["Brain-based healing"]),
        seo_content=SEOContent(primary_keywords=["neuroscience therapy"]),
        website_copy=WebsiteCopy(about_sections=["Neurodiversity affirming care"]),
        blog_content=BlogContent(article_ideas=["How the brain adapts"]),
        social_media_content=SocialMediaContent(captions=["Your brain is adapting"]),
        educational_content=EducationalContent(explanations=["Neural pathways reshape"]),
        clinical_content=ClinicalContent(interventions=["Polyvagal exercises"]),
        research_content=ResearchContent(findings=["Improved regulation"]),
        content_creation_ideas=ContentCreationIdeas(topic_ideas=["Executive function support"]),
        client_profiles=ClientProfileSet(
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
                            trigger_context="Assumes Metro rides always deplete regulation.",
                            surprising_pivot="[Source 1] shows commuters who schedule sensory resets at the Arch grounds overlook.",
                            intended_cognitive_effect="Encourage experimenting with restorative micro-pauses en route.",
                        ),
                        PredictionErrorEntry(
                            trigger_context="Believes quiet spaces are unavailable downtown after work.",
                            surprising_pivot="[Source 1] plus [Ext 1] cite library sensory hours near the Metro platform.",
                            intended_cognitive_effect="Highlight civic assets that disrupt depletion narratives.",
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
                            trigger_context="Assumes decompression can only happen at home.",
                            surprising_pivot="[Source 1] documents playground sensory diets between school and home improving regulation.",
                            intended_cognitive_effect="Encourage families to treat the pickup window as an intervention zone.",
                        ),
                        PredictionErrorEntry(
                            trigger_context="Believes caregivers have to manage regulation alone.",
                            surprising_pivot="[Source 1] plus [Ext 2] surface neighborhood carpool pods trading regulation duties.",
                            intended_cognitive_effect="Shift the mindset toward community distributed support.",
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
                            trigger_context="Assumes recovery can only happen after reaching home.",
                            surprising_pivot="[Source 1] references on-site dim rooms available before the commute.",
                            intended_cognitive_effect="Prompt planning for immediate post-shift decompression.",
                        ),
                        PredictionErrorEntry(
                            trigger_context="Believes every drive home is equally overwhelming.",
                            surprising_pivot="[Source 1] plus [Ext 3] identify quieter riverfront routes for night shifts.",
                            intended_cognitive_effect="Encourage experimentation with less triggering corridors.",
                        ),
                    ],
                ),
            ]
        ),
        full_document_text="Neuroscience-based therapy helps regulate the nervous system and builds resilience.",
    )


def test_embedding_ingestion_generates_chunks():
    vector_store = QdrantVectorStore(collection_name="test_ingest", host="invalid")
    pipeline = EmbeddingIngestionPipeline(vector_store=vector_store, chunk_size_tokens=128)

    entry = build_sample_entry()
    stats = pipeline.ingest_entry(entry)

    assert stats.chunks_ingested > 0
    assert stats.full_text_chunks > 0
    assert vector_store.count_by_document(entry.metadata.document_id) == stats.chunks_ingested


def test_context_rag_agent_returns_results():
    vector_store = QdrantVectorStore(collection_name="test_context", host="invalid")
    pipeline = EmbeddingIngestionPipeline(vector_store=vector_store, chunk_size_tokens=128)

    entry = build_sample_entry("doc-002")
    pipeline.ingest_document(
        document_id=entry.metadata.document_id,
        full_text=entry.full_document_text,
        agent_outputs={
            "clinical_content": {"interventions": entry.clinical_content.interventions},
            "marketing_content": {"headlines": entry.marketing_content.headlines},
        },
        metadata={"document_id": entry.metadata.document_id},
        rebuild=True,
    )

    agent = ContextRAGAgent(vector_store=vector_store, top_k=3)

    async def run_agent():
        await agent.initialize()
        output = await agent.process(
            {
                "document_text": "Regulating the nervous system is key for trauma healing.",
                "client_insights": {"themes": ["trauma", "executive function"]},
                "founder_insights": {"voice": "Your brain isn't broken"},
                "st_louis_context": {"needs": ["nervous system regulation"]},
                "intermediate_results": {"science_extraction": {"key_points": ["neuroplasticity"]}},
            }
        )
        await agent.cleanup()
        return output

    result = asyncio.run(run_agent())

    assert result["context_enhanced"] is True
    rag = result["rag_retrieval"]
    assert rag["top_passages"], "Expected retrieval results"
    assert entry.metadata.document_id in rag["related_documents"]
    first_passage = rag["top_passages"][0]
    assert "alignment" in first_passage
    assert "combined_score" in first_passage
    assert isinstance(first_passage["alignment"].get("score"), float)
    summary = rag["alignment_summary"]
    assert summary["count"] == len(rag["top_passages"])
    assert summary["average_alignment"] <= summary["average_combined_score"] + 1  # sanity bound
