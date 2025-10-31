import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.agents.context_rag_agent import ContextRAGAgent
from src.models.enlitens_schemas import (
    BlogContent,
    ClinicalContent,
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
from src.retrieval.embedding_ingestion import EmbeddingIngestionPipeline
from src.retrieval.vector_store import QdrantVectorStore


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
