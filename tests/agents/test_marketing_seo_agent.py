import asyncio

from src.agents.marketing_seo_agent import MarketingSEOAgent
from src.models.enlitens_schemas import MarketingContent, SEOContent


class DummyOllamaClient:
    def __init__(self, marketing_payload, seo_payload):
        self._marketing_payload = marketing_payload
        self._seo_payload = seo_payload

    async def generate_structured_response(self, *, response_model, **kwargs):
        if response_model is MarketingContent:
            return response_model(**self._marketing_payload)
        if response_model is SEOContent:
            return response_model(**self._seo_payload)
        raise ValueError("Unexpected response model requested")

    def clone_with_model(self, model):
        return self


def test_marketing_seo_agent_process_without_title_tags():
    agent = MarketingSEOAgent()
    agent.is_initialized = True

    marketing_payload = {
        "headlines": ["Headline"],
        "taglines": ["Tagline"],
        "value_propositions": ["Value"],
        "benefits": ["Benefit"],
        "pain_points": ["Pain"],
    }

    seo_payload = {
        "primary_keywords": ["neuroscience therapy st louis"],
        "secondary_keywords": ["adhd therapy st louis"],
        "long_tail_keywords": ["neuroscience based therapy for anxiety"],
        "meta_descriptions": ["Explore neuroscience-based therapy in St. Louis for ADHD and anxiety."],
        "content_topics": ["How neuroscience therapy supports ADHD clients"],
    }

    agent.ollama_client = DummyOllamaClient(marketing_payload, seo_payload)

    context = {
        "final_context": {
            "research_content": {"key_findings": ["Finding"]},
            "clinical_content": {"treatment_approaches": ["Approach"]},
        }
    }

    result = asyncio.run(agent.process(context))

    assert result["seo_content"]["primary_keywords"] == seo_payload["primary_keywords"]
    assert "title_tags" not in result["seo_content"]
    assert result["generation_quality"] == "high"
