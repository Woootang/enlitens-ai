import pytest

pydantic = pytest.importorskip("pydantic")

from src.agents.clinical_synthesis_agent import ClinicalSynthesisAgent, ClinicalOutline
from src.models.enlitens_schemas import ClinicalContent

from tests.test_retrieved_passage_propagation import _RecordingOllamaClient


@pytest.mark.asyncio
async def test_clinical_agent_brief_includes_late_document_section():
    late_marker = "LATE_SECTION_PAYLOAD"
    long_document = "Intro text. " + " ".join(["segment" for _ in range(1500)]) + f" Closing with {late_marker}."

    def text_responder(prompt, _kwargs):
        assert late_marker in prompt
        return f"- Opening context\n- {late_marker} summary"

    client = _RecordingOllamaClient(
        {
            ClinicalOutline: lambda: ClinicalOutline(
                thesis="Hold the systemic lens.",
                sections=["Strength Lens"],
                client_strengths=["Resilience"],
                key_system_levers=["Policy shift"],
                rallying_cry="Burn the deficit script.",
            ),
            ClinicalContent: lambda: ClinicalContent(
                interventions=["Intervene late"],
                assessments=["Assess"],
                outcomes=["Outcome"],
                protocols=["Protocol"],
                guidelines=["Guideline"],
                contraindications=["Contra"],
                side_effects=["Side"],
                monitoring=["Monitor"],
            ),
        },
        text_responder=text_responder,
    )

    agent = ClinicalSynthesisAgent()
    agent.ollama_client = client

    context = {
        "science_data": {"research_content": {"findings": ["Example finding"]}},
        "document_text": long_document,
        "retrieved_passages": [],
        "raw_client_context": "",
    }

    await agent.process(context)

    assert client.text_prompts, "Expected summarization helper to be invoked"
    assert late_marker in client.text_prompts[0]
    merged_prompts = "\n".join(prompt for prompt, _ in client.prompts)
    assert late_marker in merged_prompts
    assert "DOCUMENT BRIEF" in merged_prompts


@pytest.mark.asyncio
async def test_retrieval_overflow_triggers_brief_summary():
    extra_phrase = "Overflow summary"

    def text_responder(prompt, _kwargs):
        assert "[Extra 1]" in prompt
        return extra_phrase

    client = _RecordingOllamaClient(
        {
            ClinicalOutline: lambda: ClinicalOutline(
                thesis="Outline",
                sections=["S1"],
                client_strengths=["Strength"],
                key_system_levers=["Lever"],
                rallying_cry="Rally",
            ),
            ClinicalContent: lambda: ClinicalContent(
                interventions=["Intervention"],
                assessments=["Assessment"],
                outcomes=["Outcome"],
                protocols=["Protocol"],
                guidelines=["Guideline"],
                contraindications=["Contra"],
                side_effects=["Side"],
                monitoring=["Monitor"],
            ),
        },
        text_responder=text_responder,
    )

    agent = ClinicalSynthesisAgent()
    agent.ollama_client = client

    retrieved_passages = [
        {"text": f"Passage {idx} insight."}
        for idx in range(1, 11)
    ]

    context = {
        "science_data": {"research_content": {"findings": ["Example finding"]}},
        "document_text": "",
        "retrieved_passages": retrieved_passages,
        "raw_client_context": "",
    }

    await agent.process(context)

    assert client.text_prompts, "Expected overflow summary prompt"
    assert any("[Extra 1]" in prompt for prompt in client.text_prompts)
    merged_prompts = "\n".join(prompt for prompt, _ in client.prompts)
    assert extra_phrase in merged_prompts
    assert "Additional retrieved insights" in merged_prompts
