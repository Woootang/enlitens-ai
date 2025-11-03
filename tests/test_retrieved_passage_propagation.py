import pytest

pydantic = pytest.importorskip("pydantic")

from src.agents.clinical_synthesis_agent import ClinicalSynthesisAgent, ClinicalOutline
from src.agents.educational_content_agent import EducationalContentAgent
from src.agents.rebellion_framework_agent import RebellionFrameworkAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.agents.workflow_state import create_initial_state
from src.models.enlitens_schemas import ClinicalContent, EducationalContent, RebellionFramework


class _RecordingOllamaClient:
    """Test double that records prompts and returns canned responses."""

    def __init__(self, responses, *, text_responder=None):
        self.responses = responses
        self.prompts = []
        self.text_prompts = []
        self._text_responder = text_responder

    async def generate_structured_response(self, *, prompt, response_model, **kwargs):  # type: ignore[override]
        self.prompts.append((prompt, response_model))
        factory = self.responses.get(response_model)
        if callable(factory):
            return factory()
        return factory

    async def generate_response(self, prompt, **kwargs):  # type: ignore[override]
        self.text_prompts.append(prompt)
        responder = self._text_responder
        if callable(responder):
            text = responder(prompt, kwargs)
        elif isinstance(responder, str):
            text = responder
        else:
            text = ""
        return {"response": text}


def _sample_retrieved_passages():
    return [
        {
            "text": "Neurodivergent brains in high-adversity cities show heightened sensory load and resilience markers.",
            "chunk_id": "chunk-42",
            "document_id": "doc-ctx",
            "score": 0.91,
        }
    ]


def test_supervisor_agent_context_includes_retrieved_passages():
    supervisor = SupervisorAgent()
    state = create_initial_state(
        document_id="doc-1",
        document_text="Example text",
        doc_type="default",
        client_insights={},
        founder_insights={},
        st_louis_context={},
        raw_client_context="Client raw context about executive function.",
        raw_founder_context="Founder raw context about rebellious tone.",
        cache_prefix="doc-1",
        cache_chunk_id="doc-1:root",
    )
    state["context_result"] = {
        "rag_retrieval": {"top_passages": _sample_retrieved_passages()}
    }

    context = supervisor._build_agent_context(state, {}, "clinical_synthesis", 1)

    retrieved = context.get("retrieved_passages")
    assert isinstance(retrieved, list) and retrieved, "Expected retrieved passages"
    assert retrieved[0]["text"].startswith("Neurodivergent brains"), "Incorrect passage text propagated"


@pytest.mark.asyncio
async def test_clinical_agent_prompt_and_output_reference_retrieved_passage():
    agent = ClinicalSynthesisAgent()
    client = _RecordingOllamaClient(
        {
            ClinicalOutline: lambda: ClinicalOutline(
                thesis="[Source 1] cite the sensory load thesis.",
                sections=["Strength Lens", "Systems"],
                client_strengths=["Pattern spotting [Source 1]"],
                key_system_levers=["Change lighting"],
                rallying_cry="Brains adapt, systems must follow.",
            ),
            ClinicalContent: lambda: ClinicalContent(
                interventions=[
                    "Deploy sensory diet referencing [Source 1] findings.",
                    "Use co-regulated breath sets.",
                    "Map executive off-ramps.",
                ],
                assessments=[
                    "Run QEEG baseline [Source 1]",
                    "Collect HRV",
                    "Executive functioning interview",
                ],
                outcomes=[
                    "Morning activation improves.",
                    "Sensory load decreases [Source 1]",
                    "Clients report agency",
                ],
                protocols=["90-day sensory plan", "Weekly vestibular reset", "Partner coaching"],
                guidelines=["Name adaptations", "Cite [Source 1] load", "Avoid pathologizing"],
                contraindications=["No shame tactics", "Avoid sedation", "Skip isolation"],
                side_effects=["Possible fatigue", "Emotional release", "Short-term detox"],
                monitoring=["Track HRV", "Sensory diary [Source 1]", "Executive check-ins"],
            ),
        }
    )
    agent.ollama_client = client

    context = {
        "science_data": {"research_content": {"findings": ["Example finding"]}},
        "document_text": "Document body",
        "retrieved_passages": _sample_retrieved_passages(),
        "raw_client_context": "Client raw context",
    }

    result = await agent.process(context)

    prompts = "\n".join(p for p, _ in client.prompts)
    assert "Neurodivergent brains" in prompts
    assert "[Source 1]" in result["clinical_content"]["interventions"][0]


@pytest.mark.asyncio
async def test_educational_agent_prompt_and_output_reference_retrieved_passage():
    agent = EducationalContentAgent()
    client = _RecordingOllamaClient(
        {
            EducationalContent: lambda: EducationalContent(
                explanations=["Explain nervous system shifts [Source 1]"],
                examples=["Morning commute meltdown [Source 1]"],
                analogies=["Brain like a smoke alarm"],
                definitions=["Interoception"],
                processes=["How regulation works"],
                comparisons=["ADHD vs. burnout"],
                visual_aids=["Sensory map"],
                learning_objectives=["Name two adaptations"],
            )
        }
    )
    agent.ollama_client = client

    context = {
        "document_text": "Document body",
        "science_data": {"research_content": {"findings": ["Example finding"]}},
        "clinical_content": {},
        "retrieved_passages": _sample_retrieved_passages(),
    }

    result = await agent.process(context)

    prompt = client.prompts[0][0]
    assert "Neurodivergent brains" in prompt
    assert "[Source 1]" in result["educational_content"]["explanations"][0]


@pytest.mark.asyncio
async def test_rebellion_agent_prompt_and_output_reference_retrieved_passage():
    agent = RebellionFrameworkAgent()
    client = _RecordingOllamaClient(
        {
            RebellionFramework: lambda: RebellionFramework(
                narrative_deconstruction=["Flip deficit script [Source 1]"],
                sensory_profiling=["Sensory load story"],
                executive_function=["Context-first planning"],
                social_processing=["Explain co-regulation"],
                strengths_synthesis=["Hyperfocus is survival [Source 1]"],
                rebellion_themes=["Systems change"],
                aha_moments=["You're not broken [Source 1]"]
            )
        }
    )
    agent.ollama_client = client

    context = {
        "document_text": "Document body",
        "science_data": {"research_content": {"findings": ["Example finding"]}},
        "clinical_content": {},
        "retrieved_passages": _sample_retrieved_passages(),
    }

    result = await agent.process(context)

    prompt = client.prompts[0][0]
    assert "Neurodivergent brains" in prompt
    assert "[Source 1]" in result["rebellion_framework"]["narrative_deconstruction"][0]
