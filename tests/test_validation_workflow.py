import pytest

pydantic = pytest.importorskip("pydantic")
import pytest
from pydantic import ValidationError

from src.models.enlitens_schemas import BlogContent
from src.validation.chain_of_verification import ChainOfVerification
from src.agents.validation_agent import ValidationAgent


def test_blog_statistics_validate_with_context():
    full_text = "The trial reported a 30% improvement in attention spans among participants."
    blog_data = {
        "statistics": [
            {
                "claim": "According to Rivera et al. (2021) the trial reported a 30% improvement in attention spans.",
                "citation": {
                    "quote": "a 30% improvement in attention spans",
                    "source_id": "doc-123",
                    "source_title": "Attention Study",
                },
            }
        ]
    }

    parsed = BlogContent.model_validate(blog_data, context={"source_text": full_text})
    assert parsed.statistics[0].claim.startswith("According to Rivera")


def test_blog_statistics_missing_quote_fails():
    full_text = "Baseline cortisol levels decreased modestly after the intervention."
    blog_data = {
        "statistics": [
            {
                "claim": "According to Singh et al. (2020) participants saw a 45% increase in energy.",
                "citation": {
                    "quote": "participants reported feeling invincible",
                    "source_id": "doc-456",
                },
            }
        ]
    }

    with pytest.raises(ValidationError):
        BlogContent.model_validate(blog_data, context={"source_text": full_text})


def test_marketing_compliance_flags_banned_terms():
    output = {
        "marketing_content": {
            "headlines": ["Guaranteed results in two weeks"],
            "value_propositions": ["Clinician-approved"],
        }
    }
    cov = ChainOfVerification()
    report = cov.run(output)
    marketing_step = next(step for step in report["steps"] if step["name"] == "Marketing compliance")
    assert not marketing_step["passed"]
    assert any("guarantee" in issue.lower() for issue in marketing_step["issues"])


@pytest.mark.asyncio
async def test_validation_agent_end_to_end():
    agent = ValidationAgent()
    await agent.initialize()

    source_text = (
        "Neuroplasticity-driven coaching improved executive function and attention in trial participants. "
        "According to Doe et al. (2022) [Source: Journal], cognitive retraining led to significant gains."
    )

    complete_output = {
        "research_content": {
            "findings": ["Neuroplastic changes observed."],
            "methodologies": ["Randomized controlled trial"],
            "statistics": ["According to Doe et al. (2022) [Source: Journal]"] ,
            "implications": ["Cognitive training improves executive regulation"],
            "citations": ["Doe et al., 2022"],
            "references": ["Doe, J. (2022). Journal of Executive Function."],
        },
        "clinical_content": {
            "interventions": [
                "Co-design rebellious routines with the client—torch the bullshit deficit story and map environmental triggers.",
                "Polyvagal regulation practice so the nervous system feels safe before cognitive work.",
            ],
            "assessments": ["Context-and-strengths inventory co-created with the client."],
            "protocols": [
                "Eight-week executive coaching arc with choice-driven sessions and collaborative checkpoints."
            ],
            "guidelines": [
                "Context audit: analyse workplace system pressures and burn the old metrics that ignore environment.",
                "Plan advocacy moves targeting school and workplace ableism."
            ],
            "outcomes": [
                "Client graduates with an autonomy blueprint and self-advocacy scripts for real-world labs."
            ],
            "contraindications": [
                "Pause immediately if the nervous system signals overwhelm; consent-driven pacing only."
            ],
            "side_effects": ["Possible fatigue when systems refuse to adapt—monitor and adjust the water, not the fish."],
            "monitoring": ["Track evidence citations and log gaps for future research pulls."],
        },
        "marketing_content": {
            "headlines": ["Rewire executive function with neuroscience"],
            "value_propositions": ["Grounded in peer-reviewed research"],
        },
        "seo_content": {
            "meta_descriptions": ["Explore neuroscience-backed coaching"],
            "primary_keywords": ["neuroplasticity coaching"],
        },
        "self_consistency": {"num_samples": 3, "vote_threshold": 2},
        "blog_content": {
            "statistics": [
                {
                    "claim": "According to Doe et al. (2022) data show improvement in executive control.",
                    "citation": {
                        "quote": "cognitive retraining led to significant gains",
                        "source_id": "doc-789",
                    },
                }
            ]
        },
        "full_document_text": source_text,
    }

    context = {"complete_output": complete_output, "document_text": source_text, "document_id": "doc-789", "retry_attempt": 1}
    result = await agent.process(context)

    assert result["final_validation"]["passed"] is True
    assert result["citation_report"]["failed"] == []
    assert result["retry_metadata"]["needs_retry"] is False
    assert result["self_critique"] is None
    review = result["constitutional_review"]
    assert all(principle["passed"] for principle in review["principles"])
    assert result["quality_scores"]["overall_quality"] == pytest.approx(1.0)
