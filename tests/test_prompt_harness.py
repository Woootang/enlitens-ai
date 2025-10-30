from src.testing.prompt_harness import PromptHarness


def test_prompt_harness_flags_missing_clauses():
    harness = PromptHarness()
    prompt_variants = {
        "strong": (
            "Extract ONLY from the provided document, cite with [Source: ...] and respond with "
            '"Refusal: insufficient grounding" when evidence is missing.'
        ),
        "weak": "Extract ONLY from the document without refusal clause.",
    }

    evaluation = harness.evaluate_prompt_variants("science_extraction", prompt_variants)

    strong = evaluation["variants"]["strong"]
    weak = evaluation["variants"]["weak"]

    assert strong["score"] == 1.0
    assert "Refusal" in weak["missing_criteria"]


def test_prompt_harness_ranking_prefers_high_score():
    harness = PromptHarness()
    prompt_variants = {
        "baseline": "Extract ONLY from the provided document and cite with [Source:",
        "enhanced": (
            "Extract ONLY from the provided document, cite with [Source: ...], and use Refusal: insufficient grounding "
            "if information is absent."
        ),
    }

    evaluation = harness.evaluate_prompt_variants("science_extraction", prompt_variants)
    best = evaluation["best_variant"]
    assert best == "enhanced"
