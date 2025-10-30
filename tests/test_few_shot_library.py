from src.synthesis.few_shot_library import FEW_SHOT_LIBRARY


def test_science_extraction_example_selected_by_similarity():
    query = "dopamine transporter mindfulness intervention"
    examples = FEW_SHOT_LIBRARY.get_examples("science_extraction", query, k=1)
    assert examples, "Expected at least one few-shot example"
    assert "dopamine" in examples[0].description.lower()


def test_render_for_prompt_includes_heading():
    rendered = FEW_SHOT_LIBRARY.render_for_prompt("science_extraction", "", k=1)
    assert "Few-shot example 1" in rendered
