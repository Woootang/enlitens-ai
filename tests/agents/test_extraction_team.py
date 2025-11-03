import asyncio

from src.agents.extraction_team import ExtractionTeam


def test_heuristic_entities_present_when_hf_disabled(monkeypatch):
    monkeypatch.setattr(ExtractionTeam, "_resolve_hf_support", lambda self: False)

    team = ExtractionTeam()

    sample_text = (
        "The patient showed improved clinical outcomes after targeted neural therapy. "
        "Neurotransmitter regulation supported hippocampal neurons while cellular biomarkers "
        "tracked disease progression."
    )

    extraction_payload = {
        "archival_content": {"full_document_text_markdown": sample_text}
    }

    entities = asyncio.run(team.extract_entities(extraction_payload))

    for bucket in ("biomedical", "neuroscience", "clinical"):
        assert bucket in entities
        assert entities[bucket], f"Expected non-empty heuristic entities for {bucket}"
        for entity in entities[bucket]:
            assert set(entity.keys()) == {"text", "label", "confidence", "start", "end"}
            assert isinstance(entity["text"], str)
            assert isinstance(entity["confidence"], float)

    list_buckets = [value for value in entities.values() if isinstance(value, list)]
    expected_total = sum(len(bucket) for bucket in list_buckets)
    assert entities["total_entities"] == expected_total
