from src.monitoring.voice_audit import LightweightTfidfBackend, VoiceAuditor, load_reference_texts


def test_voice_auditor_scores(tmp_path):
    reference_file = tmp_path / "voice.md"
    reference_file.write_text("Sample voice paragraph one.\n\nAnother bold Enlitens paragraph.")

    auditor = VoiceAuditor(load_reference_texts(reference_file), backend=LightweightTfidfBackend())

    samples = [
        {"document_id": "doc-pass", "text": "Bold Enlitens paragraph with similar energy."},
        {"document_id": "doc-drift", "text": "This is a clinical summary without rebellion."},
    ]

    results = auditor.batch_score(samples, threshold=0.3)
    scores = {result.document_id: result for result in results}

    assert scores["doc-pass"].verdict == "pass"
    assert scores["doc-drift"].verdict == "drift"
    assert 0.0 <= scores["doc-pass"].similarity <= 1.0


def test_load_reference_texts(tmp_path):
    reference_file = tmp_path / "voice.md"
    reference_file.write_text("Paragraph A.\n\nParagraph B.\n\n")
    blocks = load_reference_texts(reference_file)
    assert blocks == ["Paragraph A.", "Paragraph B."]
