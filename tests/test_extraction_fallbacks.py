import asyncio
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

if "fitz" not in sys.modules:
    fake_fitz = types.SimpleNamespace()

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError("fitz unavailable in test environment")

    fake_fitz.open = _unavailable
    sys.modules["fitz"] = fake_fitz

if "pdfplumber" not in sys.modules:
    class _FakePDF:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        pages: list = []

    def _fake_pdfplumber_open(*_args, **_kwargs):
        return _FakePDF()

    sys.modules["pdfplumber"] = types.SimpleNamespace(open=_fake_pdfplumber_open)

if "docling" not in sys.modules:
    class _FakeDoclingConverter:
        def __init__(self, *args, **kwargs):
            pass

        def convert(self, *_args, **_kwargs):
            raise RuntimeError("docling stub invoked")

    sys.modules["docling"] = types.SimpleNamespace()
    sys.modules["docling.document_converter"] = types.SimpleNamespace(
        DocumentConverter=_FakeDoclingConverter
    )
    sys.modules["docling.datamodel"] = types.SimpleNamespace(pipeline_options=types.SimpleNamespace())
    sys.modules["docling.datamodel.pipeline_options"] = types.SimpleNamespace(
        PdfPipelineOptions=lambda **_kwargs: None
    )

if "torch" not in sys.modules:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    sys.modules["torch"] = types.SimpleNamespace(cuda=_FakeCuda())

if "transformers" not in sys.modules:
    def _fake_pipeline(*_args, **_kwargs):
        raise RuntimeError("transformers pipeline stub should not be called")

    sys.modules["transformers"] = types.SimpleNamespace(pipeline=_fake_pipeline)

from src.extraction.enhanced_pdf_extractor import EnhancedPDFExtractor
from src.extraction.enhanced_pdf_extractor_v2 import EnhancedPDFExtractorV2
from src.agents.extraction_team import ExtractionTeam


def _make_pdf(path: Path, text: str) -> None:
    fake_pdf = (
        "%PDF-1.4\n"
        "1 0 obj<<>>endobj\n"
        "2 0 obj<< /Length {length} >>stream\n{text}\nendstream\nendobj\n"
        "trailer<<>>\n%%EOF\n"
    ).format(length=len(text.encode("utf-8")), text=text)
    path.write_bytes(fake_pdf.encode("utf-8"))


def test_pymupdf_fallback(monkeypatch, tmp_path):
    pdf_path = tmp_path / "simple.pdf"
    _make_pdf(pdf_path, "Hello Enlitens")

    extractor = EnhancedPDFExtractor(cache_dir=tmp_path / "cache")

    def boom(_path):
        raise RuntimeError("docling broke")

    monkeypatch.setattr(extractor.docling_converter, "convert", boom)
    monkeypatch.setattr(extractor, "_fallback_with_marker", lambda _path: "")

    monkeypatch.setattr(extractor, "_fallback_with_pymupdf", lambda _path: "Hello Enlitens")

    result = extractor.extract(str(pdf_path))

    assert result["extraction_method"].endswith("pymupdf")
    assert "Hello Enlitens" in result["archival_content"]["full_document_text_markdown"]


def test_pdfplumber_fallback(monkeypatch, tmp_path):
    pdf_path = tmp_path / "plumber.pdf"
    _make_pdf(pdf_path, "Fallback to pdfplumber")

    extractor = EnhancedPDFExtractor(cache_dir=tmp_path / "cache2")

    def boom(_path):
        raise RuntimeError("docling broke")

    monkeypatch.setattr(extractor.docling_converter, "convert", boom)
    monkeypatch.setattr(extractor, "_fallback_with_marker", lambda _path: "")

    def pymupdf_failure(_path):
        raise RuntimeError("pymupdf unavailable")

    monkeypatch.setattr(extractor, "_fallback_with_pymupdf", pymupdf_failure)
    monkeypatch.setattr(extractor, "_fallback_with_pdfplumber", lambda _path: "Fallback to pdfplumber")

    result = extractor.extract(str(pdf_path))

    assert result["extraction_method"].endswith("pdfplumber")
    assert "Fallback to pdfplumber" in result["archival_content"]["full_document_text_markdown"]
    assert any("pymupdf" in err for err in result["quality_metrics"].get("fallback_errors", []))


def test_ocr_fallback(monkeypatch, tmp_path):
    pdf_path = tmp_path / "ocr.pdf"
    _make_pdf(pdf_path, "This text will require OCR")

    extractor = EnhancedPDFExtractor(cache_dir=tmp_path / "cache3")

    def boom(_path):
        raise RuntimeError("docling broke")

    monkeypatch.setattr(extractor.docling_converter, "convert", boom)
    monkeypatch.setattr(extractor, "_fallback_with_marker", lambda _path: "")
    monkeypatch.setattr(extractor, "_fallback_with_pymupdf", lambda _path: "")
    monkeypatch.setattr(extractor, "_fallback_with_pdfplumber", lambda _path: "")
    monkeypatch.setattr(extractor, "_fallback_with_ocr", lambda _path: "OCR fallback content")

    result = extractor.extract(str(pdf_path))

    assert result["extraction_method"].endswith("ocr")
    assert "OCR fallback content" in result["archival_content"]["full_document_text_markdown"]


def test_extraction_team_uses_cpu(monkeypatch):
    calls = []

    def fake_pipeline(task, model, device, **kwargs):
        calls.append((task, model, device, kwargs))

        class DummyPipeline:
            def __call__(self, text):
                return []

        return DummyPipeline()

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    team = ExtractionTeam(pipeline_factory=fake_pipeline)

    asyncio.run(team._extract_biomedical_entities("sample text"))
    asyncio.run(team._extract_biomedical_entities("another sample"))

    assert calls[0][2] == -1
    assert len(calls) == 1


def test_v2_pdfplumber_fallback(monkeypatch, tmp_path):
    pdf_path = tmp_path / "v2.pdf"
    _make_pdf(pdf_path, "Enhanced extractor V2 fallback")

    extractor = EnhancedPDFExtractorV2(cache_dir=tmp_path / "cache4")

    def boom(_path):
        raise RuntimeError("docling broke")

    monkeypatch.setattr(extractor.docling_converter, "convert", boom)
    monkeypatch.setattr(extractor, "_fallback_with_marker", lambda _path: "")
    monkeypatch.setattr(extractor, "_fallback_with_pymupdf", lambda _path: "")
    monkeypatch.setattr(extractor, "_fallback_with_pdfplumber", lambda _path: "Enhanced extractor V2 fallback")

    result = extractor.extract(str(pdf_path))

    assert result["extraction_method"].endswith("pdfplumber")
    assert result["archival_content"]["full_document_text_markdown"].strip()


def test_marker_fallback(monkeypatch, tmp_path):
    pdf_path = tmp_path / "marker.pdf"
    _make_pdf(pdf_path, "Marker saves the day")

    extractor = EnhancedPDFExtractor(cache_dir=tmp_path / "cache_marker")

    def boom(_path):
        raise RuntimeError("docling broke")

    monkeypatch.setattr(extractor.docling_converter, "convert", boom)
    monkeypatch.setattr(extractor, "_fallback_with_marker", lambda _path: "Marker saves the day")

    result = extractor.extract(str(pdf_path))

    assert result["extraction_method"].endswith("marker")
    assert "Marker saves the day" in result["archival_content"]["full_document_text_markdown"]


def test_v2_marker_fallback(monkeypatch, tmp_path):
    pdf_path = tmp_path / "v2_marker.pdf"
    _make_pdf(pdf_path, "Marker v2 success")

    extractor = EnhancedPDFExtractorV2(cache_dir=tmp_path / "cache_v2_marker")

    def boom(_path):
        raise RuntimeError("docling broke")

    monkeypatch.setattr(extractor.docling_converter, "convert", boom)
    monkeypatch.setattr(extractor, "_fallback_with_marker", lambda _path: "Marker v2 success")
    monkeypatch.setattr(extractor, "_fallback_with_pymupdf", lambda _path: "")

    result = extractor.extract(str(pdf_path))

    assert result["extraction_method"].endswith("marker")
    assert "Marker v2 success" in result["archival_content"]["full_document_text_markdown"]
