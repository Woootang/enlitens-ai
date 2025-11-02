import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

pytest.importorskip("pydantic")

from src.models.enlitens_schemas import ResearchContent
from src.synthesis.normalize import normalize_research_content_payload


def test_normalize_research_content_single_nested_finding():
    raw_payload = {
        "findings": [["Single finding"]],
        "statistics": [],
        "methodologies": [],
        "limitations": [],
        "future_directions": [],
        "implications": [],
        "citations": [],
        "references": [],
    }

    normalized = normalize_research_content_payload(raw_payload)
    model = ResearchContent.model_validate(normalized)

    assert model.findings == ["Single finding"]
