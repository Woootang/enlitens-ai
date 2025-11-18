import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agents.profile_matcher_agent import ProfileMatcherAgent
from src.utils.llm_response_cleaner import (
    extract_json_object,
    extract_persona_filenames,
    strip_reasoning_artifacts,
)


def test_extract_json_object_strips_think_blocks():
    raw = '<think>internal deliberation</think>\n{"status": "pass", "foo": 1}\nConfidence: 0.42'
    json_blob = extract_json_object(raw)
    assert json_blob is not None
    parsed = json.loads(json_blob)
    assert parsed["status"] == "pass"
    assert "think" not in json_blob.lower()


def test_profile_matcher_parses_persona_ids_with_reasoning():
    agent = ProfileMatcherAgent(personas_dir="enlitens_client_profiles/profiles")
    raw = "<think>analysis</think>\n1. persona_cluster_001.json - aligned\nConfidence: 0.38"
    ids, error = agent._parse_selection_response(raw)
    assert ids == ["persona_cluster_001.json"]
    assert error is None


def test_extract_json_object_handles_code_fence():
    raw = '```json\n{"status": "revise"}\n```'
    json_blob = extract_json_object(raw)
    assert json_blob == '{"status": "revise"}'
    parsed = json.loads(json_blob)
    assert parsed["status"] == "revise"


def test_profile_matcher_parses_json_array():
    agent = ProfileMatcherAgent(personas_dir="enlitens_client_profiles/profiles")
    raw = '["persona_cluster_010.json", "persona_cluster_011.json"]'
    ids, error = agent._parse_selection_response(raw)
    assert ids == ["persona_cluster_010.json", "persona_cluster_011.json"]
    assert error is None


def test_profile_matcher_handles_truncated_json_payload():
    agent = ProfileMatcherAgent(personas_dir="enlitens_client_profiles/profiles")
    raw = (
        '{"selected_persona_ids":["persona_cluster_020.json","persona_cluster_021.json"],'
        '"justifications":{"persona_cluster_020.json":"match","persona_cluster_021.json":"match"}},'
        '"\n\n\n excessive trailing tokens'
    )
    ids, error = agent._parse_selection_response(raw)
    assert ids == ["persona_cluster_020.json", "persona_cluster_021.json"]
    assert error is None


def test_extract_persona_filenames_recovers_unique_ids():
    messy = """
    <think>reasoning</think>
    persona_cluster_001.json was mentioned alongside persona_cluster_001.json again.
    The model also referenced persona_cluster_004.json.
    """
    ids = extract_persona_filenames(messy)
    assert ids == ["persona_cluster_001.json", "persona_cluster_004.json"]


