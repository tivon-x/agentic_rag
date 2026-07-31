from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from agent.adaptive_graph import _route
from evals.m4_1_1_runner import _score_claims, _validate_dataset_hash


def test_m4_1_1_frozen_datasets_are_balanced_and_new() -> None:
    route_path = Path("evals/datasets/m4_1_1_route_v1.json")
    answer_path = Path("evals/datasets/m4_1_1_answer_v1.json")
    old_answer_path = Path("evals/datasets/m4_1_answer_v1.json")
    route_cases = json.loads(route_path.read_text(encoding="utf-8"))
    answer_cases = json.loads(answer_path.read_text(encoding="utf-8"))
    old_questions = {
        item["query"]
        for item in json.loads(old_answer_path.read_text(encoding="utf-8"))
    }

    assert Counter(item["expected_route"] for item in route_cases) == {
        "direct": 12,
        "fixed": 12,
        "adaptive": 12,
        "refuse": 12,
    }
    assert len(answer_cases) == 24
    assert sum(item["authoring_source"] == "m3_difficulty_taxonomy" for item in answer_cases) == 12
    assert sum(item["authoring_source"] == "independent" for item in answer_cases) == 12
    assert not old_questions.intersection(item["query"] for item in answer_cases)
    assert all(item["claim_specs"] for item in answer_cases)


def test_m4_1_1_manifest_hashes_validate() -> None:
    manifest = Path("evals/datasets/m4_1_1_dataset_manifest.json")
    assert _validate_dataset_hash(
        Path("evals/datasets/m4_1_1_route_v1.json"),
        manifest,
        "m4_1_1_route",
    )["milestone"] == "M4.1.1"
    assert _validate_dataset_hash(
        Path("evals/datasets/m4_1_1_answer_v1.json"),
        manifest,
        "m4_1_1_answer",
    )["milestone"] == "M4.1.1"


def test_claim_level_scoring_requires_matching_evidence_and_semantic_support() -> None:
    claims = [
        {"claim": "supported", "evidence_ids": ["e1"], "major": True},
        {"claim": "wrong citation", "evidence_ids": ["e2"], "major": True},
    ]
    specs = [
        {
            "id": "r1",
            "acceptable_evidence_ids": ["e1"],
        }
    ]
    judgments = [
        {
            "claim_index": 0,
            "claim_spec_id": "r1",
            "semantically_supported": True,
        },
        {
            "claim_index": 1,
            "claim_spec_id": "r1",
            "semantically_supported": True,
        },
    ]

    scored = _score_claims(claims, specs, judgments, {"e1", "e2"})

    assert scored["requirement_coverage"] == 1.0
    assert scored["citation_correctness"] == 0.5
    assert scored["major_fact_support_rate"] == 0.5
    assert scored["unsupported_major_claim_count"] == 1
    assert scored["semantic_false_positive_count"] == 1
    assert scored["gold_evidence_miss_count"] == 0


def test_pre_retrieval_route_only_handles_direct_and_refuse_boundaries() -> None:
    assert _route("收到") == "direct"
    assert _route("今天北京天气怎样") == "refuse"
    assert _route("Transformer 的 positional encoding 有何作用？") == "fact"
