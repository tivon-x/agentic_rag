from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document

from agent.adaptive import _live_answerer, _to_evidence, _validate_claims
from agent.schemas import AdaptiveAnswer


def test_major_claim_without_valid_evidence_is_removed_and_gap_is_reported():
    evidence = [{"evidence_id": "e1", "quote": "source", "token_count": 1}]
    answer = {"answer": "Answer", "claims": [{"claim": "supported", "evidence_ids": ["e1"], "major": True}, {"claim": "unsupported", "evidence_ids": ["missing"], "major": True}], "limitations": ""}

    validated = _validate_claims(answer, evidence, [{"requirement_id": "r1", "covered": False, "coverage": 0.0}])

    assert validated["claims"] == [{"claim": "supported", "evidence_ids": ["e1"], "major": True}]
    assert "r1" in validated["limitations"]


def test_evidence_requires_id_quote_page_and_current_index_version():
    document = Document(page_content="quote", metadata={"passage_id": "e1", "page": 1, "source": "paper.pdf", "index_version": "v1"})
    assert _to_evidence(document, expected_index_version="v1")["evidence_id"] == "e1"

    with pytest.raises(ValueError, match="different active index"):
        _to_evidence(document, expected_index_version="v2")

    with pytest.raises(ValueError, match="stable ID, quote, or page"):
        _to_evidence(Document(page_content="quote", metadata={"passage_id": "e1", "page": 0}), expected_index_version=None)


def test_live_answerer_includes_the_user_query_in_model_payload(monkeypatch):
    captured: dict[str, object] = {}

    def fake_invoke(_, __, messages):
        captured["payload"] = json.loads(str(messages[-1].content))
        return AdaptiveAnswer(answer="answer", claims=[], limitations="")

    monkeypatch.setattr("agent.adaptive._invoke_structured_json", fake_invoke)

    _live_answerer("What does the paper report?", [], [])

    assert captured["payload"] == {
        "query": "What does the paper report?",
        "evidence": [],
        "assessments": [],
    }


def test_cited_answer_text_is_backfilled_when_structured_claims_are_missing():
    evidence_id = "a" * 64
    validated = _validate_claims(
        {
            "answer": f"The paper reports the result [{evidence_id}].",
            "claims": [],
            "limitations": "",
        },
        [{"evidence_id": evidence_id, "quote": "result", "token_count": 1}],
        [],
    )

    assert validated["claims"] == [
        {
            "claim": "The paper reports the result .",
            "evidence_ids": [evidence_id],
            "major": True,
        }
    ]
