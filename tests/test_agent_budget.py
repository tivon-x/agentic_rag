from __future__ import annotations

from langchain_core.documents import Document

from agent.adaptive import AdaptiveEvidenceLoop, _normalize_assessments


class FakeRetriever:
    def __init__(self, responses: dict[str, list[Document]]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def retrieve(self, query: str, *, query_plan: dict):
        self.calls.append(query)
        return self.responses.get(query, [])


def _document(evidence_id: str, quote: str = "Supported source text.") -> Document:
    return Document(
        page_content=quote,
        metadata={"passage_id": evidence_id, "quote_text": quote, "page": 1, "source": "paper.pdf"},
    )


def _answerer(_, evidence, __):
    return {"answer": "Grounded answer.", "claims": [{"claim": "Grounded answer.", "evidence_ids": [item["evidence_id"] for item in evidence[:1]], "major": True}], "limitations": ""}


def test_first_round_sufficient_does_not_follow_up():
    retriever = FakeRetriever({"first": [_document("e1")]})
    loop = AdaptiveEvidenceLoop(
        retriever,
        expected_index_version=None,
        planner=lambda _: [{"id": "r1", "requirement": "fact", "query": "first"}],
        assessor=lambda _, evidence: [{"requirement_id": "r1", "covered": bool(evidence), "evidence_ids": ["e1"], "coverage": 1.0}],
        follow_up=lambda _: "second",
        answerer=_answerer,
    )

    result = loop.run("question")

    assert result.strategy == "fixed"
    assert result.rounds == 1
    assert result.tool_calls == 1
    assert retriever.calls == ["first"]


def test_live_style_first_round_query_can_keep_full_user_wording():
    retriever = FakeRetriever({"full question": [_document("e1")]})
    loop = AdaptiveEvidenceLoop(
        retriever,
        expected_index_version=None,
        planner=lambda _: [
            {"id": "r1", "requirement": "first fact", "query": "shortened query"},
            {"id": "r2", "requirement": "second fact", "query": "another shortened query"},
        ],
        assessor=lambda _, evidence: [
            {"requirement_id": "r1", "covered": bool(evidence), "evidence_ids": ["e1"], "coverage": 1.0},
            {"requirement_id": "r2", "covered": bool(evidence), "evidence_ids": ["e1"], "coverage": 1.0},
        ],
        follow_up=lambda _: "second",
        answerer=_answerer,
        first_round_queries=lambda query, _: [query],
    )

    result = loop.run("full question")

    assert result.strategy == "fixed"
    assert retriever.calls == ["full question"]


def test_follow_up_is_once_and_only_for_missing_requirements():
    retriever = FakeRetriever({"first": [_document("e1")], "missing": [_document("e2")]})

    def assess(_, evidence):
        ids = {item["evidence_id"] for item in evidence}
        return [
            {"requirement_id": "covered", "covered": "e1" in ids, "evidence_ids": ["e1"], "coverage": 1.0},
            {"requirement_id": "missing", "covered": "e2" in ids, "evidence_ids": ["e2"] if "e2" in ids else [], "coverage": 1.0 if "e2" in ids else 0.0, "recommended_follow_up_query": "missing"},
        ]

    loop = AdaptiveEvidenceLoop(
        retriever,
        expected_index_version=None,
        planner=lambda _: [{"id": "covered", "requirement": "first fact", "query": "first"}, {"id": "missing", "requirement": "second fact", "query": "first"}],
        assessor=assess,
        follow_up=lambda missing: missing[0]["recommended_follow_up_query"],
        answerer=_answerer,
    )

    result = loop.run("question")

    assert result.strategy == "adaptive"
    assert result.rounds == 2
    assert result.tool_calls == 2
    assert retriever.calls == ["first", "missing"]


def test_duplicate_follow_up_stops_without_second_retrieval():
    retriever = FakeRetriever({"first": [_document("e1")]})
    loop = AdaptiveEvidenceLoop(
        retriever,
        expected_index_version=None,
        planner=lambda _: [{"id": "r1", "requirement": "fact", "query": "first"}],
        assessor=lambda _, __: [{"requirement_id": "r1", "covered": False, "evidence_ids": [], "coverage": 0.0, "recommended_follow_up_query": "first"}],
        follow_up=lambda missing: missing[0]["recommended_follow_up_query"],
        answerer=_answerer,
    )

    result = loop.run("question")

    assert result.strategy == "refuse"
    assert result.termination_reason == "duplicate_query_scope"
    assert result.tool_calls == 1


def test_evidence_and_context_budgets_are_hard_limits():
    documents = [_document(f"e{index}", "word " * 5000) for index in range(20)]
    retriever = FakeRetriever({"first": documents})
    loop = AdaptiveEvidenceLoop(
        retriever,
        expected_index_version=None,
        planner=lambda _: [{"id": "r1", "requirement": "fact", "query": "first"}],
        assessor=lambda _, evidence: [{"requirement_id": "r1", "covered": bool(evidence), "evidence_ids": [item["evidence_id"] for item in evidence], "coverage": 1.0}],
        follow_up=lambda _: "",
        answerer=_answerer,
    )

    result = loop.run("question")

    assert len(result.evidence) <= 12
    assert result.context_tokens <= 12_000
    assert result.tool_calls <= 4


def test_cancel_and_retrieval_errors_stop_without_more_calls():
    retriever = FakeRetriever({"first": [_document("e1")]})
    loop = AdaptiveEvidenceLoop(
        retriever,
        expected_index_version=None,
        planner=lambda _: [{"id": "r1", "requirement": "fact", "query": "first"}],
        assessor=lambda _, __: [],
        follow_up=lambda _: "second",
        answerer=_answerer,
    )

    cancelled = loop.run("question", cancelled=lambda: True)

    assert cancelled.termination_reason == "cancelled"
    assert cancelled.tool_calls == 0

    class BrokenRetriever(FakeRetriever):
        def retrieve(self, query: str, *, query_plan: dict):
            raise RuntimeError("network unavailable")

    broken = AdaptiveEvidenceLoop(
        BrokenRetriever({}),
        expected_index_version=None,
        planner=lambda _: [{"id": "r1", "requirement": "fact", "query": "first"}],
        assessor=lambda _, __: [],
        follow_up=lambda _: "second",
        answerer=_answerer,
    ).run("question")

    assert broken.termination_reason == "retrieval_error"
    assert broken.tool_calls == 0


def test_assessor_cannot_mark_requirement_covered_with_unreturned_evidence():
    assessments = _normalize_assessments(
        [
            {
                "requirement_id": "r1",
                "covered": True,
                "evidence_ids": ["invented"],
                "coverage": 1.0,
            }
        ],
        [{"id": "r1", "requirement": "fact", "query": "fact"}],
        {"returned"},
    )

    assert assessments[0]["covered"] is False
    assert assessments[0]["coverage"] == 0.0
