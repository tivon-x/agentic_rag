from pathlib import Path

import pytest
from langchain_core.documents import Document

from evals.metrics import (
    answer_completeness,
    citation_precision,
    groundedness_score,
    hallucination_rate_rule,
    ndcg_at_k,
    normalize_identifier,
    recall_at_k,
    reciprocal_rank,
    redundancy_rate,
    route_accuracy,
)
from evals.runner import (
    EvalCase,
    _build_leaderboard,
    _is_comparable_suite_result,
    _build_offline_extractive_answer,
    _document_matches_case,
    _load_eval_cases,
    _ordered_unique,
)


def test_load_eval_cases_reads_jsonl(tmp_path):
    path = tmp_path / "retrieval_cases.jsonl"
    path.write_text(
        '{"case_id":"c1","question":"What is Transformer?","expected_route":"retrieve","gold_doc_ids":["paper.pdf"],"gold_node_ids":[],"reference_answer":"A model.","difficulty":"easy","notes":"demo"}\n',
        encoding="utf-8",
    )

    cases = _load_eval_cases(path)

    assert cases == [
        EvalCase(
            case_id="c1",
            question="What is Transformer?",
            expected_route="retrieve",
            gold_doc_ids=["paper.pdf"],
            gold_node_ids=[],
            reference_answer="A model.",
            difficulty="easy",
            notes="demo",
        )
    ]


def test_retrieval_metrics_compute_expected_values():
    relevances = [0, 1, 0]

    assert recall_at_k(relevances, 2, k=3) == 0.5
    assert reciprocal_rank(relevances) == 0.5
    assert round(ndcg_at_k(relevances, 2, k=3), 4) == 0.3869
    assert redundancy_rate(["same", "same", "different"]) == pytest.approx(1 / 3)


def test_answer_metrics_measure_groundedness_and_hallucination():
    answer = "Transformer uses attention and removes convolutions."
    evidence = [
        "The Transformer is based solely on attention mechanisms.",
        "The architecture dispenses with recurrence and convolutions entirely.",
    ]

    assert groundedness_score(answer, evidence) > 0.5
    assert answer_completeness(
        answer,
        "The Transformer uses attention mechanisms instead of recurrence and convolutions.",
    ) > 0.5
    assert (
        hallucination_rate_rule(
            answer,
            evidence,
            reference_answer="Transformer removes convolutions.",
        )
        < 0.5
    )


def test_citation_precision_counts_matching_docs_and_nodes():
    assert citation_precision(["paper.pdf"], ["paper.pdf"]) == 1.0
    assert (
        citation_precision(
            ["other.pdf"],
            ["paper.pdf"],
            cited_node_ids=["node-1"],
            gold_node_ids=["node-1"],
        )
        == 1.0
    )
    assert (
        citation_precision(
            ["paper.pdf"],
            ["paper.pdf"],
            cited_node_ids=["node-1"],
            gold_node_ids=[],
        )
        == 1.0
    )


def test_document_matches_case_uses_source_basename_and_node_ids():
    case = EvalCase(
        case_id="c1",
        question="q",
        expected_route="retrieve",
        gold_doc_ids=["paper.pdf"],
        gold_node_ids=["node-7"],
    )
    document = Document(
        page_content="content",
        metadata={"source": str(Path("nested") / "paper.pdf"), "node_id": "node-1"},
    )
    merged_document = Document(
        page_content="content",
        metadata={"source": "other.pdf", "merged_node_ids": ["node-7"]},
    )

    assert normalize_identifier("nested\\paper.pdf") == "paper.pdf"
    assert _document_matches_case(document, case) is True
    assert _document_matches_case(merged_document, case) is True


def test_route_accuracy_is_binary():
    assert route_accuracy("retrieve", "retrieve") == 1.0
    assert route_accuracy("retrieve", "out_of_scope") == 0.0


def test_ordered_unique_preserves_first_occurrence_order():
    assert _ordered_unique(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_build_leaderboard_sorts_variants_by_suite_score():
    leaderboard = _build_leaderboard(
        {
            "baseline_flat": {
                "suites": {
                    "retrieval": {
                        "metrics": {
                            "recall_at_k": 0.8,
                            "mrr": 0.6,
                            "ndcg": 0.7,
                            "redundancy_rate": 0.1,
                        }
                    }
                }
            },
            "hierarchical": {
                "suites": {
                    "retrieval": {
                        "metrics": {
                            "recall_at_k": 0.9,
                            "mrr": 0.8,
                            "ndcg": 0.85,
                            "redundancy_rate": 0.05,
                        }
                    }
                }
            },
        }
    )

    assert leaderboard["retrieval"][0]["variant"] == "hierarchical"
    assert leaderboard["retrieval"][0]["comparable"] is True


def test_answer_leaderboard_marks_offline_fallback_as_non_comparable():
    leaderboard = _build_leaderboard(
        {
            "baseline_flat": {
                "suites": {
                    "answer": {
                        "metrics": {
                            "groundedness": 1.0,
                            "answer_completeness": 0.3,
                            "citation_precision": 0.2,
                            "hallucination_rate_rule": 0.0,
                            "evaluation_mode": "offline_extractive_fallback",
                        }
                    }
                }
            },
            "hierarchical": {
                "suites": {
                    "answer": {
                        "metrics": {
                            "groundedness": 0.8,
                            "answer_completeness": 0.7,
                            "citation_precision": 0.6,
                            "hallucination_rate_rule": 0.1,
                            "evaluation_mode": "generative_grounded",
                        }
                    }
                }
            },
        }
    )

    assert leaderboard["answer"][0]["variant"] == "hierarchical"
    assert leaderboard["answer"][0]["comparable"] is True
    assert leaderboard["answer"][1]["comparable"] is False


def test_is_comparable_suite_result_only_flags_answer_fallback():
    assert _is_comparable_suite_result("retrieval", {"ndcg": 0.8}) is True
    assert (
        _is_comparable_suite_result(
            "answer",
            {"evaluation_mode": "offline_extractive_fallback"},
        )
        is False
    )


def test_build_offline_extractive_answer_prefers_relevant_sentences():
    answer = _build_offline_extractive_answer(
        "What does the Transformer replace recurrence with?",
        evidence_quotes=[
            "The Transformer is based solely on attention mechanisms. It removes recurrence and convolutions entirely.",
            "This unrelated sentence discusses optimization tricks.",
        ],
    )

    assert "attention" in answer.casefold()
    assert "recurrence" in answer.casefold()
    assert "unrelated sentence" not in answer.casefold()
