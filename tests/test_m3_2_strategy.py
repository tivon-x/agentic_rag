from dataclasses import replace
from pathlib import Path

import yaml

from evals.m3_1_experiments import round1_seed_configs
from evals.m3_2_strategy import _m4_difficult_cases, strategy_gate
from evals.v2_corpus import sha256_file
from indexing.retrieval_pipeline import get_pipeline_config


def test_s1_matches_frozen_m3_1_source_configuration():
    source = round1_seed_configs()["r1_01_quote_mixed_minmax"]
    expected = replace(source, name="v2_fixed_hybrid")

    assert get_pipeline_config("s1") == expected
    assert get_pipeline_config("v2_fixed_hybrid") == expected
    assert get_pipeline_config("s1").use_rerank is False


def test_m3_2_config_freezes_only_b1_and_s1_with_expected_hashes():
    repo_root = Path(__file__).resolve().parent.parent
    config = yaml.safe_load(
        (repo_root / "evals/configs/v2_m3_2_strategy.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["pipelines"] == ["b1", "s1"]
    assert sha256_file(repo_root / config["retrieval_dataset"]) == config[
        "retrieval_dataset_sha256"
    ]
    assert sha256_file(repo_root / config["holdout_dataset"]) == config[
        "holdout_dataset_sha256"
    ]
    assert config["retrieval_dataset_sha256"] != config[
        "holdout_dataset_sha256"
    ]


def test_strategy_gate_requires_every_frozen_quality_and_latency_check():
    baseline = _report()
    candidate = _report(
        recall=0.8,
        mrr=0.8,
        ndcg=0.8,
        p95=90.0,
        context=0.8,
        subset_hits=12,
        improved=10,
    )

    gate = strategy_gate(candidate, baseline=baseline)

    assert gate["passed"] is True
    assert gate["pairwise"]["wins"] == 10
    assert gate["pairwise"]["losses"] == 0

    candidate["metrics"]["p95_latency_ms"] = 101.0
    assert strategy_gate(candidate, baseline=baseline)["passed"] is False


def test_m4_difficult_cases_only_exposes_old_dev_s1_regressions():
    report = {
        "datasets": {
            "old_dev": {
                "gate": {
                    "pairwise": {
                        "cases": [
                            {
                                "case_id": "loss",
                                "category": "experiment_number_table",
                                "baseline_rank": 1,
                                "candidate_rank": 2,
                                "outcome": "loss",
                            },
                            {
                                "case_id": "win",
                                "category": "cross_paper_or_section",
                                "baseline_rank": 2,
                                "candidate_rank": 1,
                                "outcome": "win",
                            },
                        ]
                    }
                },
                "pipelines": {"s1": {"retrieval": {"cases": [
                    {
                        "case_id": "loss",
                        "question": "number question",
                        "tags": ["表格", "数字"],
                        "context_passage_recall": 0.0,
                        "stage_results": {},
                    },
                    {
                        "case_id": "win",
                        "question": "cross question",
                        "tags": ["跨论文"],
                        "context_passage_recall": 1.0,
                        "stage_results": {},
                    },
                ]}}},
            }
        }
    }

    difficult = _m4_difficult_cases(report)

    assert [row["case_id"] for row in difficult] == ["loss"]
    assert difficult[0]["observability_signals"]["table_or_number_localization"]
    assert difficult[0]["observability_signals"]["first_context_incomplete"]


def _report(
    *,
    recall: float = 0.7,
    mrr: float = 0.7,
    ndcg: float = 0.7,
    p95: float = 100.0,
    context: float = 0.7,
    subset_hits: int = 10,
    improved: int = 0,
) -> dict:
    categories = (
        "exact_term_definition",
        "method_section_location",
        "experiment_number_table",
        "cross_paper_or_section",
    )
    cases = []
    for index in range(12):
        for category in categories:
            rank = 1 if len(cases) < improved else None
            cases.append(
                {
                    "case_id": f"{category}-{index}",
                    "category": category,
                    "first_gold_rank": rank,
                }
            )
    return {
        "metrics": {
            "recall_at_10": recall,
            "mrr_at_10": mrr,
            "ndcg_at_10": ndcg,
            "p95_latency_ms": p95,
            "context_passage_recall": context,
        },
        "subsets": {
            category: {"recall_at_10_hit_count": subset_hits}
            for category in categories
        },
        "cases": cases,
    }
