"""Deterministic M3.1 experiment generation and candidate selection."""

from __future__ import annotations

from dataclasses import replace
import random
from typing import Any

from indexing.retrieval_pipeline import (
    RetrievalPipelineConfig,
    get_pipeline_config,
)


RERANKER_MODELS = (
    "ms-marco-TinyBERT-L-2-v2",
    "ms-marco-MiniLM-L-12-v2",
    "ms-marco-MultiBERT-L-12",
    "rank-T5-flan",
)
BLEND_WEIGHTS = (
    (0.75, 0.25),
    (0.50, 0.50),
    (0.25, 0.75),
)


def pipeline_from_dict(payload: dict[str, Any]) -> RetrievalPipelineConfig:
    values = dict(payload)
    values["metadata_prefix_fields"] = tuple(
        values["metadata_prefix_fields"]
    )
    return RetrievalPipelineConfig(**values)


def round1_seed_configs() -> dict[str, RetrievalPipelineConfig]:
    base = get_pipeline_config("b2_no_rerank")
    common = {
        "use_metadata_prefix": False,
        "dense_use_metadata_prefix": False,
        "sparse_use_metadata_prefix": False,
        "tokenizer": "mixed_v1",
        "use_rerank": False,
        "neighbor_window": 0,
    }
    return {
        "r1_01_quote_mixed_minmax": replace(
            base,
            name="m3_1_r1_quote_mixed_minmax",
            fusion_method="minmax",
            **common,
        ),
        "r1_02_quote_mixed_rrf": replace(
            base,
            name="m3_1_r1_quote_mixed_rrf",
            fusion_method="rrf",
            **common,
        ),
        "r1_03_section_mixed_rrf": replace(
            base,
            name="m3_1_r1_section_mixed_rrf",
            fusion_method="rrf",
            use_metadata_prefix=True,
            dense_use_metadata_prefix=True,
            sparse_use_metadata_prefix=True,
            metadata_prefix_fields=("section",),
            tokenizer="mixed_v1",
            use_rerank=False,
            neighbor_window=0,
        ),
        "r1_04_title_section_mixed_rrf": replace(
            base,
            name="m3_1_r1_title_section_mixed_rrf",
            fusion_method="rrf",
            use_metadata_prefix=True,
            dense_use_metadata_prefix=True,
            sparse_use_metadata_prefix=True,
            metadata_prefix_fields=("title", "section"),
            tokenizer="mixed_v1",
            use_rerank=False,
            neighbor_window=0,
        ),
        "r1_05_full_metadata_mixed_rrf": replace(
            base,
            name="m3_1_r1_full_metadata_mixed_rrf",
            fusion_method="rrf",
            use_metadata_prefix=True,
            dense_use_metadata_prefix=True,
            sparse_use_metadata_prefix=True,
            metadata_prefix_fields=(
                "title",
                "authors",
                "year",
                "section",
                "block",
            ),
            tokenizer="mixed_v1",
            use_rerank=False,
            neighbor_window=0,
        ),
    }


def round1_boost_off(
    best_key: str,
    best: RetrievalPipelineConfig,
) -> tuple[str, RetrievalPipelineConfig]:
    return (
        "r1_06_best_boost_off",
        replace(
            best,
            name=f"{best.name}_boost_off",
            boost_policy="off",
        ),
    )


def round2_configs(
    finalists: list[tuple[str, RetrievalPipelineConfig]],
) -> dict[str, RetrievalPipelineConfig]:
    output: dict[str, RetrievalPipelineConfig] = {}
    for finalist_index, (_, pipeline) in enumerate(finalists, start=1):
        for model_index, model in enumerate(RERANKER_MODELS, start=1):
            key = f"r2_{finalist_index}_{model_index}"
            output[key] = replace(
                pipeline,
                name=f"m3_1_{key}",
                use_rerank=True,
                reranker_model=model,
                rerank_input="quote",
                rerank_merge_mode="replace",
            )
    return output


def round3_blend_configs(
    finalists: list[tuple[str, RetrievalPipelineConfig]],
) -> dict[str, RetrievalPipelineConfig]:
    output: dict[str, RetrievalPipelineConfig] = {}
    for finalist_index, (_, pipeline) in enumerate(finalists, start=1):
        for weight_index, (
            fusion_weight,
            rerank_weight,
        ) in enumerate(BLEND_WEIGHTS, start=1):
            key = f"r3_{finalist_index}_{weight_index}"
            output[key] = replace(
                pipeline,
                name=f"m3_1_{key}",
                rerank_merge_mode="weighted_rrf",
                fusion_rank_weight=fusion_weight,
                rerank_rank_weight=rerank_weight,
            )
    return output


def round3_stability_configs(
    best_blended: RetrievalPipelineConfig,
) -> dict[str, RetrievalPipelineConfig]:
    return {
        "r3_07_title_section_quote": replace(
            best_blended,
            name="m3_1_r3_title_section_quote",
            rerank_input="title_section_quote",
        ),
        "r3_08_retrieval_input": replace(
            best_blended,
            name="m3_1_r3_retrieval_input",
            rerank_input="retrieval",
        ),
        "r3_09_dense_1_25_sparse_0_75": replace(
            best_blended,
            name="m3_1_r3_dense_1_25_sparse_0_75",
            dense_rrf_weight=1.25,
            sparse_rrf_weight=0.75,
        ),
        "r3_10_dense_0_75_sparse_1_25": replace(
            best_blended,
            name="m3_1_r3_dense_0_75_sparse_1_25",
            dense_rrf_weight=0.75,
            sparse_rrf_weight=1.25,
        ),
    }


def pairwise(
    baseline_cases: list[dict[str, Any]],
    candidate_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = {row["case_id"]: row for row in baseline_cases}
    candidate = {row["case_id"]: row for row in candidate_cases}
    if baseline.keys() != candidate.keys():
        raise ValueError("Pairwise reports use different retrieval cases.")
    rows: list[dict[str, Any]] = []
    for case_id, left in baseline.items():
        right = candidate[case_id]
        left_rank = left["first_gold_rank"] or 1000
        right_rank = right["first_gold_rank"] or 1000
        outcome = (
            "win"
            if right_rank < left_rank
            else "loss"
            if right_rank > left_rank
            else "tie"
        )
        rows.append(
            {
                "case_id": case_id,
                "category": left["category"],
                "fold": _case_fold(case_id, left["category"], baseline),
                "baseline_rank": left["first_gold_rank"],
                "candidate_rank": right["first_gold_rank"],
                "outcome": outcome,
            }
        )
    return {
        "wins": sum(row["outcome"] == "win" for row in rows),
        "ties": sum(row["outcome"] == "tie" for row in rows),
        "losses": sum(row["outcome"] == "loss" for row in rows),
        "cases": rows,
    }


def rank_candidates(
    reports: dict[str, dict[str, Any]],
    *,
    baseline: dict[str, Any],
) -> list[str]:
    """Rank candidates lexicographically without an aggregate score."""
    eligible = {
        key: report
        for key, report in reports.items()
        if float(report["metrics"]["recall_at_10"])
        >= float(baseline["metrics"]["recall_at_10"])
    }

    def ranking_key(key: str) -> tuple[Any, ...]:
        report = eligible[key]
        comparison = pairwise(baseline["cases"], report["cases"])
        subset_deltas = [
            int(report["subsets"][category]["recall_at_10_hit_count"])
            - int(
                baseline["subsets"][category][
                    "recall_at_10_hit_count"
                ]
            )
            for category in baseline["subsets"]
        ]
        return (
            comparison["losses"],
            -comparison["wins"],
            -float(report["metrics"]["recall_at_10"]),
            -min(subset_deltas),
            -float(report["metrics"]["ndcg_at_10"]),
            float(report["metrics"]["p95_latency_ms"]),
            key,
        )

    return sorted(eligible, key=ranking_key)


def dev_gate(
    candidate: dict[str, Any],
    *,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    comparison = pairwise(baseline["cases"], candidate["cases"])
    subset_deltas = {
        category: (
            int(candidate["subsets"][category]["recall_at_10_hit_count"])
            - int(
                baseline["subsets"][category][
                    "recall_at_10_hit_count"
                ]
            )
        )
        for category in baseline["subsets"]
    }
    winning_rows = [
        row for row in comparison["cases"] if row["outcome"] == "win"
    ]
    winning_folds = {row["fold"] for row in winning_rows}
    winning_categories = {row["category"] for row in winning_rows}
    b1_p95 = float(baseline["metrics"]["p95_latency_ms"])
    candidate_p95 = float(candidate["metrics"]["p95_latency_ms"])
    latency_ratio = (
        candidate_p95 / b1_p95
        if b1_p95 > 0
        else (1.0 if candidate_p95 == 0 else float("inf"))
    )
    checks = {
        "recall_delta_at_least_0_02": (
            float(candidate["metrics"]["recall_at_10"])
            - float(baseline["metrics"]["recall_at_10"])
            >= 0.02
        ),
        "wins_at_least_10": comparison["wins"] >= 10,
        "losses_at_most_3": comparison["losses"] <= 3,
        "mrr_not_lower": (
            float(candidate["metrics"]["mrr_at_10"])
            >= float(baseline["metrics"]["mrr_at_10"])
        ),
        "ndcg_not_lower": (
            float(candidate["metrics"]["ndcg_at_10"])
            >= float(baseline["metrics"]["ndcg_at_10"])
        ),
        "each_subset_declines_at_most_1": min(subset_deltas.values()) >= -1,
        "p95_ratio_at_most_1_35": latency_ratio <= 1.35,
        "wins_span_multiple_folds": len(winning_folds) >= 2,
        "wins_span_multiple_categories": len(winning_categories) >= 2,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "pairwise": comparison,
        "subset_hit_deltas": subset_deltas,
        "winning_folds": sorted(winning_folds),
        "winning_categories": sorted(winning_categories),
        "p95_latency_ratio": round(latency_ratio, 6),
    }


def final_gate(
    candidate: dict[str, Any],
    *,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    comparison = pairwise(baseline["cases"], candidate["cases"])
    subset_deltas = {
        category: (
            int(candidate["subsets"][category]["recall_at_10_hit_count"])
            - int(
                baseline["subsets"][category][
                    "recall_at_10_hit_count"
                ]
            )
        )
        for category in baseline["subsets"]
    }
    b1_p95 = float(baseline["metrics"]["p95_latency_ms"])
    candidate_p95 = float(candidate["metrics"]["p95_latency_ms"])
    latency_ratio = (
        candidate_p95 / b1_p95
        if b1_p95 > 0
        else (1.0 if candidate_p95 == 0 else float("inf"))
    )
    checks = {
        "recall_not_lower": (
            float(candidate["metrics"]["recall_at_10"])
            >= float(baseline["metrics"]["recall_at_10"])
        ),
        "wins_at_least_8": comparison["wins"] >= 8,
        "losses_at_most_4": comparison["losses"] <= 4,
        "each_subset_declines_at_most_1": min(subset_deltas.values()) >= -1,
        "p95_ratio_at_most_1_5": latency_ratio <= 1.5,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "pairwise": comparison,
        "subset_hit_deltas": subset_deltas,
        "p95_latency_ratio": round(latency_ratio, 6),
        "paired_bootstrap_recall_delta_95": paired_bootstrap_interval(
            baseline["cases"],
            candidate["cases"],
            metric="recall_at_10",
        ),
    }


def paired_bootstrap_interval(
    baseline_cases: list[dict[str, Any]],
    candidate_cases: list[dict[str, Any]],
    *,
    metric: str,
    sample_count: int = 10_000,
    random_seed: int = 31,
) -> dict[str, float | int]:
    baseline = {row["case_id"]: row for row in baseline_cases}
    candidate = {row["case_id"]: row for row in candidate_cases}
    if baseline.keys() != candidate.keys():
        raise ValueError("Bootstrap reports use different retrieval cases.")
    case_ids = sorted(baseline)
    deltas = [
        float(candidate[case_id][metric])
        - float(baseline[case_id][metric])
        for case_id in case_ids
    ]
    if not deltas:
        return {
            "estimate": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "sample_count": sample_count,
        }
    rng = random.Random(random_seed)
    estimates = sorted(
        sum(rng.choice(deltas) for _ in deltas) / len(deltas)
        for _ in range(sample_count)
    )
    return {
        "estimate": round(sum(deltas) / len(deltas), 6),
        "lower": round(_percentile(estimates, 0.025), 6),
        "upper": round(_percentile(estimates, 0.975), 6),
        "sample_count": sample_count,
    }


def pareto_frontier(
    reports: dict[str, dict[str, Any]],
    *,
    baseline: dict[str, Any],
) -> list[str]:
    rows: dict[str, tuple[float, int, int, float]] = {}
    for key, report in reports.items():
        comparison = pairwise(baseline["cases"], report["cases"])
        rows[key] = (
            float(report["metrics"]["recall_at_10"]),
            comparison["wins"],
            -comparison["losses"],
            -float(report["metrics"]["p95_latency_ms"]),
        )
    frontier: list[str] = []
    for key, values in rows.items():
        dominated = any(
            other != key
            and all(left >= right for left, right in zip(other_values, values))
            and any(left > right for left, right in zip(other_values, values))
            for other, other_values in rows.items()
        )
        if not dominated:
            frontier.append(key)
    return sorted(frontier)


def _case_fold(
    case_id: str,
    category: str,
    cases: dict[str, dict[str, Any]],
) -> int:
    category_ids = sorted(
        current_id
        for current_id, row in cases.items()
        if row["category"] == category
    )
    return category_ids.index(case_id) % 4


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction
