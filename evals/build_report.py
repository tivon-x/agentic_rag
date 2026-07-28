"""Build the M3 decision report without collapsing subset regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CORE_KEYS = ("b0", "b1", "b2", "b3")
ABLATION_KEYS = (
    "b2_no_metadata",
    "b2_no_sparse",
    "b2_no_dense",
    "b2_minmax",
    "b2_no_rerank",
)


def build_report(runs_dir: Path) -> dict[str, Any]:
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(runs_dir.glob("*/report.json"))
    ]
    pipelines: dict[str, dict[str, Any]] = {}
    common_contract: dict[str, Any] | None = None
    for report in reports:
        contract = {
            "parser_artifact_sha256": report["parser_artifact_sha256"],
            "retrieval_dataset_sha256": report[
                "retrieval_dataset_sha256"
            ],
            "answer_smoke_dataset_sha256": report[
                "answer_smoke_dataset_sha256"
            ],
            "embedding": report["embedding"],
            "reranker": report["reranker"],
            "retrieval_evaluation": report["retrieval_evaluation"],
        }
        if common_contract is None:
            common_contract = contract
        elif contract != common_contract:
            raise ValueError(
                "M3 reports do not share one frozen parser, embedding, "
                "reranker, and test set."
            )
        for key, pipeline in report["pipelines"].items():
            if key in pipelines:
                raise ValueError(f"Duplicate pipeline report: {key}.")
            pipelines[key] = pipeline

    missing = [
        key
        for key in (*CORE_KEYS, *ABLATION_KEYS)
        if key not in pipelines
    ]
    if missing:
        raise ValueError(f"Missing M3 pipeline reports: {missing}.")

    b1 = pipelines["b1"]["retrieval"]
    b2 = pipelines["b2"]["retrieval"]
    b3 = pipelines["b3"]["retrieval"]
    b1_b2 = _pairwise(b1["cases"], b2["cases"])
    b2_b3 = _pairwise(b2["cases"], b3["cases"])

    subset_declines = {
        category: (
            b1["subsets"][category]["recall_at_10_hit_count"]
            - b2["subsets"][category]["recall_at_10_hit_count"]
        )
        for category in b1["subsets"]
    }
    b1_p95 = float(b1["metrics"]["p95_latency_ms"])
    b2_p95 = float(b2["metrics"]["p95_latency_ms"])
    latency_ratio = (
        b2_p95 / b1_p95
        if b1_p95 > 0
        else (1.0 if b2_p95 == 0 else float("inf"))
    )
    b2_gate_checks = {
        "recall_at_10_not_lower": (
            b2["metrics"]["recall_at_10"]
            >= b1["metrics"]["recall_at_10"]
        ),
        "gold_rank_improvements_at_least_8": (
            b1_b2["wins"] >= 8
        ),
        "gold_rank_regressions_at_most_4": (
            b1_b2["losses"] <= 4
        ),
        "no_subset_declines_by_2_or_more": all(
            decline < 2 for decline in subset_declines.values()
        ),
        "p95_latency_ratio_at_most_1_5": latency_ratio <= 1.5,
    }
    b2_passed = all(b2_gate_checks.values())

    cross_category = "cross_paper_or_section"
    cross_wins = sum(
        row["outcome"] == "win"
        for row in b2_b3["cases"]
        if row["category"] == cross_category
    )
    other_losses = sum(
        row["outcome"] == "loss"
        for row in b2_b3["cases"]
        if row["category"] != cross_category
    )
    b3_gate_checks = {
        "b2_gate_passed": b2_passed,
        "cross_section_improvements_at_least_3": cross_wins >= 3,
        "other_subset_regressions_at_most_1": other_losses <= 1,
    }
    b3_passed = all(b3_gate_checks.values())
    default_pipeline = (
        "b3" if b3_passed else "b2" if b2_passed else "b1"
    )

    ablations = {
        key: {
            "metrics": pipelines[key]["retrieval"]["metrics"],
            "delta_vs_b2": _metric_delta(
                pipelines[key]["retrieval"]["metrics"],
                b2["metrics"],
            ),
            "pairwise_vs_b2": _pairwise(
                b2["cases"],
                pipelines[key]["retrieval"]["cases"],
            ),
        }
        for key in ABLATION_KEYS
    }
    output = {
        "schema_version": 2,
        "frozen_contract": common_contract,
        "core_metrics": {
            key: pipelines[key]["retrieval"]["metrics"]
            for key in CORE_KEYS
        },
        "subset_metrics": {
            key: pipelines[key]["retrieval"]["subsets"]
            for key in CORE_KEYS
        },
        "b1_vs_b2": b1_b2,
        "b2_vs_b3": b2_b3,
        "b2_gate": {
            "passed": b2_passed,
            "checks": b2_gate_checks,
            "subset_recall_hit_declines": subset_declines,
            "p95_latency_ratio": round(latency_ratio, 6),
        },
        "b3_gate": {
            "passed": b3_passed,
            "checks": b3_gate_checks,
            "cross_section_wins": cross_wins,
            "other_subset_losses": other_losses,
        },
        "ablations": ablations,
        "bad_cases": {
            key: pipelines[key]["retrieval"]["bad_cases"]
            for key in CORE_KEYS
        },
        "answer_smoke": {
            key: pipelines[key]["answer_smoke"]
            for key in CORE_KEYS
        },
        "default_pipeline": default_pipeline,
        "core_passed": b2_passed,
        "m4_entry_ready": b2_passed,
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    json_path = runs_dir / "core_report.json"
    markdown_path = runs_dir / "core_report.md"
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _render_markdown(output),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report_json": str(json_path),
                "report_markdown": str(markdown_path),
                "core_passed": b2_passed,
                "default_pipeline": default_pipeline,
                "m4_entry_ready": b2_passed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return output


def _pairwise(
    baseline_cases: list[dict[str, Any]],
    candidate_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = {row["case_id"]: row for row in baseline_cases}
    candidate = {row["case_id"]: row for row in candidate_cases}
    if baseline.keys() != candidate.keys():
        raise ValueError("Pairwise reports use different retrieval cases.")
    rows: list[dict[str, Any]] = []
    for case_id in baseline:
        left = baseline[case_id]
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


def _metric_delta(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float]:
    return {
        key: round(float(candidate[key]) - float(baseline[key]), 6)
        for key in (
            "recall_at_5",
            "recall_at_10",
            "mrr_at_10",
            "ndcg_at_10",
            "context_passage_recall",
            "paper_recall_at_10",
            "section_recall_at_10",
            "p50_latency_ms",
            "p95_latency_ms",
        )
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Agentic RAG V2 Core Retrieval Report",
        "",
        f"- Core passed: `{report['core_passed']}`",
        f"- Default pipeline: `{report['default_pipeline']}`",
        f"- M4 entry ready: `{report['m4_entry_ready']}`",
        "",
        "## Core metrics",
        "",
        "| Pipeline | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | "
        "Context Recall | Paper Recall@10 | Section Recall@10 | p50 ms | p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in CORE_KEYS:
        metrics = report["core_metrics"][key]
        lines.append(
            f"| {key} | {metrics['recall_at_5']} | "
            f"{metrics['recall_at_10']} | {metrics['mrr_at_10']} | "
            f"{metrics['ndcg_at_10']} | "
            f"{metrics['context_passage_recall']} | "
            f"{metrics['paper_recall_at_10']} | "
            f"{metrics['section_recall_at_10']} | "
            f"{metrics['p50_latency_ms']} | "
            f"{metrics['p95_latency_ms']} |"
        )
    lines.extend(
        [
            "",
            "## Gate decisions",
            "",
            f"- B2 gate: `{report['b2_gate']['passed']}` "
            f"{json.dumps(report['b2_gate']['checks'], ensure_ascii=False)}",
            f"- B1/B2 W/T/L: "
            f"`{report['b1_vs_b2']['wins']}/"
            f"{report['b1_vs_b2']['ties']}/"
            f"{report['b1_vs_b2']['losses']}`",
            f"- B3 gate: `{report['b3_gate']['passed']}` "
            f"{json.dumps(report['b3_gate']['checks'], ensure_ascii=False)}",
            "",
            "## Ablations",
            "",
        ]
    )
    for key, value in report["ablations"].items():
        lines.append(
            f"- `{key}` delta vs B2: "
            f"`{json.dumps(value['delta_vs_b2'], ensure_ascii=False)}`"
        )
    lines.append("")
    lines.append(
        "No aggregate composite score is used; subset metrics and bad cases "
        "remain in the JSON report."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    args = parser.parse_args()
    build_report(Path(args.runs).resolve())


if __name__ == "__main__":
    main()
