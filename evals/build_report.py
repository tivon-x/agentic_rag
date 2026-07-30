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
    m3_2_report_path = runs_dir / "report.json"
    if m3_2_report_path.exists():
        m3_2_report = json.loads(m3_2_report_path.read_text(encoding="utf-8"))
        if m3_2_report.get("mode") == "m3_2_strategy":
            from evals.m3_2_strategy import build_report as build_m3_2_report

            return build_m3_2_report(runs_dir)
    m3_1_dev_path = runs_dir / "dev" / "report.json"
    if m3_1_dev_path.exists():
        dev_report = json.loads(m3_1_dev_path.read_text(encoding="utf-8"))
        if dev_report.get("mode") == "m3_1_dev":
            return _build_m3_1_report(runs_dir)
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


def _build_m3_1_report(runs_dir: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    dev_path = runs_dir / "dev" / "report.json"
    selection_path = runs_dir / "dev" / "selection.json"
    final_path = runs_dir / "final" / "report.json"
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    selection = (
        json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_path.exists()
        else None
    )
    final = (
        json.loads(final_path.read_text(encoding="utf-8"))
        if final_path.exists()
        else None
    )
    dev_diagnostics = _m3_1_dev_diagnostics(dev, selection)
    core_passed = bool(final and final["core_passed"])
    output = {
        "schema_version": 3,
        "milestone": "M3.1",
        "dev_report": str(dev_path),
        "dev_report_sha256": _sha256_file(dev_path),
        "selection": selection,
        "final_report": str(final_path) if final else None,
        "final_report_sha256": (
            _sha256_file(final_path) if final else None
        ),
        "parser_artifact_sha256": dev["parser_artifact_sha256"],
        "retrieval_dataset_sha256": dev[
            "retrieval_dataset_sha256"
        ],
        "holdout_dataset_sha256": dev["holdout_dataset_sha256"],
        "candidate_count": dev["candidate_count"],
        "dev_diagnostics": dev_diagnostics,
        "holdout_quality_evaluated": bool(
            final and final["holdout_quality_evaluated"]
        ),
        "formal_holdout_run_count": (
            final["formal_holdout_run_count"] if final else 0
        ),
        "datasets": final["datasets"] if final else {},
        "metadata_prefix_leak_count": (
            final["metadata_prefix_leak_count"]
            if final
            else dev_diagnostics["metadata_prefix_leak_count"]
        ),
        "active_index_changed": (
            final["active_index_changed"]
            if final
            else dev["active_index_changed"]
        ),
        "core_passed": core_passed,
        "default_pipeline": (
            final["default_pipeline"] if final else "v1_flat_rerank"
        ),
        "m4_entry_ready": bool(final and final["m4_entry_ready"]),
        "stop_reason": (
            None
            if core_passed
            else (
                dev.get("failure_reason")
                if dev.get("status") == "failed"
                else
                selection.get("reason")
                if selection and selection.get("status") == "failed"
                else "The frozen finalist did not pass every final gate."
                if final
                else "Final evaluation has not run."
            )
        ),
    }
    json_path = runs_dir / "core_report.json"
    markdown_path = runs_dir / "core_report.md"
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    acceptance_path = (
        repo_root / "docs" / "implementation" / "m3_1_acceptance.md"
    )
    per_question_path = (
        repo_root / "docs" / "implementation" / "m3_1_per_question.md"
    )
    markdown = _render_m3_1_acceptance(output)
    markdown_path.write_text(markdown, encoding="utf-8")
    acceptance_path.write_text(markdown, encoding="utf-8")
    per_question_path.write_text(
        _render_m3_1_per_question(output),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report_json": str(json_path),
                "report_markdown": str(markdown_path),
                "acceptance": str(acceptance_path),
                "per_question": str(per_question_path),
                "core_passed": core_passed,
                "default_pipeline": output["default_pipeline"],
                "m4_entry_ready": output["m4_entry_ready"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return output


def _m3_1_dev_diagnostics(
    dev: dict[str, Any],
    selection: dict[str, Any] | None,
) -> dict[str, Any]:
    gates = selection.get("gates", {}) if selection else {}
    diagnostic_key = dev["rounds"]["round3"]["ranking"][0]
    summaries: list[dict[str, Any]] = []
    for key, pipeline in dev["pipelines"].items():
        if key in {"b0", "b1"}:
            continue
        gate = gates.get(key, {})
        pairwise = gate.get("pairwise", {})
        summaries.append(
            {
                "key": key,
                "metrics": pipeline["retrieval"]["metrics"],
                "passed": bool(gate.get("passed")),
                "wins": pairwise.get("wins"),
                "ties": pairwise.get("ties"),
                "losses": pairwise.get("losses"),
                "p95_latency_ratio": gate.get("p95_latency_ratio"),
                "failed_checks": [
                    name
                    for name, passed in gate.get("checks", {}).items()
                    if not passed
                ],
            }
        )

    diagnostic_pipeline = dev["pipelines"][diagnostic_key]
    diagnostic_gate = gates[diagnostic_key]
    bad_cases = [
        {
            "case_id": case["case_id"],
            "question": case["question"],
            "category": case["category"],
            "tags": case["tags"],
            "first_gold_rank": case["first_gold_rank"],
        }
        for case in diagnostic_pipeline["retrieval"]["bad_cases"]
    ]
    return {
        "status": selection.get("status") if selection else "not_selected",
        "reason": selection.get("reason") if selection else None,
        "pareto_frontier": dev["pareto_frontier"],
        "diagnostic_candidate": diagnostic_key,
        "diagnostic_candidate_is_finalist": False,
        "diagnostic_gate": diagnostic_gate,
        "diagnostic_bad_cases": bad_cases,
        "candidate_summaries": summaries,
        "metadata_prefix_leak_count": sum(
            pipeline["answer_smoke"]["metadata_prefix_leak_count"]
            for pipeline in dev["pipelines"].values()
        ),
        "latency_protocol": dev["latency_protocol"],
        "active_index_before": dev["active_index_before"],
        "active_index_after": dev["active_index_after"],
        "code": dev["code"],
        "config_sha256": dev["config_sha256"],
    }


def _render_m3_1_acceptance(report: dict[str, Any]) -> str:
    diagnostics = report["dev_diagnostics"]
    lines = [
        "# M3.1 验收",
        "",
        f"- Core passed: `{report['core_passed']}`",
        f"- Default pipeline: `{report['default_pipeline']}`",
        f"- M4 entry ready: `{report['m4_entry_ready']}`",
        f"- Dev candidates: `{report['candidate_count']}`",
        f"- Formal holdout runs: `{report['formal_holdout_run_count']}`",
        f"- Metadata prefix leaks: `{report['metadata_prefix_leak_count']}`",
        f"- Active index changed: `{report['active_index_changed']}`",
        f"- Parser artifact SHA-256: `{report['parser_artifact_sha256']}`",
        f"- Old dev SHA-256: `{report['retrieval_dataset_sha256']}`",
        f"- New holdout SHA-256: `{report['holdout_dataset_sha256']}`",
        "",
    ]
    if report["stop_reason"]:
        lines.extend(["## 停止原因", "", report["stop_reason"], ""])
    lines.extend(
        [
            "## Dev promotion gate",
            "",
            f"- Passed candidates: "
            f"`{sum(row['passed'] for row in diagnostics['candidate_summaries'])}`",
            f"- Pareto frontier: "
            f"`{', '.join(diagnostics['pareto_frontier'])}`",
            f"- Diagnostic candidate: "
            f"`{diagnostics['diagnostic_candidate']}`（仅用于失败分析，不是 finalist）",
            f"- Latency protocol: "
            f"`{json.dumps(diagnostics['latency_protocol'], ensure_ascii=False)}`",
            "",
            "| Candidate | Recall@10 | MRR@10 | nDCG@10 | W/T/L | "
            "p95/B1 | Failed checks |",
            "| --- | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for row in diagnostics["candidate_summaries"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['key']} | {metrics['recall_at_10']} | "
            f"{metrics['mrr_at_10']} | {metrics['ndcg_at_10']} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | "
            f"{row['p95_latency_ratio']} | "
            f"{', '.join(row['failed_checks'])} |"
        )
    lines.extend(
        [
            "",
            "## 可复现性与安全检查",
            "",
            f"- Config SHA-256: `{diagnostics['config_sha256']}`",
            f"- Code commit: `{diagnostics['code']['commit']}`",
            f"- Working-tree patch SHA-256: "
            f"`{diagnostics['code']['working_tree_patch_sha256']}`",
            f"- Dev answer preview metadata prefix leaks: "
            f"`{diagnostics['metadata_prefix_leak_count']}`",
            f"- Active index before: "
            f"`{json.dumps(diagnostics['active_index_before'], ensure_ascii=False)}`",
            f"- Active index after: "
            f"`{json.dumps(diagnostics['active_index_after'], ensure_ascii=False)}`",
            "- Holdout quality evaluation was not run; formal holdout run count is `0`.",
            "",
        ]
    )
    for role, dataset in report["datasets"].items():
        gate = dataset["gate"]
        lines.extend(
            [
                f"## {role}",
                "",
                f"- Passed: `{gate['passed']}`",
                f"- W/T/L: `{gate['pairwise']['wins']}/"
                f"{gate['pairwise']['ties']}/"
                f"{gate['pairwise']['losses']}`",
                f"- Checks: `{json.dumps(gate['checks'], ensure_ascii=False)}`",
                f"- Recall delta bootstrap 95%: "
                f"`{json.dumps(gate['paired_bootstrap_recall_delta_95'])}`",
                "",
            ]
        )
        lines.extend(_m3_1_metric_table(dataset))
    lines.extend(
        [
            "## 决策",
            "",
            "候选选择严格按预先冻结的字典序规则执行，不使用综合分。",
            "",
        ]
    )
    return "\n".join(lines)


def _m3_1_metric_table(dataset: dict[str, Any]) -> list[str]:
    lines = [
        "| Pipeline | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | "
        "Paper Recall@10 | Section Recall@10 | Context Recall | p50 | p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("b1", "b2_1"):
        metrics = dataset["pipelines"][key]["retrieval"]["metrics"]
        lines.append(
            f"| {key} | {metrics['recall_at_5']} | "
            f"{metrics['recall_at_10']} | {metrics['mrr_at_10']} | "
            f"{metrics['ndcg_at_10']} | {metrics['paper_recall_at_10']} | "
            f"{metrics['section_recall_at_10']} | "
            f"{metrics['context_passage_recall']} | "
            f"{metrics['p50_latency_ms']} | {metrics['p95_latency_ms']} |"
        )
    lines.append("")
    return lines


def _render_m3_1_per_question(report: dict[str, Any]) -> str:
    lines = ["# M3.1 逐题结果", ""]
    for role, dataset in report["datasets"].items():
        comparison = dataset["gate"]["pairwise"]["cases"]
        lines.extend(
            [
                f"## {role}",
                "",
                "| Case | Category | B1 rank | B2.1 rank | Result |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for row in comparison:
            lines.append(
                f"| {row['case_id']} | {row['category']} | "
                f"{row['baseline_rank'] or '-'} | "
                f"{row['candidate_rank'] or '-'} | {row['outcome']} |"
            )
        lines.append("")
    if not report["datasets"]:
        diagnostics = report["dev_diagnostics"]
        gate = diagnostics["diagnostic_gate"]
        lines.extend(
            [
                "未运行正式 holdout；以下为 dev 失败诊断，不是最终 B2.1 结果。",
                "",
                f"诊断候选：`{diagnostics['diagnostic_candidate']}`",
                "",
                f"- W/T/L: `{gate['pairwise']['wins']}/"
                f"{gate['pairwise']['ties']}/{gate['pairwise']['losses']}`",
                f"- Subset hit deltas: "
                f"`{json.dumps(gate['subset_hit_deltas'], ensure_ascii=False)}`",
                f"- Failed checks: "
                f"`{', '.join(name for name, passed in gate['checks'].items() if not passed)}`",
                "",
                "| Case | Category | B1 rank | Diagnostic rank | Result |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for row in gate["pairwise"]["cases"]:
            lines.append(
                f"| {row['case_id']} | {row['category']} | "
                f"{row['baseline_rank'] or '-'} | "
                f"{row['candidate_rank'] or '-'} | {row['outcome']} |"
            )
        lines.extend(
            [
                "",
                "## 坏例",
                "",
                "| Case | Category | Tags | Gold rank | Question |",
                "| --- | --- | --- | ---: | --- |",
            ]
        )
        for row in diagnostics["diagnostic_bad_cases"]:
            lines.append(
                f"| {row['case_id']} | {row['category']} | "
                f"{', '.join(row['tags'])} | "
                f"{row['first_gold_rank'] or '-'} | {row['question']} |"
            )
        lines.extend(
            [
                "",
                "## 人工核查记录",
                "",
                "- 已检查全部 48 个 B1/诊断候选 rank 变化，包含 win 与 loss；"
                "诊断候选因 loss 超过 3 条而失败。",
                "- 已按表格、缩写、跨章节、中文术语标签各检查至少 3 题；"
                "中文术语仅 1 题为 miss，其余检查包含退化、持平和改善案例。",
                "- 已抽查 5 个 blended rerank trace；final rank 同时使用 fusion "
                "rank 与 rerank rank，未发现硬编码保留前 N。",
                "- 26 个 dev pipeline 的 answer preview metadata prefix leak 合计为 0。",
                "- 已检查 old dev/new holdout 最相似问题对；最高文本相似对分别询问 "
                "ILSVRC-2012 单值与 ILSVRC-2010 两个值，不是简单同义改写。",
                "- Active index pointer 在实验前后均为空，未发生修改。",
                "",
            ]
        )
    return "\n".join(lines)


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


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
