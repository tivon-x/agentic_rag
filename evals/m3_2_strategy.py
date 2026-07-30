"""Freeze and evaluate the M3.2 fixed retrieval strategy exactly once."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import yaml

from evals.m3_1_experiments import pairwise
from evals.m3_1_runner import (
    _ExperimentRuntime,
    _active_index_snapshot,
    _capture_code_state,
    _prepare_runtime,
    _write_json_atomic,
)
from evals.run_lock import exclusive_run_lock
from evals.v2_corpus import load_parser_artifact, sha256_file
from indexing.retrieval_pipeline import get_pipeline_config


MILESTONE = "M3.2"
SOURCE_EXPERIMENT = "r1_01_quote_mixed_minmax"
DATASET_ORDER = ("holdout", "old_dev")


def create_freeze(
    config_path: Path,
    *,
    refresh_preflight_failure: bool = False,
) -> dict[str, Any]:
    """Create the immutable pre-run contract without contacting services."""
    repo_root = Path(__file__).resolve().parent.parent
    config = _load_config(config_path, repo_root=repo_root)
    run_dir = Path(config["output_dir"])
    freeze_path = run_dir / "freeze" / "manifest.json"
    if (run_dir / "report.json").exists():
        raise RuntimeError("The M3.2 strategy run was already evaluated.")
    if freeze_path.exists() and not refresh_preflight_failure:
        raise RuntimeError("The M3.2 strategy run is already frozen or evaluated.")
    if freeze_path.exists():
        failed_path = freeze_path.with_name(
            f"preflight_failed_{sha256_file(freeze_path)}.json"
        )
        freeze_path.replace(failed_path)
    _validate_inputs(config)
    active_index = _active_index_snapshot(repo_root)
    if active_index != config["active_index_baseline"]:
        raise RuntimeError("Active production index differs from the frozen baseline.")
    run_dir.mkdir(parents=True, exist_ok=True)
    code = _capture_code_state(repo_root, run_dir=run_dir)
    b1 = get_pipeline_config("b1")
    s1 = _s1_pipeline()
    manifest = {
        "schema_version": 1,
        "milestone": MILESTONE,
        "created_at": datetime.now(UTC).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "code": code,
        "parser_artifact_sha256": config["parser_artifact_sha256"],
        "old_dev_dataset_sha256": config["retrieval_dataset_sha256"],
        "holdout_dataset_sha256": config["holdout_dataset_sha256"],
        "embedding": dict(config["embedding"]),
        "active_index_before": active_index,
        "pipelines": {
            "b1": _pipeline_snapshot(b1),
            "s1": _pipeline_snapshot(s1),
        },
        "source_experiment": SOURCE_EXPERIMENT,
        "formal_holdout_run_count": 0,
    }
    _write_json_atomic(freeze_path, manifest)
    return manifest


def run_from_config(config_path: Path) -> dict[str, Any]:
    """Run the frozen holdout then old-dev protocol once."""
    repo_root = Path(__file__).resolve().parent.parent
    config = _load_config(config_path, repo_root=repo_root)
    run_dir = Path(config["output_dir"])
    report_path = run_dir / "report.json"
    if report_path.exists():
        raise RuntimeError("The formal M3.2 holdout was already evaluated; refusing to rerun it.")
    freeze = _load_freeze(config, config_path=config_path)
    _validate_inputs(config)
    active_index_before = _active_index_snapshot(repo_root)
    if active_index_before != freeze["active_index_before"]:
        raise RuntimeError("Active production index changed after the M3.2 pre-run freeze.")

    with exclusive_run_lock(run_dir):
        code = _capture_code_state(repo_root, run_dir=run_dir)
        _validate_code_state(code, freeze["code"])
        datasets: dict[str, Any] = {}
        for role, dataset_path, dataset_sha in _dataset_specs(config):
            runtime = _prepare_runtime(
                config,
                repo_root=repo_root,
                dataset_path=dataset_path,
                expected_dataset_sha=dataset_sha,
                run_dir_override=run_dir / role,
            )
            runtime.code_state = code
            datasets[role] = _evaluate_dataset(runtime)

        metadata_prefix_leak_count = sum(
            pipeline["answer_smoke"]["metadata_prefix_leak_count"]
            for dataset in datasets.values()
            for pipeline in dataset["pipelines"].values()
        )
        answer_smoke_passed = all(
            dataset["answer_smoke_gate"]["passed"]
            for dataset in datasets.values()
        )
        active_index_after = _active_index_snapshot(repo_root)
        active_index_changed = active_index_after != active_index_before
        strategy_candidate_passed = (
            all(dataset["gate"]["passed"] for dataset in datasets.values())
            and metadata_prefix_leak_count == 0
            and answer_smoke_passed
            and not active_index_changed
        )
        selected_pipeline = (
            "v2_fixed_hybrid" if strategy_candidate_passed else "v1_flat_rerank"
        )
        report = {
            "schema_version": 1,
            "mode": "m3_2_strategy",
            "milestone": MILESTONE,
            "generated_at": datetime.now(UTC).isoformat(),
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "freeze_manifest": str(run_dir / "freeze" / "manifest.json"),
            "freeze_manifest_sha256": sha256_file(run_dir / "freeze" / "manifest.json"),
            "parser_artifact_sha256": config["parser_artifact_sha256"],
            "retrieval_dataset_sha256": config["retrieval_dataset_sha256"],
            "holdout_dataset_sha256": config["holdout_dataset_sha256"],
            "embedding": dict(config["embedding"]),
            "latency_protocol": dict(config["latency"]),
            "code": code,
            "source_experiment": SOURCE_EXPERIMENT,
            "candidate": {"key": "s1", **_pipeline_snapshot(_s1_pipeline())},
            "formal_holdout_run_count": 1,
            "datasets": datasets,
            "metadata_prefix_leak_count": metadata_prefix_leak_count,
            "answer_smoke_passed": answer_smoke_passed,
            "active_index_before": active_index_before,
            "active_index_after": active_index_after,
            "active_index_changed": active_index_changed,
            "strategy_candidate_passed": strategy_candidate_passed,
            "default_pipeline": selected_pipeline,
            "m4_fixed_baseline": selected_pipeline,
            "m3_1_core_passed": False,
            "m3_strategy_closed": True,
            "m4_entry_ready": True,
        }
        _write_json_atomic(report_path, report)
    return report


def build_report(runs_dir: Path) -> dict[str, Any]:
    """Materialize the M3.2 acceptance, per-question, and M4 bridge outputs."""
    report_path = runs_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("mode") != "m3_2_strategy":
        raise ValueError("Runs directory does not contain an M3.2 strategy report.")
    selected_key = "s1" if report["strategy_candidate_passed"] else "b1"
    selected = report["datasets"]["holdout"]["pipelines"][selected_key]
    m4_contract = {
        "schema_version": 1,
        "milestone": MILESTONE,
        "selected_pipeline_name": report["m4_fixed_baseline"],
        "pipeline_config": selected["pipeline"],
        "pipeline_config_hash": selected["pipeline_config_hash"],
        "index_contract": selected["index_contract"],
        "index_manifest_sha256": selected["manifest_sha256"],
        "parser_artifact_sha256": report["parser_artifact_sha256"],
        "embedding": report["embedding"],
        "old_dev_dataset_sha256": report["retrieval_dataset_sha256"],
        "holdout_dataset_sha256": report["holdout_dataset_sha256"],
        "code": report["code"],
        "active_index_version": report["active_index_after"],
        "quality_metrics": {
            role: dataset["pipelines"][selected_key]["retrieval"]["metrics"]
            for role, dataset in report["datasets"].items()
        },
        "latency_metrics": {
            role: dataset["pipelines"][selected_key]["retrieval"]["latency_protocol"]
            for role, dataset in report["datasets"].items()
        },
        "strategy_candidate_passed": report["strategy_candidate_passed"],
        "selection_reason": _selection_reason(report),
        "m3_1_core_passed": False,
        "m3_strategy_closed": True,
        "m4_entry_ready": True,
    }
    _write_json_atomic(runs_dir / "m4_fixed_baseline.json", m4_contract)
    core = {
        "schema_version": 1,
        "milestone": MILESTONE,
        "formal_holdout_run_count": report["formal_holdout_run_count"],
        "strategy_candidate_passed": report["strategy_candidate_passed"],
        "default_pipeline": report["default_pipeline"],
        "m4_fixed_baseline": report["m4_fixed_baseline"],
        "m3_1_core_passed": False,
        "m3_strategy_closed": True,
        "m4_entry_ready": True,
        "metadata_prefix_leak_count": report["metadata_prefix_leak_count"],
        "active_index_changed": report["active_index_changed"],
        "datasets": report["datasets"],
        "m4_difficult_cases": _m4_difficult_cases(report),
        "m4_contract": str(runs_dir / "m4_fixed_baseline.json"),
    }
    _write_json_atomic(runs_dir / "core_report.json", core)
    markdown = _render_acceptance(core)
    (runs_dir / "core_report.md").write_text(markdown, encoding="utf-8")
    repo_root = Path(__file__).resolve().parent.parent
    (repo_root / "docs" / "implementation" / "m3_2_strategy_acceptance.md").write_text(markdown, encoding="utf-8")
    (repo_root / "docs" / "implementation" / "m3_2_strategy_per_question.md").write_text(_render_per_question(core), encoding="utf-8")
    return core


def _evaluate_dataset(runtime: _ExperimentRuntime) -> dict[str, Any]:
    b1 = runtime.evaluate("b1", get_pipeline_config("b1"))
    s1 = runtime.evaluate("s1", _s1_pipeline())
    gate = strategy_gate(s1["retrieval"], baseline=b1["retrieval"])
    return {
        "dataset": str(runtime.config["retrieval_dataset"]),
        "dataset_sha256": runtime.dataset_sha,
        "pipelines": {"b1": b1, "s1": s1},
        "gate": gate,
        "answer_smoke_gate": _answer_smoke_gate(s1["answer_smoke"], baseline=b1["answer_smoke"]),
    }


def strategy_gate(candidate: dict[str, Any], *, baseline: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-declared M3.2 non-inferior, faster strategy gate."""
    comparison = pairwise(baseline["cases"], candidate["cases"])
    subset_deltas = {
        category: int(candidate["subsets"][category]["recall_at_10_hit_count"])
        - int(baseline["subsets"][category]["recall_at_10_hit_count"])
        for category in baseline["subsets"]
    }
    baseline_p95 = float(baseline["metrics"]["p95_latency_ms"])
    candidate_p95 = float(candidate["metrics"]["p95_latency_ms"])
    latency_ratio = candidate_p95 / baseline_p95 if baseline_p95 else float("inf")
    checks = {
        "recall_at_10_not_lower": float(candidate["metrics"]["recall_at_10"]) >= float(baseline["metrics"]["recall_at_10"]),
        "mrr_at_10_not_lower": float(candidate["metrics"]["mrr_at_10"]) >= float(baseline["metrics"]["mrr_at_10"]),
        "ndcg_at_10_not_lower": float(candidate["metrics"]["ndcg_at_10"]) >= float(baseline["metrics"]["ndcg_at_10"]),
        "wins_at_least_10": comparison["wins"] >= 10,
        "losses_at_most_8": comparison["losses"] <= 8,
        "each_subset_declines_at_most_1": min(subset_deltas.values()) >= -1,
        "p95_latency_not_higher": candidate_p95 <= baseline_p95,
        "context_passage_recall_not_lower": float(candidate["metrics"]["context_passage_recall"]) >= float(baseline["metrics"]["context_passage_recall"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "pairwise": comparison,
        "subset_hit_deltas": subset_deltas,
        "p95_latency_ratio": round(latency_ratio, 6),
    }


def _answer_smoke_gate(candidate: dict[str, Any], *, baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_rows = {row["case_id"]: row for row in candidate["cases"]}
    baseline_rows = {row["case_id"]: row for row in baseline["cases"]}
    if candidate_rows.keys() != baseline_rows.keys():
        raise ValueError("Answer smoke reports use different cases.")
    evidence_not_lower = all(
        int(candidate_rows[case_id]["evidence_count"]) >= int(row["evidence_count"])
        for case_id, row in baseline_rows.items()
    )
    checks = {
        "candidate_metadata_prefix_leaks_zero": candidate["metadata_prefix_leak_count"] == 0,
        "baseline_metadata_prefix_leaks_zero": baseline["metadata_prefix_leak_count"] == 0,
        "context_packing_not_lower": evidence_not_lower,
        "candidate_citations_and_pages_present": _answer_previews_have_pages(candidate_rows.values()),
        "baseline_citations_and_pages_present": _answer_previews_have_pages(baseline_rows.values()),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _answer_previews_have_pages(rows: Any) -> bool:
    return all(
        "## Top excerpts" in str(row["answer_preview"])
        and " p." in str(row["answer_preview"])
        for row in rows
    )


def _s1_pipeline():
    pipeline = get_pipeline_config("s1")
    if pipeline.name != "v2_fixed_hybrid" or pipeline.use_rerank:
        raise RuntimeError("S1 must remain the frozen no-rerank v2_fixed_hybrid pipeline.")
    return pipeline


def _pipeline_snapshot(pipeline: Any) -> dict[str, Any]:
    return {"config": asdict(pipeline), "config_sha256": pipeline.config_hash()}


def _dataset_specs(config: dict[str, Any]) -> tuple[tuple[str, Path, str], ...]:
    return (
        ("holdout", Path(config["holdout_dataset"]), config["holdout_dataset_sha256"]),
        ("old_dev", Path(config["retrieval_dataset"]), config["retrieval_dataset_sha256"]),
    )


def _load_freeze(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    path = Path(config["output_dir"]) / "freeze" / "manifest.json"
    if not path.exists():
        raise RuntimeError("Run `python -m evals.m3_2_strategy --freeze` before formal evaluation.")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("config_sha256") != sha256_file(config_path):
        raise ValueError("M3.2 config drifted after the pre-run freeze.")
    frozen_s1 = dict(freeze.get("pipelines", {}).get("s1") or {})
    if frozen_s1.get("config_sha256") != _s1_pipeline().config_hash():
        raise ValueError("S1 configuration drifted after the pre-run freeze.")
    return freeze


def _validate_inputs(config: dict[str, Any]) -> None:
    _, artifact_sha = load_parser_artifact(
        Path(config["parser_artifact"]),
        expected_sha256=config["parser_artifact_sha256"],
        corpus_dir=Path(config["corpus_dir"]),
    )
    if artifact_sha != config["parser_artifact_sha256"]:
        raise ValueError("Parser artifact does not match the frozen SHA-256.")
    for _, path, expected_sha in _dataset_specs(config):
        if sha256_file(path) != expected_sha:
            raise ValueError("M3.2 retrieval dataset does not match the frozen SHA-256.")


def _validate_code_state(current: dict[str, Any], frozen: dict[str, Any]) -> None:
    keys = ("commit", "dirty", "working_tree_patch_sha256")
    if any(current.get(key) != frozen.get(key) for key in keys):
        raise RuntimeError("Code commit or working-tree patch changed after the M3.2 pre-run freeze.")


def _load_config(path: Path, *, repo_root: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 3:
        raise ValueError("Unsupported M3.2 strategy config schema.")
    if payload.get("mode") != "m3_2_strategy":
        raise ValueError("M3.2 config mode must be m3_2_strategy.")
    for key in (
        "output_dir", "corpus_dir", "parser_artifact", "parser_gold",
        "retrieval_dataset", "holdout_dataset", "answer_smoke_dataset",
    ):
        candidate = Path(str(payload[key]))
        payload[key] = str(candidate if candidate.is_absolute() else (repo_root / candidate).resolve())
    latency = dict(payload["latency"])
    if latency != {"warmup_count": 1, "repeat_count": 5, "random_seed": 31}:
        raise ValueError("M3.2 latency protocol must remain frozen.")
    if payload["retrieval_dataset_sha256"] == payload["holdout_dataset_sha256"]:
        raise ValueError("Old dev and new holdout must have distinct SHAs.")
    if tuple(payload.get("pipelines", ())) != ("b1", "s1"):
        raise ValueError("M3.2 must evaluate exactly B1 and S1.")
    artifacts_root = (repo_root / "artifacts").resolve()
    if not Path(payload["output_dir"]).resolve().is_relative_to(artifacts_root):
        raise ValueError("M3.2 output_dir must stay under artifacts.")
    cache_dir = Path(str(payload["reranker"]["cache_dir"]))
    payload["reranker"]["cache_dir"] = str(cache_dir if cache_dir.is_absolute() else (repo_root / cache_dir).resolve())
    return payload


def _selection_reason(report: dict[str, Any]) -> str:
    if report["strategy_candidate_passed"]:
        return "S1 passed every frozen M3.2 gate on holdout and old dev."
    return "S1 failed at least one frozen M3.2 gate; B1 remains the fixed baseline."


def _m4_difficult_cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = report["datasets"]["old_dev"]
    comparison = dataset["gate"]["pairwise"]["cases"]
    s1_rows = {row["case_id"]: row for row in dataset["pipelines"]["s1"]["retrieval"]["cases"]}
    return [
        {
            "case_id": row["case_id"],
            "category": row["category"],
            "baseline_rank": row["baseline_rank"],
            "s1_rank": row["candidate_rank"],
            "question": s1_rows[row["case_id"]]["question"],
            "tags": s1_rows[row["case_id"]]["tags"],
            "observability_signals": {
                "dense_sparse_top_result_disagreement": _channel_disagreement(s1_rows[row["case_id"]]),
                "top_score_gap_small": _top_score_gap_small(s1_rows[row["case_id"]]),
                "table_or_number_localization": row["category"] == "experiment_number_table",
                "abbreviation": "缩写" in s1_rows[row["case_id"]]["tags"],
                "cross_section": "跨章节" in s1_rows[row["case_id"]]["tags"],
                "cross_paper": "跨论文" in s1_rows[row["case_id"]]["tags"],
                "multiple_constraints": row["category"] == "cross_paper_or_section",
                "first_context_incomplete": s1_rows[row["case_id"]]["context_passage_recall"] < 1.0,
            },
            "stage_trace": s1_rows[row["case_id"]]["stage_results"],
        }
        for row in comparison
        if row["outcome"] == "loss"
    ]


def _channel_disagreement(row: dict[str, Any]) -> bool | None:
    stages = row.get("stage_results", {})
    dense = _stage_passage_ids(stages.get("dense", []))
    sparse = _stage_passage_ids(stages.get("sparse", []))
    return dense[0] != sparse[0] if dense and sparse else None


def _top_score_gap_small(row: dict[str, Any]) -> bool | None:
    fusion = list(row.get("stage_results", {}).get("fusion", []))
    scores = [float(item["score"]) for item in fusion[:2] if "score" in item]
    return abs(scores[0] - scores[1]) <= 0.01 if len(scores) == 2 else None


def _stage_passage_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("passage_id") or "") for row in rows if row.get("passage_id")]


def _render_acceptance(core: dict[str, Any]) -> str:
    lines = [
        "# M3.2 策略收口验收", "",
        f"- Strategy candidate passed: `{core['strategy_candidate_passed']}`",
        f"- Default fixed pipeline: `{core['default_pipeline']}`",
        f"- M4 fixed baseline: `{core['m4_fixed_baseline']}`",
        "- M3.1 core passed: `false`（历史失败结论保持不变）",
        f"- M3 strategy closed: `{core['m3_strategy_closed']}`",
        f"- M4 entry ready: `{core['m4_entry_ready']}`",
        f"- Formal holdout runs: `{core['formal_holdout_run_count']}`",
        f"- Metadata prefix leaks: `{core['metadata_prefix_leak_count']}`",
        f"- Active index changed: `{core['active_index_changed']}`", "",
        "## 冻结 gate", "",
    ]
    for role, dataset in core["datasets"].items():
        gate = dataset["gate"]
        lines.extend([
            f"### {role}", "",
            f"- Passed: `{gate['passed']}`",
            f"- W/T/L: `{gate['pairwise']['wins']}/{gate['pairwise']['ties']}/{gate['pairwise']['losses']}`",
            f"- Checks: `{json.dumps(gate['checks'], ensure_ascii=False)}`",
            f"- Answer smoke: `{json.dumps(dataset['answer_smoke_gate']['checks'], ensure_ascii=False)}`", "",
        ])
    lines.extend(["## 决策", "", _selection_reason(core), "", "M4 只可使用冻结 baseline contract；本里程碑不实现 M4。", ""])
    return "\n".join(lines)


def _render_per_question(core: dict[str, Any]) -> str:
    lines = ["# M3.2 逐题结果", ""]
    for role, dataset in core["datasets"].items():
        lines.extend([f"## {role}", "", "| Case | Category | B1 rank | S1 rank | Result |", "| --- | --- | ---: | ---: | --- |"])
        for row in dataset["gate"]["pairwise"]["cases"]:
            lines.append(f"| {row['case_id']} | {row['category']} | {row['baseline_rank'] or '-'} | {row['candidate_rank'] or '-'} | {row['outcome']} |")
        lines.append("")
    lines.extend(["## M4 困难查询输入", "", "以下是 old dev 中 S1 相比 B1 的退化题及其 trace 信号；它们不是运行时 gold 规则。", ""])
    for row in core["m4_difficult_cases"]:
        lines.append(f"- `{row['case_id']}`：`{json.dumps(row['observability_signals'], ensure_ascii=False)}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--refresh-freeze", action="store_true")
    parser.add_argument("--build-report", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        output = create_freeze(
            args.config,
            refresh_preflight_failure=args.refresh_freeze,
        )
    elif args.build_report:
        repo_root = Path(__file__).resolve().parent.parent
        config = _load_config(args.config, repo_root=repo_root)
        output = build_report(Path(config["output_dir"]))
    else:
        output = run_from_config(args.config)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
