"""Frozen M4.1.1 retrieval-quality evaluation over the read-only M3.2 B1 index."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from agent.adaptive import (
    AdaptiveRunResult,
    build_live_loop,
    invoke_structured_json,
    run_fixed_b1,
    validate_m4_baseline,
)
from agent.adaptive_graph import _route
from agent.prompts import get_claim_support_grader_prompt
from agent.schemas import ClaimSupportAssessment
from core.factory import build_retriever
from core.settings import AppSettings, load_settings
from indexing.index_versions import embedding_contract
from llms.llm import configure_llm_router


def run_from_config(config_path: Path) -> dict[str, Any]:
    """Run a frozen M4.1.1 route or answer configuration."""
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Unsupported M4.1.1 eval config.")
    kind = str(config.get("kind", ""))
    if kind not in {"m4_1_1_route", "m4_1_1_answer"}:
        raise ValueError("Config is not an M4.1.1 route or answer evaluation.")

    settings = load_settings(base_dir=root)
    configured_models = config.get("task_models", {})
    if (
        not isinstance(configured_models, dict)
        or ("task_models" in config and not configured_models)
        or not all(
        isinstance(task, str) and isinstance(model, str) and model
        for task, model in configured_models.items()
        )
    ):
        raise ValueError("M4.1.1 task_models must be a non-empty string mapping.")
    if configured_models:
        settings = replace(
            settings,
            llm_task_models={**settings.llm_task_models, **configured_models},
        )
    configure_llm_router(settings.llm_config())
    validate_m4_baseline(settings, base_dir=root)
    dataset_path = root / str(config["dataset"])
    manifest_path = root / str(config["dataset_manifest"])
    manifest = _validate_dataset_hash(dataset_path, manifest_path, kind)
    _validate_manifest_contract(manifest)
    index_manifest_path = root / str(config["evaluation_index_manifest"])
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    _validate_index_contract(settings, index_manifest)
    retriever = build_retriever(_settings_for_eval_index(settings, index_manifest))
    if retriever is None:
        raise ValueError("Frozen M3.2 B1 evaluation index is unavailable.")

    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    if kind == "m4_1_1_route":
        report = _run_route(cases, settings, retriever)
    else:
        report = _run_answer(cases, settings, retriever, config["thresholds"])
    report["frozen_manifest"] = str(config["dataset_manifest"])
    report["baseline_contract"] = str(config["baseline_contract"])
    report["evaluation_index_manifest"] = str(config["evaluation_index_manifest"])
    report["task_models"] = dict(settings.llm_task_models)

    output_dir = root / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{kind}_report.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _run_route(
    cases: list[dict[str, Any]],
    settings: AppSettings,
    retriever: Any,
) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(
            executor.map(
                lambda case: _run_route_case(case, settings, retriever),
                cases,
            )
        )
    return {
        "kind": "m4_1_1_route",
        "dataset_case_count": len(rows),
        "cases": rows,
        "metrics": _route_metrics(rows),
    }


def _run_route_case(
    case: dict[str, Any],
    settings: AppSettings,
    retriever: Any,
) -> dict[str, Any]:
    started = perf_counter()
    initial = _route(str(case["query"]))
    if initial == "fact":
        loop = build_live_loop(settings, retriever)
        loop.answerer = _route_answerer
        result = loop.run(str(case["query"]), scope=list(case.get("scope", [])))
        predicted = result.strategy
        rounds = result.rounds
        calls = result.tool_calls
        reason = result.termination_reason
        evidence_ids = [item["evidence_id"] for item in result.evidence]
    else:
        predicted, rounds, calls = initial, 0, 0
        reason = f"{initial}_no_retrieval"
        evidence_ids = []
    return {
        "id": case["id"],
        "expected": case["expected_route"],
        "predicted": predicted,
        "rounds": rounds,
        "tool_calls": calls,
        "evidence_ids": evidence_ids,
        "termination_reason": reason,
        "latency_ms": round((perf_counter() - started) * 1000, 4),
    }


def _route_answerer(
    _: str,
    __: list[dict[str, Any]],
    ___: list[dict[str, Any]],
) -> dict[str, Any]:
    """Avoid answer generation in the route-only diagnostic."""
    return {"answer": "", "claims": [], "limitations": "route-only evaluation"}


def _run_answer(
    cases: list[dict[str, Any]],
    settings: AppSettings,
    retriever: Any,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(
            executor.map(
                lambda case: _run_answer_case(case, settings, retriever),
                cases,
            )
        )
    metrics = _answer_metrics(rows, thresholds)
    return {
        "kind": "m4_1_1_answer",
        "dataset_case_count": len(rows),
        "cases": rows,
        "metrics": metrics,
    }


def _run_answer_case(
    case: dict[str, Any],
    settings: AppSettings,
    retriever: Any,
) -> dict[str, Any]:
    query = str(case["query"])
    scope = list(case.get("scope", []))
    fixed = run_fixed_b1(settings, retriever, query, scope=scope)
    adaptive = build_live_loop(settings, retriever).run(query, scope=scope)
    return {
        "id": case["id"],
        "fixed": _score_result(fixed, case),
        "adaptive": _score_result(adaptive, case),
    }


def _score_result(
    result: AdaptiveRunResult,
    case: dict[str, Any],
) -> dict[str, Any]:
    claims = [
        claim
        for claim in result.final_answer.get("claims", [])
        if bool(claim.get("major", True))
    ]
    judgments, grading_error = _grade_claims(claims, result.evidence, case["claim_specs"])
    scored = _score_claims(
        claims,
        case["claim_specs"],
        judgments,
        {item["evidence_id"] for item in result.evidence},
    )
    return {
        **scored,
        "answer": result.final_answer.get("answer", ""),
        "claims": claims,
        "limitations": result.final_answer.get("limitations", ""),
        "evidence": result.evidence,
        "evidence_ids": [item["evidence_id"] for item in result.evidence],
        "semantic_judgments": judgments,
        "grading_error": grading_error,
        "rounds": result.rounds,
        "tool_calls": result.tool_calls,
        "latency_ms": result.latency_ms,
        "tokens": result.context_tokens,
        "termination_reason": result.termination_reason,
    }


def _grade_claims(
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    claim_specs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    if not claims:
        return [], ""
    payload = {
        "claim_specs": claim_specs,
        "claims": [
            {"claim_index": index, **claim}
            for index, claim in enumerate(claims)
        ],
        "evidence": [
            {
                key: item[key]
                for key in ("evidence_id", "quote", "page", "section_path")
            }
            for item in evidence
        ],
    }
    try:
        response = invoke_structured_json(
            "adaptive_claim_grader",
            ClaimSupportAssessment,
            [
                SystemMessage(content=get_claim_support_grader_prompt()),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ],
        )
        return [item.model_dump() for item in response.items], ""
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def _score_claims(
    claims: list[dict[str, Any]],
    claim_specs: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    available_evidence_ids: set[str],
) -> dict[str, Any]:
    specs = {str(item["id"]): item for item in claim_specs}
    by_index = {int(item.get("claim_index", -1)): item for item in judgments}
    supported_indices: list[int] = []
    covered_spec_ids: set[str] = set()
    semantic_false_positive = 0
    semantic_false_negative = 0
    gold_evidence_miss_count = 0
    semantic_boolean_repair_count = 0

    for index, claim in enumerate(claims):
        judgment = by_index.get(index, {})
        spec_id = str(judgment.get("claim_spec_id") or "")
        spec = specs.get(spec_id)
        cited_ids = {str(item) for item in claim.get("evidence_ids", []) if str(item)}
        deterministic_match = bool(
            spec
            and cited_ids
            and cited_ids.issubset(available_evidence_ids)
            and cited_ids.issubset(set(spec["acceptable_evidence_ids"]))
        )
        semantic_support = bool(judgment.get("semantically_supported", False))
        if semantic_support and not deterministic_match:
            semantic_false_positive += 1
        if deterministic_match and not semantic_support:
            semantic_false_negative += 1
        if deterministic_match and semantic_support:
            supported_indices.append(index)
            covered_spec_ids.add(spec_id)

    total_claims = len(claims)
    cited_claims = sum(bool(item.get("evidence_ids", [])) for item in claims)
    supported_count = len(supported_indices)
    spec_count = len(claim_specs)
    coverage = len(covered_spec_ids) / spec_count if spec_count else 1.0
    return {
        "requirement_coverage": round(coverage, 4),
        "citation_correctness": round(
            supported_count / cited_claims if cited_claims else 0.0,
            4,
        ),
        "citation_completeness": round(
            len(covered_spec_ids) / spec_count if spec_count else 1.0,
            4,
        ),
        "major_fact_support_rate": round(
            supported_count / total_claims if total_claims else 0.0,
            4,
        ),
        "unsupported_major_claim_count": total_claims - supported_count,
        "supported_claim_indices": supported_indices,
        "covered_claim_spec_ids": sorted(covered_spec_ids),
        "semantic_false_positive_count": semantic_false_positive,
        "semantic_false_negative_count": semantic_false_negative,
        "gold_evidence_miss_count": gold_evidence_miss_count,
        "semantic_boolean_repair_count": semantic_boolean_repair_count,
    }


def _route_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("direct", "fixed", "adaptive", "refuse")
    matrix = {label: {other: 0 for other in labels} for label in labels}
    for row in rows:
        matrix[row["expected"]][row["predicted"]] += 1
    recalls: dict[str, float] = {}
    f1s: list[float] = []
    for label in labels:
        true_positive = matrix[label][label]
        actual = sum(matrix[label].values())
        predicted = sum(matrix[other][label] for other in labels)
        recall = true_positive / actual if actual else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls[label] = round(recall, 4)
        f1s.append(f1)
    return {
        "confusion_matrix": matrix,
        "class_recall": recalls,
        "macro_f1": round(mean(f1s), 4),
        "direct_refuse_safety_passed": recalls["direct"] >= 0.75
        and recalls["refuse"] >= 0.75,
        "latency": _latency_metrics(rows),
    }


def _answer_metrics(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    fixed = [row["fixed"] for row in rows]
    adaptive = [row["adaptive"] for row in rows]
    improvements = [
        row["id"]
        for row in rows
        if row["adaptive"]["requirement_coverage"]
        > row["fixed"]["requirement_coverage"]
    ]
    regressions = [
        row["id"]
        for row in rows
        if row["adaptive"]["requirement_coverage"]
        < row["fixed"]["requirement_coverage"]
    ]
    fixed_metrics = _mean_metrics(fixed)
    adaptive_metrics = _mean_metrics(adaptive)
    termination_rate = round(
        sum(_successful_termination(item["termination_reason"]) for item in adaptive)
        / len(adaptive),
        4,
    ) if adaptive else 0.0
    duplicate_count = sum(
        item["termination_reason"] == "duplicate_query_scope" for item in adaptive
    )
    quality_gate = {
        "coverage": len(improvements) >= int(thresholds["coverage_improvements_min"])
        and len(regressions) <= int(thresholds["coverage_regressions_max"]),
        "citation_correctness": adaptive_metrics["citation_correctness"]
        >= fixed_metrics["citation_correctness"],
        "citation_completeness": adaptive_metrics["citation_completeness"]
        >= fixed_metrics["citation_completeness"],
        "major_fact_support": adaptive_metrics["major_fact_support_rate"]
        >= fixed_metrics["major_fact_support_rate"],
        "unsupported_major_claims": adaptive_metrics["unsupported_major_claim_count"]
        <= fixed_metrics["unsupported_major_claim_count"],
        "termination": termination_rate == 1.0,
        "round_budget": all(item["rounds"] <= int(thresholds["max_rounds"]) for item in adaptive),
        "tool_budget": all(item["tool_calls"] <= int(thresholds["max_tool_calls"]) for item in adaptive),
        "duplicate_query_scope": duplicate_count == 0,
        "latency": "record_only",
    }
    quality_gate["passed"] = all(
        value for key, value in quality_gate.items() if key not in {"latency", "passed"}
    )
    return {
        "coverage_improvement_case_ids": improvements,
        "coverage_regression_case_ids": regressions,
        "coverage_improvements": len(improvements),
        "coverage_regressions": len(regressions),
        "fixed": fixed_metrics,
        "adaptive": adaptive_metrics,
        "successful_termination_rate": termination_rate,
        "duplicate_query_scope_count": duplicate_count,
        "coverage_not_improved_stop_count": sum(
            item["termination_reason"] == "coverage_not_improved" for item in adaptive
        ),
        "quality_gate": quality_gate,
    }


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "requirement_coverage",
        "citation_correctness",
        "citation_completeness",
        "major_fact_support_rate",
        "unsupported_major_claim_count",
        "rounds",
        "tool_calls",
        "latency_ms",
        "tokens",
        "semantic_false_positive_count",
        "semantic_false_negative_count",
    )
    result = {
        key: round(mean(float(row[key]) for row in rows), 4) if rows else 0.0
        for key in keys
    }
    result["p95_latency_ms"] = round(
        _percentile([float(row["latency_ms"]) for row in rows], 0.95),
        4,
    )
    return result


def _latency_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = [float(row["latency_ms"]) for row in rows]
    return {
        "mean_ms": round(mean(values), 4) if values else 0.0,
        "p95_ms": round(_percentile(values, 0.95), 4),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _successful_termination(reason: str) -> bool:
    return reason not in {
        "",
        "cancelled",
        "empty_plan",
        "model_error",
        "retrieval_error",
    }


def _validate_dataset_hash(
    dataset_path: Path,
    manifest_path: Path,
    kind: str,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = "route_dataset" if kind == "m4_1_1_route" else "answer_dataset"
    actual = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if actual != manifest[key]["sha256"]:
        raise ValueError("M4.1.1 frozen dataset hash does not match its manifest.")
    return manifest


def _validate_manifest_contract(manifest: dict[str, Any]) -> None:
    baseline = manifest.get("baseline_contract", {})
    if baseline.get("selected_pipeline_name") != "v1_flat_rerank":
        raise ValueError("M4.1.1 manifest does not select frozen B1.")
    if (
        baseline.get("pipeline_config_hash")
        != "ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17"
    ):
        raise ValueError("M4.1.1 manifest B1 hash differs from the frozen contract.")


def _validate_index_contract(settings: AppSettings, manifest: dict[str, Any]) -> None:
    if (
        manifest.get("pipeline_config_hash")
        != "ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17"
    ):
        raise ValueError("Evaluation index does not use the frozen B1 pipeline.")
    if manifest.get("embedding") != embedding_contract(settings):
        raise ValueError("Runtime embedding contract differs from frozen B1 index.")


def _settings_for_eval_index(
    settings: AppSettings,
    manifest: dict[str, Any],
) -> AppSettings:
    index_dir = Path(str(manifest["content_index_dir"]))
    return replace(
        settings,
        index_write_mode="legacy",
        index_dir=index_dir,
        faiss_dir=index_dir / "faiss",
        bm25_path=index_dir / "bm25.pkl",
        nodes_path=index_dir / "nodes.jsonl",
        doc_trees_path=index_dir / "doc_trees.json",
    )
