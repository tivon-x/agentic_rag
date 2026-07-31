"""Frozen M4.1.2 route and answer evaluation with separated score layers."""

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

from agent.adaptive import AdaptiveRunResult, build_live_loop, run_fixed_b1, validate_m4_baseline
from agent.adaptive_graph import _route
from core.factory import build_retriever
from core.settings import AppSettings, load_settings
from evals.m4_1_1_runner import (
    _grade_claims,
    _percentile,
    _route_answerer,
    _route_metrics,
    _settings_for_eval_index,
    _successful_termination,
    _validate_index_contract,
)
from llms.llm import configure_llm_router


def run_from_config(config_path: Path) -> dict[str, Any]:
    """Run one frozen M4.1.2 configuration without mutating the active index."""
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Unsupported M4.1.2 eval config.")
    kind = str(config.get("kind", ""))
    if kind not in {"m4_1_2_route", "m4_1_2_answer"}:
        raise ValueError("Config is not an M4.1.2 evaluation.")
    settings = load_settings(base_dir=root)
    task_models = config.get("task_models")
    if not isinstance(task_models, dict) or not task_models:
        raise ValueError("M4.1.2 task_models must be a non-empty mapping.")
    settings = replace(settings, llm_task_models={**settings.llm_task_models, **task_models})
    configure_llm_router(settings.llm_config())
    validate_m4_baseline(settings, base_dir=root)
    dataset_path = root / str(config["dataset"])
    _validate_frozen_inputs(dataset_path, root / str(config["dataset_manifest"]), kind)
    index_manifest = json.loads((root / str(config["evaluation_index_manifest"])).read_text(encoding="utf-8"))
    _validate_index_contract(settings, index_manifest)
    retriever = build_retriever(_settings_for_eval_index(settings, index_manifest))
    if retriever is None:
        raise ValueError("Frozen M3.2 B1 evaluation index is unavailable.")
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    report = _run_route(cases, settings, retriever) if kind.endswith("route") else _run_answer(cases, settings, retriever, config["thresholds"])
    report.update({"frozen_manifest": str(config["dataset_manifest"]), "baseline_contract": str(config["baseline_contract"]), "evaluation_index_manifest": str(config["evaluation_index_manifest"]), "task_models": dict(settings.llm_task_models)})
    output_dir = root / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{kind}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _validate_frozen_inputs(dataset_path: Path, manifest_path: Path, kind: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = "route_dataset" if kind.endswith("route") else "answer_dataset"
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != manifest[key]["sha256"]:
        raise ValueError("M4.1.2 frozen dataset hash does not match its manifest.")
    baseline = manifest.get("baseline_contract", {})
    if baseline.get("selected_pipeline_name") != "v1_flat_rerank" or baseline.get("pipeline_config_hash") != "ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17":
        raise ValueError("M4.1.2 manifest does not select the frozen B1 contract.")
    return manifest


def _run_route(cases: list[dict[str, Any]], settings: AppSettings, retriever: Any) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(lambda case: _route_case(case, settings, retriever), cases))
    metrics = _route_metrics(rows)
    metrics["quality_gate"] = {"class_recall": all(value >= 0.75 for value in metrics["class_recall"].values()), "macro_f1": metrics["macro_f1"] >= 0.80}
    metrics["quality_gate"]["passed"] = all(metrics["quality_gate"].values())
    return {"kind": "m4_1_2_route", "dataset_case_count": len(rows), "cases": rows, "metrics": metrics}


def _route_case(case: dict[str, Any], settings: AppSettings, retriever: Any) -> dict[str, Any]:
    started = perf_counter()
    initial = _route(str(case["query"]))
    if initial == "fact":
        loop = build_live_loop(settings, retriever)
        loop.answerer = _route_answerer
        result = loop.run(str(case["query"]), scope=list(case.get("scope", [])))
        predicted, rounds, calls, reason = result.strategy, result.rounds, result.tool_calls, result.termination_reason
        evidence_ids = [item["evidence_id"] for item in result.evidence]
    else:
        predicted, rounds, calls, reason, evidence_ids = initial, 0, 0, f"{initial}_no_retrieval", []
    return {"id": case["id"], "expected": case["expected_route"], "predicted": predicted, "rounds": rounds, "tool_calls": calls, "evidence_ids": evidence_ids, "termination_reason": reason, "latency_ms": round((perf_counter() - started) * 1000, 4)}


def _run_answer(cases: list[dict[str, Any]], settings: AppSettings, retriever: Any, thresholds: dict[str, Any]) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(lambda case: _answer_case(case, settings, retriever), cases))
    return {"kind": "m4_1_2_answer", "dataset_case_count": len(rows), "cases": rows, "metrics": _answer_metrics(rows, thresholds), "blind_review_list": _blind_review_list(rows)}


def _answer_case(case: dict[str, Any], settings: AppSettings, retriever: Any) -> dict[str, Any]:
    query, scope = str(case["query"]), list(case.get("scope", []))
    fixed = run_fixed_b1(settings, retriever, query, scope=scope)
    adaptive = build_live_loop(settings, retriever).run(query, scope=scope)
    return {"id": case["id"], "case_type": case["case_type"], "fixed": _score_result(fixed, case), "adaptive": _score_result(adaptive, case)}


def _score_result(result: AdaptiveRunResult, case: dict[str, Any]) -> dict[str, Any]:
    claims = [claim for claim in result.final_answer.get("claims", []) if bool(claim.get("major", True))]
    judgments, grading_error = _grade_claims(claims, result.evidence, case["claim_specs"])
    scored = score_claims_m4_1_2(claims, case["claim_specs"], judgments, result.evidence)
    return {**scored, "strategy": result.strategy, "answer": result.final_answer.get("answer", ""), "claims": claims, "limitations": result.final_answer.get("limitations", ""), "requirements": result.plan_items, "coverage": result.coverage, "evidence": result.evidence, "evidence_ids": [item["evidence_id"] for item in result.evidence], "semantic_judgments": judgments, "grading_error": grading_error, "rounds": result.rounds, "tool_calls": result.tool_calls, "latency_ms": result.latency_ms, "tokens": result.context_tokens, "termination_reason": result.termination_reason}


def score_claims_m4_1_2(claims: list[dict[str, Any]], specs: list[dict[str, Any]], judgments: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep deterministic validity, semantic support, and gold audit separate."""
    spec_by_id = {str(item["id"]): item for item in specs}
    judgment_by_index = {int(item.get("claim_index", -1)): item for item in judgments}
    evidence_by_id = {str(item.get("evidence_id", "")): item for item in evidence}
    supported, covered, deterministic_errors, inconsistent, false_positive, false_negative, gold_misses = [], set(), [], [], 0, 0, 0
    for index, claim in enumerate(claims):
        judgment = judgment_by_index.get(index, {})
        cited = [str(value) for value in claim.get("evidence_ids", []) if str(value)]
        valid = bool(cited) and all(_valid_evidence(evidence_by_id.get(value)) for value in cited)
        if not valid:
            deterministic_errors.append(index)
        semantic = bool(judgment.get("semantically_supported", False))
        reason = str(judgment.get("reason", "")).casefold()
        contradictory = _grader_inconsistent(semantic, reason)
        if contradictory:
            inconsistent.append(index)
        spec = spec_by_id.get(str(judgment.get("claim_spec_id") or ""))
        is_supported = valid and semantic and spec is not None
        if semantic and not valid:
            false_positive += 1
        if valid and spec is not None and not semantic:
            false_negative += 1
        if is_supported:
            supported.append(index)
            covered.add(str(spec["id"]))
            if not set(cited).intersection(set(spec.get("acceptable_evidence_ids", []))):
                gold_misses += 1
    total, cited_claims, spec_count = len(claims), sum(bool(item.get("evidence_ids")) for item in claims), len(specs)
    return {"requirement_coverage": round(len(covered) / spec_count if spec_count else 1.0, 4), "citation_correctness": round(len(supported) / cited_claims if cited_claims else 0.0, 4), "citation_completeness": round(len(covered) / spec_count if spec_count else 1.0, 4), "major_fact_support_rate": round(len(supported) / total if total else 0.0, 4), "unsupported_major_claim_count": total - len(supported), "supported_claim_indices": supported, "covered_claim_spec_ids": sorted(covered), "deterministic_invalid_claim_indices": deterministic_errors, "grader_inconsistent_indices": inconsistent, "semantic_false_positive_count": false_positive, "semantic_false_negative_count": false_negative, "gold_evidence_miss_count": gold_misses}


def _valid_evidence(item: dict[str, Any] | None) -> bool:
    return bool(item and item.get("evidence_id") and item.get("quote") and item.get("paper_id") and item.get("section_path") and isinstance(item.get("page"), int) and item["page"] > 0)


def _grader_inconsistent(value: bool, reason: str) -> bool:
    negative = any(token in reason for token in ("not support", "does not support", "unsupported", "不支持", "不匹配"))
    positive = any(token in reason for token in ("directly supports", "supports the claim", "matches the", "直接支持", "匹配该"))
    return (value and negative) or (not value and positive and not negative)


def _answer_metrics(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    fixed_all, adaptive_all = [row["fixed"] for row in rows], [row["adaptive"] for row in rows]
    adaptive_rows = [row for row in rows if row["case_type"] == "adaptive_eligible"]
    fixed_rows = [row for row in rows if row["case_type"] == "fixed_eligible"]
    improvements = [row["id"] for row in adaptive_rows if row["adaptive"]["requirement_coverage"] > row["fixed"]["requirement_coverage"]]
    regressions = [row["id"] for row in adaptive_rows if row["adaptive"]["requirement_coverage"] < row["fixed"]["requirement_coverage"]]
    fixed_false_trigger = [row["id"] for row in fixed_rows if row["adaptive"]["strategy"] == "adaptive"]
    fixed_metrics, adaptive_metrics = _mean_metrics(fixed_all), _mean_metrics(adaptive_all)
    termination = sum(_successful_termination(row["adaptive"]["termination_reason"]) for row in rows) / len(rows) if rows else 0.0
    duplicate = sum(row["adaptive"]["termination_reason"] == "duplicate_query_scope" for row in rows)
    gate = {"coverage": len(improvements) >= int(thresholds["coverage_improvements_min"]) and len(regressions) <= int(thresholds["coverage_regressions_max"]), "citation_correctness": adaptive_metrics["citation_correctness"] >= fixed_metrics["citation_correctness"], "citation_completeness": adaptive_metrics["citation_completeness"] >= fixed_metrics["citation_completeness"], "major_fact_support": adaptive_metrics["major_fact_support_rate"] >= fixed_metrics["major_fact_support_rate"], "unsupported_major_claims": adaptive_metrics["unsupported_major_claim_count"] <= fixed_metrics["unsupported_major_claim_count"], "termination": round(termination, 4) == float(thresholds["successful_termination_rate"]), "average_rounds": adaptive_metrics["rounds"] <= float(thresholds["average_rounds_max"]), "round_budget": all(row["adaptive"]["rounds"] <= int(thresholds["max_rounds"]) for row in rows), "tool_budget": all(row["adaptive"]["tool_calls"] <= int(thresholds["max_tool_calls"]) for row in rows), "duplicate_query_scope": duplicate == int(thresholds["duplicate_query_scope_count"])}
    gate["passed"] = all(gate.values())
    return {"adaptive_eligible": {"coverage_improvement_case_ids": improvements, "coverage_regression_case_ids": regressions, "coverage_improvements": len(improvements), "coverage_regressions": len(regressions)}, "fixed_eligible": {"false_trigger_case_ids": fixed_false_trigger, "false_trigger_rate": round(len(fixed_false_trigger) / len(fixed_rows), 4) if fixed_rows else 0.0}, "fixed": fixed_metrics, "adaptive": adaptive_metrics, "successful_termination_rate": round(termination, 4), "duplicate_query_scope_count": duplicate, "quality_gate": gate}


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("requirement_coverage", "citation_correctness", "citation_completeness", "major_fact_support_rate", "unsupported_major_claim_count", "rounds", "tool_calls", "latency_ms", "tokens", "semantic_false_positive_count", "semantic_false_negative_count")
    result = {key: round(mean(float(row[key]) for row in rows), 4) if rows else 0.0 for key in keys}
    result["p95_latency_ms"] = round(_percentile([float(row["latency_ms"]) for row in rows], 0.95), 4)
    return result


def _blind_review_list(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())[:5]
    return [{"case_id": row["id"], "items": [{"claim": claim["claim"], "evidence_ids": claim.get("evidence_ids", [])} for claim in row["adaptive"]["claims"]], "automatic_flags": {"false_positive": row["adaptive"]["semantic_false_positive_count"], "false_negative": row["adaptive"]["semantic_false_negative_count"], "grader_inconsistent": len(row["adaptive"]["grader_inconsistent_indices"])}} for row in selected]
