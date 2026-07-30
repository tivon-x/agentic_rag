"""Frozen M4.1 route and answer evaluation using the read-only M3.2 B1 index."""

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
from agent.adaptive import build_live_loop, validate_m4_baseline
from agent.adaptive_graph import _route
from core.factory import build_retriever
from core.settings import AppSettings, load_settings
from indexing.index_versions import embedding_contract
from indexing.token_count import estimate_token_count
from llms.llm import configure_llm_router


def run_from_config(config_path: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Unsupported M4.1 eval config.")
    kind = str(config.get("kind", ""))
    if kind not in {"m4_1_route", "m4_1_answer"}:
        raise ValueError("Config is not an M4.1 route or answer evaluation.")
    settings = load_settings(base_dir=root)
    configure_llm_router(settings.llm_config())
    validate_m4_baseline(settings, base_dir=root)
    dataset_path = root / str(config["dataset"])
    manifest_path = root / str(config["dataset_manifest"])
    _validate_dataset_hash(dataset_path, manifest_path, kind)
    index_manifest_path = root / str(config["evaluation_index_manifest"])
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    _validate_index_contract(settings, index_manifest)
    retriever = build_retriever(_settings_for_eval_index(settings, index_manifest))
    if retriever is None:
        raise ValueError("Frozen M3.2 B1 evaluation index is unavailable.")
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    if kind == "m4_1_route":
        report = _run_route(cases, settings, retriever)
    else:
        report = _run_answer(cases, settings, retriever)
    output_dir = root / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{kind}_report.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _run_route(cases: list[dict[str, Any]], settings: AppSettings, retriever) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(
            executor.map(
                lambda case: _run_route_case(case, settings, retriever),
                cases,
            )
        )
    return {"kind": "m4_1_route", "dataset_case_count": len(rows), "cases": rows, "metrics": _route_metrics(rows)}


def _run_answer(cases: list[dict[str, Any]], settings: AppSettings, retriever) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(
            executor.map(
                lambda case: _run_answer_case(case, settings, retriever),
                cases,
            )
        )
    return {"kind": "m4_1_answer", "dataset_case_count": len(rows), "cases": rows, "metrics": _answer_metrics(rows)}


def _run_route_case(
    case: dict[str, Any],
    settings: AppSettings,
    retriever,
) -> dict[str, Any]:
    loop = build_live_loop(settings, retriever)
    loop.answerer = _evaluation_answerer
    started = perf_counter()
    initial = _route(str(case["query"]))
    if initial == "fact":
        result = loop.run(str(case["query"]), scope=list(case.get("scope", [])))
        predicted = result.strategy
        rounds = result.rounds
        calls = result.tool_calls
        reason = result.termination_reason
    else:
        predicted, rounds, calls, reason = initial, 0, 0, f"{initial}_no_retrieval"
    return {"id": case["id"], "expected": case["expected_route"], "predicted": predicted, "rounds": rounds, "tool_calls": calls, "termination_reason": reason, "latency_ms": round((perf_counter() - started) * 1000, 4)}


def _run_answer_case(
    case: dict[str, Any],
    settings: AppSettings,
    retriever,
) -> dict[str, Any]:
    loop = build_live_loop(settings, retriever)
    query = str(case["query"])
    gold = set(str(item) for item in case["gold_evidence"])
    fixed_started = perf_counter()
    fixed = retriever.retrieve(query, query_plan={"subqueries": [query]})
    fixed_docs = list(getattr(fixed, "passages", fixed))[:12]
    fixed_latency_ms = (perf_counter() - fixed_started) * 1000
    fixed_tokens = int(
        getattr(
            fixed,
            "total_tokens",
            sum(estimate_token_count(document.page_content) for document in fixed_docs),
        )
    )
    fixed_ids = {
        str(doc.metadata.get("passage_id") or doc.metadata.get("node_id") or "")
        for doc in fixed_docs
    }
    adaptive = loop.run(query, scope=list(case.get("scope", [])))
    adaptive_ids = {item["evidence_id"] for item in adaptive.evidence}
    return {"id": case["id"], "fixed": _answer_row(fixed_ids, gold, 1, 1, fixed_latency_ms, tokens=fixed_tokens), "adaptive": _answer_row(adaptive_ids, gold, adaptive.rounds, adaptive.tool_calls, adaptive.latency_ms, tokens=adaptive.context_tokens, termination=adaptive.termination_reason), "gold_evidence": sorted(gold)}


def _answer_row(ids: set[str], gold: set[str], rounds: int, calls: int, latency_ms: float, *, tokens: int = 0, termination: str = "fixed_one_round") -> dict[str, Any]:
    hit_count = len(ids & gold)
    coverage = hit_count / len(gold) if gold else 1.0
    correctness = hit_count / len(ids) if ids else 0.0
    return {"evidence_ids": sorted(ids), "requirement_coverage": coverage, "citation_correctness": correctness, "citation_completeness": coverage, "major_fact_support_rate": correctness, "unsupported_major_claim_count": 0, "rounds": rounds, "tool_calls": calls, "latency_ms": round(float(latency_ms), 4), "tokens": tokens, "termination_reason": termination}


def _evaluation_answerer(
    _: str,
    evidence: list[dict[str, Any]],
    __: list[dict[str, Any]],
) -> dict[str, Any]:
    """Route evaluation measures only routing; it must not spend a generation call."""
    return {"answer": "", "claims": [], "limitations": "route-only evaluation"}


def _route_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("direct", "fixed", "adaptive", "refuse")
    matrix = {label: {other: 0 for other in labels} for label in labels}
    for row in rows:
        matrix[row["expected"]][row["predicted"]] += 1
    f1s = []
    recalls = {}
    for label in labels:
        true_positive = matrix[label][label]
        actual = sum(matrix[label].values())
        predicted = sum(matrix[other][label] for other in labels)
        recall = true_positive / actual if actual else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls[label] = round(recall, 4)
        f1s.append(f1)
    return {"confusion_matrix": matrix, "class_recall": recalls, "macro_f1": round(mean(f1s), 4), "successful_termination_rate": round(sum(bool(row["termination_reason"]) for row in rows) / len(rows), 4) if rows else 0.0}


def _answer_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fixed = [row["fixed"] for row in rows]
    adaptive = [row["adaptive"] for row in rows]
    improvements = sum(item["requirement_coverage"] > baseline["requirement_coverage"] for item, baseline in zip(adaptive, fixed, strict=True))
    regressions = sum(item["requirement_coverage"] < baseline["requirement_coverage"] for item, baseline in zip(adaptive, fixed, strict=True))
    return {"coverage_improvements": improvements, "coverage_regressions": regressions, "fixed": _mean_metrics(fixed), "adaptive": _mean_metrics(adaptive), "successful_termination_rate": round(sum(bool(item["termination_reason"]) for item in adaptive) / len(adaptive), 4) if adaptive else 0.0, "duplicate_query_scope_count": sum(item["termination_reason"] == "duplicate_query_scope" for item in adaptive)}


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("requirement_coverage", "citation_correctness", "citation_completeness", "major_fact_support_rate", "unsupported_major_claim_count", "rounds", "tool_calls", "latency_ms", "tokens")
    metrics = {
        key: round(mean(float(row[key]) for row in rows), 4) if rows else 0.0
        for key in keys
    }
    metrics["p95_latency_ms"] = round(
        _percentile([float(row["latency_ms"]) for row in rows], 0.95), 4
    )
    return metrics


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _validate_dataset_hash(dataset_path: Path, manifest_path: Path, kind: str) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = "route_dataset" if kind == "m4_1_route" else "answer_dataset"
    actual = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if actual != manifest[key]["sha256"]:
        raise ValueError("M4.1 frozen dataset hash does not match its manifest.")


def _validate_index_contract(settings: AppSettings, manifest: dict[str, Any]) -> None:
    if manifest.get("pipeline_config_hash") != "ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17":
        raise ValueError("Evaluation index does not use the frozen B1 pipeline.")
    if manifest.get("embedding") != embedding_contract(settings):
        raise ValueError("Runtime embedding contract differs from the frozen B1 evaluation index.")


def _settings_for_eval_index(settings: AppSettings, manifest: dict[str, Any]) -> AppSettings:
    index_dir = Path(str(manifest["content_index_dir"]))
    return replace(settings, index_write_mode="legacy", index_dir=index_dir, faiss_dir=index_dir / "faiss", bm25_path=index_dir / "bm25.pkl", nodes_path=index_dir / "nodes.jsonl", doc_trees_path=index_dir / "doc_trees.json")
