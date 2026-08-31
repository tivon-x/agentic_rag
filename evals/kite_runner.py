"""Prepare, run and report the frozen KITE end-to-end evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import re
from statistics import mean
import subprocess
from time import perf_counter
from typing import Any

import yaml
from langchain_core.messages import HumanMessage

from agent.graph import create_agent_graph
from agent.tools import ToolFactory
from core.settings import AppSettings, load_settings
from evals.kite import (
    KITE_CASE_COUNT,
    KITE_COMMIT,
    KITE_CORPUS_FILE_COUNT,
    KITE_CORPUS_SHA256,
    KITE_EMPTY_RUBRIC_COUNT,
    KITE_PARSER_NAME,
    KITE_PARSER_VERSION,
    KITE_QUERY_SHA256,
    KITE_REPOSITORY,
    KiteCase,
    KiteDataError,
    build_kite_manifest,
    build_kite_parser_artifact,
    load_kite_cases,
    validate_manifest_payload,
)
from evals.v2_corpus import artifact_documents, sha256_file
from evals.v2_runner import (
    _build_eval_index,
    _capture_code_state,
    _embedding_contract,
    _load_eval_retriever,
    _prepare_variant_settings,
)
from indexing.retrieval_pipeline import get_pipeline_config
from indexing.parsers.paper_parser import NORMALIZATION_VERSION
from llms.llm import configure_llm_router, get_llm_by_type


LOGGER = logging.getLogger(__name__)
PIPELINES = ("b0", "b1", "b2", "b3")
JUDGE_PROMPT_VERSION = "kite-official-compatible-v1"
JUDGE_SCORE_RE = re.compile(r"^\s*(10|[0-9])\s*$")
PUBLIC_EVIDENCE_KEYS = (
    "doc_id",
    "node_id",
    "paper_id",
    "paper_title",
    "source",
    "section_path",
    "page",
    "quote",
    "score",
    "relevance",
)
JUDGE_PROMPT = """
Your job is to evaluate the performance of an AI-powered question answering system. You will be given a query, a ground truth answer, and the answer given by the AI. Your task is to grade the AI's answer on a scale of 0-10. A score of 0 means the AI's answer is wrong. A score of 10 means the AI's answer is completely correct.

Your response must ONLY be an integer between 0 and 10 (inclusive). Do not include any other text in your response.

GUIDELINES FOR GRADING
- The ground truth answers are often lacking in detail, so if the AI's answer is more detailed than the ground truth answer, then that's generally a good sign.
- Be wary of overly broad or general AI answers. If the AI's answer lacks specifics, then it probably isn't a good answer.
- If a grading rubric is included in the GRADING RUBRIC section, then pay close attention to it. The rubric will tell you specific things to look for in the AI's answer.
- Maintain high standards when grading. A score of 10 should be reserved for answers that are nearly perfect. Answers that miss key details or don't fully answer the question should be heavily penalized.

QUERY
{query}

GROUND TRUTH ANSWER
{ground_truth_answer}

GRADING RUBRIC
{rubric}

AI-GENERATED ANSWER
{model_answer}

GRADE
""".strip()


def prepare_from_config(config_path: Path) -> dict[str, Any]:
    """Validate the frozen checkout and write its path-safe manifest."""
    repo_root = Path(__file__).resolve().parent.parent
    config = _load_config(config_path)
    benchmark = config["benchmark"]
    manifest = build_kite_manifest(
        query_path=_resolve_path(benchmark["query_path"], repo_root),
        corpus_root=_resolve_path(benchmark["corpus_root"], repo_root),
        upstream_commit=str(benchmark.get("upstream_commit", KITE_COMMIT)),
        expected_query_sha256=str(
            benchmark.get("query_sha256", KITE_QUERY_SHA256)
        ),
        expected_case_count=int(benchmark.get("case_count", KITE_CASE_COUNT)),
        expected_empty_rubric_count=int(
            benchmark.get("empty_rubric_count", KITE_EMPTY_RUBRIC_COUNT)
        ),
        expected_corpus_file_count=int(
            benchmark.get("corpus_file_count", KITE_CORPUS_FILE_COUNT)
        ),
        expected_corpus_sha256=str(
            benchmark.get("corpus_sha256", KITE_CORPUS_SHA256)
        ),
    )
    manifest_path = _resolve_output_path(
        config.get("output", {}).get(
            "manifest", "artifacts/evals/kite/manifest.json"
        ),
        repo_root,
    )
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest_payload(existing)
        immutable = tuple(key for key in manifest if key != "created_at")
        if any(existing.get(key) != manifest[key] for key in immutable):
            raise KiteDataError("Existing KITE manifest differs from the frozen checkout.")
        manifest = existing
    else:
        _write_json_atomic(manifest_path, manifest)
    result = {
        "manifest": _logical_path(manifest_path, repo_root),
        "manifest_sha256": sha256_file(manifest_path),
        "benchmark_name": manifest["benchmark_name"],
        "upstream_commit": manifest["upstream_commit"],
        "case_count": manifest["case_count"],
        "corpus_file_count": manifest["corpus_file_count"],
        "corpus_file_sha256": manifest["corpus_file_sha256"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return manifest


def run_from_config(config_path: Path, *, allow_dirty: bool = False) -> dict[str, Any]:
    """Run one fixed pipeline against KITE, including generation and judging."""
    repo_root = Path(__file__).resolve().parent.parent
    config = _load_config(config_path)
    output_dir = _resolve_output_path(
        config.get("output", {}).get("dir", "artifacts/evals/kite/b1"),
        repo_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise KiteDataError(f"KITE report already exists: {report_path}")
    code_state = _capture_code_state(repo_root, run_dir=output_dir)
    if code_state["dirty"] and not allow_dirty:
        raise KiteDataError(
            "KITE formal runs require a clean working tree; use --allow-dirty only for diagnostics."
        )
    code_state.pop("working_tree_patch_path", None)
    code_state["config_path"] = _logical_path(config_path, repo_root)
    settings, generation, judge = _runtime_settings(config, repo_root)
    benchmark = config["benchmark"]
    query_path = _resolve_path(benchmark["query_path"], repo_root)
    corpus_root = _resolve_path(benchmark["corpus_root"], repo_root)
    manifest_path = _resolve_output_path(
        config.get("output", {}).get(
            "manifest", "artifacts/evals/kite/manifest.json"
        ),
        repo_root,
    )
    manifest = _load_and_validate_manifest(
        manifest_path,
        query_path=query_path,
        corpus_root=corpus_root,
        benchmark=benchmark,
    )
    manifest_sha = sha256_file(manifest_path)

    root = (repo_root / "artifacts" / "evals" / "kite").resolve()
    artifact_path = _resolve_output_path(
        config.get("output", {}).get(
            "parser_artifact", "artifacts/evals/kite/parser_artifact.json"
        ),
        repo_root,
    )
    artifact, artifact_sha = _load_or_build_parser_artifact(
        settings,
        artifact_path=artifact_path,
        corpus_root=corpus_root,
        expected_manifest=manifest["corpus_manifest"],
    )
    cases = load_kite_cases(
        query_path,
        expected_sha256=manifest["query_sha256"],
        expected_case_count=manifest["case_count"],
        expected_empty_rubric_count=manifest["empty_rubric_count"],
    )
    cases = _select_cases(cases, config.get("runtime", {}).get("case_ids"))
    documents = artifact_documents(artifact)
    evidence_lookup = {
        str(document.metadata.get("node_id") or document.metadata.get("passage_id")): dict(
            document.metadata
        )
        for document in documents
    }
    pipeline_key = str(config["pipeline"]["name"]).strip().lower()
    if pipeline_key not in PIPELINES:
        raise KiteDataError(f"KITE pipeline must be one of {PIPELINES}.")
    pipeline = get_pipeline_config(pipeline_key)
    reranker = _mapping(config.get("reranker"), "reranker")
    if reranker.get("backend", "flashrank") != "flashrank":
        raise KiteDataError("KITE evaluation requires the flashrank reranker.")
    reranker = {
        "backend": "flashrank",
        "model": str(
            reranker.get("model", settings.flashrank_model)
        ),
    }
    variant_settings, index_dir = _prepare_variant_settings(
        settings,
        run_dir=root,
        pipeline_key=pipeline_key,
    )
    reuse_source = _find_reusable_index(
        root / "indexes",
        pipeline=pipeline,
        settings=variant_settings,
        parser_artifact_sha256=artifact_sha,
        exclude=pipeline_key,
    )
    index_manifest = _build_eval_index(
        variant_settings,
        documents=documents,
        pipeline_key=pipeline_key,
        pipeline=pipeline,
        index_dir=index_dir,
        parser_artifact_sha256=artifact_sha,
        corpus_manifest=manifest["corpus_manifest"],
        dataset_sha256=manifest["query_sha256"],
        code_state=code_state,
        reranker=reranker,
        force_reindex=bool(config.get("runtime", {}).get("force_reindex", False)),
        reuse_source=reuse_source,
    )
    retriever = _load_eval_retriever(
        variant_settings,
        pipeline=pipeline,
        pipeline_key=pipeline_key,
        index_dir=index_dir,
        manifest=index_manifest,
        parser_artifact_sha256=artifact_sha,
        reranker=reranker,
    )
    configure_llm_router(variant_settings.llm_config())
    tool_factory = ToolFactory(retriever)
    graph = create_agent_graph(
        tool_factory.create_tools(),
        tool_factory=tool_factory,
        max_context_tokens=variant_settings.max_context_tokens,
        keep_messages=variant_settings.keep_messages,
        max_iterations=variant_settings.max_iterations,
        max_tool_calls=variant_settings.max_tool_calls,
    )
    rows: list[dict[str, Any]] = []
    runtime = _mapping(config.get("runtime"), "runtime")
    started_at = datetime.now(UTC).isoformat()
    for case in cases:
        rows.append(
            _run_case(
                graph,
                case,
                judge=judge,
                thread_prefix=f"kite-{pipeline_key}",
                pipeline_name=pipeline_key,
                pipeline_config_hash=pipeline.config_hash(),
                evidence_lookup=evidence_lookup,
            )
        )
    completed_at = datetime.now(UTC).isoformat()
    report = _build_report(
        config=config,
        config_path=config_path,
        repo_root=repo_root,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha=manifest_sha,
        artifact_path=artifact_path,
        artifact_sha=artifact_sha,
        index_dir=index_dir,
        index_manifest=index_manifest,
        pipeline=pipeline,
        settings=variant_settings,
        generation=generation,
        judge=judge,
        runtime=runtime,
        rows=rows,
        code_state=code_state,
        started_at=started_at,
        completed_at=completed_at,
    )
    _write_json_atomic(report_path, report)
    print(
        json.dumps(
            {
                "pipeline": pipeline_key,
                "report": _logical_path(report_path, repo_root),
                "case_count": report["metrics"]["case_count"],
                "valid_count": report["metrics"]["valid_count"],
                "mean_score": report["metrics"]["mean_score"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return report


def report_from_runs(runs_dir: Path) -> dict[str, Any]:
    """Compare frozen pipeline reports and write the conservative decision."""
    repo_root = Path(__file__).resolve().parent.parent
    reports = {
        key: json.loads((runs_dir / key / "report.json").read_text(encoding="utf-8"))
        for key in PIPELINES
    }
    _validate_reports(reports, repo_root, runs_dir)
    baseline = reports["b1"]
    pairwise: dict[str, Any] = {}
    for key in PIPELINES:
        if key == "b1":
            continue
        pairwise[key] = _pairwise_scores(baseline, reports[key])
    candidates: list[str] = []
    baseline_score = baseline["metrics"].get("mean_score")
    baseline_p95 = baseline["metrics"].get("p95_latency_ms")
    if isinstance(baseline_score, (int, float)) and isinstance(
        baseline_p95, (int, float)
    ) and baseline["metrics"].get("valid_count") == baseline["metrics"].get(
        "case_count"
    ):
        for key in (candidate for candidate in PIPELINES if candidate != "b1"):
            metrics = reports[key]["metrics"]
            if (
                metrics.get("valid_count") == metrics.get("case_count")
                and isinstance(metrics.get("mean_score"), (int, float))
                and isinstance(metrics.get("p95_latency_ms"), (int, float))
                and isinstance(metrics.get("mean_context_tokens"), (int, float))
                and isinstance(baseline["metrics"].get("mean_context_tokens"), (int, float))
                and metrics["mean_score"] >= baseline_score + 0.5
                and metrics["p95_latency_ms"] <= baseline_p95 * 1.5
                and metrics["mean_context_tokens"]
                <= baseline["metrics"]["mean_context_tokens"] * 1.5
                and pairwise[key]["candidate_wins"] >= 4
                and pairwise[key]["candidate_losses"] <= 2
                and _evidence_contract_ok(reports[key])
            ):
                candidates.append(key)
    summary = {
        "schema_version": 1,
        "benchmark_name": "kite-ai-papers",
        "upstream_repository": baseline["benchmark"].get("upstream_repository", KITE_REPOSITORY),
        "upstream_commit": baseline["benchmark"]["upstream_commit"],
        "query_sha256": baseline["benchmark"]["query_sha256"],
        "corpus_file_count": baseline["benchmark"]["corpus_file_count"],
        "corpus_file_sha256": baseline["benchmark"]["corpus_file_sha256"],
        "generation_model": baseline["generation"]["model"],
        "judge_model": baseline["judge"]["model"],
        "judge_prompt_version": baseline["judge"]["prompt_version"],
        "pipelines": {
            key: {
                "mean_score": reports[key]["metrics"].get("mean_score"),
                "valid_count": reports[key]["metrics"].get("valid_count"),
                "case_count": reports[key]["metrics"].get("case_count"),
                "p95_latency_ms": reports[key]["metrics"].get("p95_latency_ms"),
                "mean_context_tokens": reports[key]["metrics"].get("mean_context_tokens"),
                "report": f"{key}/report.json",
            }
            for key in PIPELINES
        },
        "pairwise_vs_b1": pairwise,
        "promotion_candidates": candidates,
        "promotion_gate": {
            "minimum_score_delta": 0.5,
            "minimum_wins": 4,
            "maximum_losses": 2,
            "max_p95_multiplier": 1.5,
            "max_context_multiplier": 1.5,
            "evidence_contract_checked": True,
        },
        "production_decision": {
            "default_pipeline": "b1",
            "default_name": "v1_flat_rerank",
            "auto_switch": False,
            "rationale": "KITE is evidence for a decision; it does not mutate the product default.",
        },
    }
    summary_path = runs_dir / "summary.json"
    _write_json_atomic(summary_path, summary)
    _write_reports(summary, reports, repo_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _run_case(
    graph: Any,
    case: KiteCase,
    *,
    judge: dict[str, Any],
    thread_prefix: str,
    pipeline_name: str,
    pipeline_config_hash: str,
    evidence_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    started = perf_counter()
    row: dict[str, Any] = {
        "case_id": case.id,
        "source_index": case.source_index,
        "query": case.query,
        "reference_answer": case.reference_answer,
        "rubric": case.rubric,
        "pipeline_name": pipeline_name,
        "pipeline_config_hash": pipeline_config_hash,
        "answer": None,
        "evidence": [],
        "context_tokens": None,
        "llm_calls": None,
        "input_tokens": None,
        "output_tokens": None,
        "judge_model": judge.get("model"),
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "kite_score": None,
        "judge_score": None,
        "judge_attempts": 0,
        "judge_error": None,
        "run_error": None,
        "error": None,
    }
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=case.query)]},
            config={"configurable": {"thread_id": f"{thread_prefix}-{case.id}"}},
        )
        grounded = result.get("groundedAnswer") if isinstance(result, dict) else None
        if not isinstance(grounded, dict) or not str(grounded.get("answer", "")).strip():
            raise RuntimeError("graph did not return a grounded answer")
        if result.get("answerGenerationFailed"):
            raise RuntimeError("graph answer generation used the failure fallback")
        row["answer"] = str(grounded["answer"]).strip()
        row["evidence"] = _public_evidence(
            grounded.get("evidence", []),
            evidence_lookup=evidence_lookup,
            retrieved_evidence=_retrieved_evidence(result),
        )
        row["context"] = _context_diagnostics(result)
        row["context_tokens"] = row["context"]["total_tokens"]
        row["kite_score"], row["judge_attempts"], judge_error = _judge_answer(
            case,
            row["answer"],
            judge,
        )
        row["judge_score"] = row["kite_score"]
        if judge_error:
            row["judge_error"] = judge_error
            row["error"] = judge_error
    except Exception as exc:
        row["run_error"] = f"{type(exc).__name__}: {exc}"
        row["error"] = row["run_error"]
        LOGGER.exception("KITE case failed: %s", case.id)
    row["latency_ms"] = round((perf_counter() - started) * 1000, 4)
    return row


def _judge_answer(case: KiteCase, answer: str, judge: dict[str, Any]) -> tuple[int | None, int, str | None]:
    prompt = JUDGE_PROMPT.format(
        query=case.query,
        ground_truth_answer=case.reference_answer,
        rubric=case.rubric,
        model_answer=answer,
    )
    attempts = 0
    last_error: str | None = None
    llm = get_llm_by_type(str(judge["task_type"])).model_copy(
        update={
            "temperature": 0,
            "request_timeout": float(judge["timeout_seconds"]),
        }
    )
    for _ in range(2):
        attempts += 1
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = str(getattr(response, "content", response)).strip()
            score = _parse_judge_score(content)
            if score is not None:
                return score, attempts, None
            last_error = f"invalid judge output: {content[:120]}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return None, attempts, last_error or "judge failed"


def _parse_judge_score(content: object) -> int | None:
    match = JUDGE_SCORE_RE.fullmatch(str(content).strip())
    return int(match.group(1)) if match else None


def _build_report(**kwargs: Any) -> dict[str, Any]:
    rows = kwargs["rows"]
    valid = [row for row in rows if row["judge_score"] is not None and not row["error"]]
    latencies = [float(row["latency_ms"]) for row in rows]
    scores = [int(row["judge_score"]) for row in valid]
    context_rows = [row.get("context", {}) for row in rows]
    return {
        "schema_version": 1,
        "benchmark": {
            "name": "kite-ai-papers",
            "upstream_repository": KITE_REPOSITORY,
            "upstream_commit": kwargs["manifest"]["upstream_commit"],
            "query_sha256": kwargs["manifest"]["query_sha256"],
            "corpus_file_count": kwargs["manifest"]["corpus_file_count"],
            "corpus_file_sha256": kwargs["manifest"]["corpus_file_sha256"],
        },
        "pipeline": {
            "key": kwargs["config"]["pipeline"]["name"],
            "config": asdict(kwargs["pipeline"]),
            "config_hash": kwargs["pipeline"].config_hash(),
        },
        "generation": {
            "model": kwargs["generation"]["model"],
            "task_models": kwargs["generation"].get("task_models", {}),
            "strategy": kwargs["generation"].get("strategy", "fixed"),
        },
        "judge": {
            "task_type": kwargs["judge"]["task_type"],
            "model": kwargs["judge"]["model"],
            "prompt_version": JUDGE_PROMPT_VERSION,
            "temperature": 0,
            "timeout_seconds": kwargs["judge"]["timeout_seconds"],
        },
        "evidence_policy": "retrieval-owned metadata and quote_text; retrieval_text is never public",
        "embedding": _embedding_contract(kwargs["settings"]),
        "reranker": kwargs["index_manifest"]["reranker"],
        "provenance": {
            "config_path": _logical_path(kwargs["config_path"], kwargs["repo_root"]),
            "config_sha256": sha256_file(kwargs["config_path"]),
            "manifest_path": _logical_path(kwargs["manifest_path"], kwargs["repo_root"]),
            "manifest_sha256": kwargs["manifest_sha"],
            "parser_artifact_path": _logical_path(kwargs["artifact_path"], kwargs["repo_root"]),
            "parser_artifact_sha256": kwargs["artifact_sha"],
            "index_path": _logical_path(kwargs["index_dir"], kwargs["repo_root"]),
            "index_manifest_sha256": sha256_file(kwargs["index_dir"] / "manifest.json"),
            "code": kwargs["code_state"],
        },
        "metrics": {
            "case_count": len(rows),
            "valid_count": len(valid),
            "mean_score": round(mean(scores), 4) if len(valid) == len(rows) and scores else None,
            "p50_latency_ms": round(_percentile(latencies, 0.5), 4),
            "p95_latency_ms": round(_percentile(latencies, 0.95), 4),
            "mean_evidence_count": round(mean(len(row["evidence"]) for row in rows), 4) if rows else 0,
            "mean_context_tokens": round(mean(float(item.get("total_tokens", 0)) for item in context_rows), 4) if context_rows else 0,
            "judge_retry_count": sum(max(0, int(row["judge_attempts"]) - 1) for row in rows),
            "failed_case_ids": [row["case_id"] for row in rows if row["error"]],
            "run_error_count": sum(1 for row in rows if row.get("run_error")),
            "judge_error_count": sum(1 for row in rows if row.get("judge_error")),
        },
        "runtime": kwargs["runtime"],
        "formal_run": not kwargs["code_state"]["dirty"],
        "started_at": kwargs["started_at"],
        "completed_at": kwargs["completed_at"],
        "cases": rows,
    }


def _context_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    packed = result.get("packedContexts", [])
    contexts = [item for item in packed if isinstance(item, dict)]
    return {
        "packed_context_count": len(contexts),
        "total_tokens": sum(int(item.get("total_tokens") or 0) for item in contexts),
        "passage_count": sum(int(item.get("passage_count") or 0) for item in contexts),
        "message_count": len(result.get("messages", [])) if isinstance(result.get("messages"), list) else 0,
    }


def _public_evidence(
    value: object,
    *,
    evidence_lookup: dict[str, dict[str, Any]] | None = None,
    retrieved_evidence: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id") or item.get("passage_id") or "").strip()
        retrieved = (retrieved_evidence or {}).get(node_id)
        if retrieved_evidence is not None and not retrieved:
            raise KiteDataError(f"Answer evidence was not retrieved: {node_id or '<missing>'}")
        canonical = (evidence_lookup or {}).get(node_id)
        if not canonical:
            raise KiteDataError(f"Answer evidence is absent from the parser artifact: {node_id or '<missing>'}")
        item = {
            "doc_id": canonical.get("doc_id") or (retrieved or {}).get("doc_id"),
            "node_id": node_id,
            "paper_id": canonical.get("paper_id"),
            "paper_title": canonical.get("paper_title"),
            "source": canonical.get("source"),
            "section_path": canonical.get("section_path")
            or canonical.get("heading_path")
            or [],
            "page": canonical.get("page_start") or canonical.get("page"),
            "quote": canonical.get("quote_text"),
            "score": (retrieved or {}).get("score"),
            "relevance": (retrieved or {}).get("relevance"),
        }
        result.append(
            {key: item.get(key) for key in PUBLIC_EVIDENCE_KEYS if key in item}
        )
    return result


def _retrieved_evidence(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("node_id") or item.get("passage_id") or "").strip(): item
        for group in result.get("evidenceGroups", [])
        if isinstance(group, dict)
        for item in group.get("evidence", [])
        if isinstance(item, dict)
    }


def _load_or_build_parser_artifact(settings: AppSettings, *, artifact_path: Path, corpus_root: Path, expected_manifest: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    if not artifact_path.exists():
        artifact, artifact_sha = build_kite_parser_artifact(
            settings,
            corpus_root=corpus_root,
            output_path=artifact_path,
        )
    else:
        artifact_sha = sha256_file(artifact_path)
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != 1 or artifact.get("kind") != "kite-parser-artifact":
        raise KiteDataError("Invalid KITE parser artifact schema.")
    if (
        artifact.get("parser_name") != KITE_PARSER_NAME
        or artifact.get("parser_version") != KITE_PARSER_VERSION
        or artifact.get("normalization_version") != NORMALIZATION_VERSION
        or artifact.get("embedding_max_input_chars") != settings.embedding_max_input_chars
    ):
        raise KiteDataError("KITE parser artifact settings mismatch.")
    if artifact.get("corpus_manifest") != expected_manifest:
        raise KiteDataError("KITE parser artifact corpus differs from manifest.")
    if len(artifact.get("papers", [])) != len(expected_manifest):
        raise KiteDataError("KITE parser artifact paper count mismatch.")
    papers = artifact["papers"]
    expected_files = {row["file_name"]: row["sha256"] for row in expected_manifest}
    if any(
        expected_files.get(paper.get("file_name")) != paper.get("file_sha256")
        or not isinstance(paper.get("passages"), list)
        or not paper["passages"]
        for paper in papers
    ):
        raise KiteDataError("KITE parser artifact paper provenance mismatch.")
    return artifact, artifact_sha


def _load_and_validate_manifest(path: Path, *, query_path: Path, corpus_root: Path, benchmark: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise KiteDataError(f"KITE manifest does not exist: {path}; run prepare first.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise KiteDataError("KITE manifest must be a JSON object.")
    validate_manifest_payload(manifest)
    expected = build_kite_manifest(
        query_path=query_path,
        corpus_root=corpus_root,
        upstream_commit=str(benchmark.get("upstream_commit", KITE_COMMIT)),
        expected_query_sha256=str(benchmark.get("query_sha256", KITE_QUERY_SHA256)),
        expected_case_count=int(benchmark.get("case_count", KITE_CASE_COUNT)),
        expected_empty_rubric_count=int(benchmark.get("empty_rubric_count", KITE_EMPTY_RUBRIC_COUNT)),
        expected_corpus_file_count=int(benchmark.get("corpus_file_count", KITE_CORPUS_FILE_COUNT)),
        expected_corpus_sha256=str(benchmark.get("corpus_sha256", KITE_CORPUS_SHA256)),
    )
    immutable = tuple(key for key in expected if key != "created_at")
    if any(manifest.get(key) != expected[key] for key in immutable):
        raise KiteDataError("KITE manifest does not match the frozen checkout.")
    return manifest


def _runtime_settings(config: dict[str, Any], repo_root: Path) -> tuple[AppSettings, dict[str, Any], dict[str, Any]]:
    settings = load_settings(base_dir=repo_root)
    embedding = _mapping(config.get("embedding"), "embedding")
    actual = _embedding_contract(settings)
    mismatches = [key for key, value in embedding.items() if actual.get(key) != value]
    if mismatches:
        raise KiteDataError("Frozen embedding configuration mismatch: " + ", ".join(mismatches))
    if settings.offline_mode or not settings.embedding_api_key or not settings.embedding_api_base:
        raise KiteDataError("KITE evaluation requires the configured embedding provider.")
    generation = _mapping(config.get("generation"), "generation")
    if generation.get("strategy", "fixed") != "fixed":
        raise KiteDataError("KITE generation.strategy must be fixed.")
    model = str(generation.get("model") or settings.llm_model).strip()
    if not model or not settings.llm_api_key or not settings.llm_api_base:
        raise KiteDataError("KITE evaluation requires a generation model and API configuration.")
    judge = _mapping(config.get("judge"), "judge")
    if judge.get("prompt_version", JUDGE_PROMPT_VERSION) != JUDGE_PROMPT_VERSION or judge.get("temperature", 0) != 0:
        raise KiteDataError("KITE judge config must use the frozen prompt and temperature 0.")
    judge = dict(judge)
    judge["task_type"] = str(judge.get("task_type", "kite_judge"))
    judge["model"] = str(judge.get("model") or model).strip()
    try:
        judge["timeout_seconds"] = float(judge.get("timeout_seconds", 60))
    except (TypeError, ValueError) as exc:
        raise KiteDataError("KITE judge timeout_seconds must be positive.") from exc
    if judge["timeout_seconds"] <= 0:
        raise KiteDataError("KITE judge timeout_seconds must be positive.")
    task_models = dict(settings.llm_task_models)
    configured_tasks = generation.get("task_models", {})
    if configured_tasks is not None:
        if not isinstance(configured_tasks, dict):
            raise KiteDataError("generation.task_models must be a mapping.")
        task_models.update({str(key): str(value) for key, value in configured_tasks.items()})
    task_models[judge["task_type"]] = judge["model"]
    return replace(
        settings,
        llm_model=model,
        llm_task_models=task_models,
        answer_strategy="fixed",
        offline_mode=False,
    ), {**generation, "model": model}, judge


def _find_reusable_index(root: Path, *, pipeline: Any, settings: AppSettings, parser_artifact_sha256: str, exclude: str) -> tuple[str, Path] | None:
    if not root.is_dir():
        return None
    for candidate in PIPELINES:
        if candidate == exclude:
            continue
        manifest_path = root / candidate / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if (
            manifest.get("embedding") == _embedding_contract(settings)
            and manifest.get("retrieval") == pipeline.index_contract()
            and manifest.get("parser_artifact_sha256") == parser_artifact_sha256
        ):
            return candidate, manifest_path.parent
    return None


def _validate_reports(
    reports: dict[str, dict[str, Any]],
    repo_root: Path,
    runs_dir: Path | None = None,
) -> None:
    expected_ids = {f"ai-papers-{index:03d}" for index in range(1, KITE_CASE_COUNT + 1)}
    expected_benchmark = {
        "name": "kite-ai-papers",
        "upstream_repository": KITE_REPOSITORY,
        "upstream_commit": KITE_COMMIT,
        "query_sha256": KITE_QUERY_SHA256,
        "corpus_file_count": KITE_CORPUS_FILE_COUNT,
        "corpus_file_sha256": KITE_CORPUS_SHA256,
    }
    baseline_identity: dict[str, Any] | None = None
    baseline_cases: dict[str, tuple[Any, ...]] | None = None
    for key in PIPELINES:
        report = reports.get(key)
        if not isinstance(report, dict) or report.get("schema_version") != 1:
            raise KiteDataError(f"Invalid KITE {key} report schema.")
        config_path = repo_root / "evals" / "configs" / f"kite_{key}.yaml"
        config = _load_config(config_path)
        pipeline = get_pipeline_config(key)
        judge_config = _mapping(config.get("judge"), "judge")
        expected_judge = {
            "task_type": str(judge_config.get("task_type", "kite_judge")),
            "model": str(judge_config["model"]),
            "prompt_version": JUDGE_PROMPT_VERSION,
            "temperature": 0,
            "timeout_seconds": float(judge_config.get("timeout_seconds", 60)),
        }
        if (
            report.get("benchmark") != expected_benchmark
            or json.loads(
                json.dumps(report.get("pipeline", {}).get("config"))
            )
            != json.loads(json.dumps(asdict(pipeline)))
            or report.get("pipeline", {}).get("config_hash") != pipeline.config_hash()
            or report.get("generation")
            != {
                **_mapping(config.get("generation"), "generation"),
                "model": str(config["generation"]["model"]),
            }
            or report.get("judge") != expected_judge
            or report.get("embedding") != _mapping(config.get("embedding"), "embedding")
            or report.get("reranker") != _mapping(config.get("reranker"), "reranker")
            or report.get("runtime", {}).get("concurrency") != 1
        ):
            raise KiteDataError(f"KITE {key} report differs from the frozen config.")
        if report.get("pipeline", {}).get("key") != key or report.get("formal_run") is not True:
            raise KiteDataError(f"KITE {key} report is not a formal {key} run.")
        rows = report.get("cases")
        if not isinstance(rows, list) or len(rows) != KITE_CASE_COUNT:
            raise KiteDataError(f"KITE {key} report must contain exactly {KITE_CASE_COUNT} cases.")
        ids = [row.get("case_id") for row in rows if isinstance(row, dict)]
        if len(ids) != KITE_CASE_COUNT or set(ids) != expected_ids or len(set(ids)) != len(ids):
            raise KiteDataError(f"KITE {key} report case IDs do not match the frozen dataset.")
        if any(
            row.get("error")
            or type(row.get("judge_score")) is not int
            or not 0 <= row["judge_score"] <= 10
            for row in rows
        ):
            raise KiteDataError(f"KITE {key} report contains an invalid case result.")
        expected_metrics = {
            "case_count": KITE_CASE_COUNT,
            "valid_count": KITE_CASE_COUNT,
            "mean_score": round(mean(row["judge_score"] for row in rows), 4),
            "p95_latency_ms": round(
                _percentile([float(row["latency_ms"]) for row in rows], 0.95), 4
            ),
            "mean_context_tokens": round(
                mean(float(row.get("context", {}).get("total_tokens", 0)) for row in rows),
                4,
            ),
        }
        metrics = report.get("metrics", {})
        if any(metrics.get(metric) != value for metric, value in expected_metrics.items()):
            raise KiteDataError(f"KITE {key} report metrics do not match its cases.")
        if not _evidence_contract_ok(report):
            raise KiteDataError(f"KITE {key} report violates the public evidence contract.")

        provenance = report.get("provenance", {})
        if (
            provenance.get("config_path") != f"evals/configs/kite_{key}.yaml"
            or provenance.get("config_sha256") != sha256_file(config_path)
        ):
            raise KiteDataError(f"KITE {key} report config provenance mismatch.")
        code = dict(provenance.get("code", {}))
        code.pop("config_path", None)
        identity = {
            "benchmark": report.get("benchmark"),
            "generation": report.get("generation"),
            "judge": report.get("judge"),
            "embedding": report.get("embedding"),
            "manifest_sha256": provenance.get("manifest_sha256"),
            "parser_artifact_sha256": provenance.get("parser_artifact_sha256"),
            "code": code,
        }
        cases = {
            row["case_id"]: (row.get("query"), row.get("reference_answer"), row.get("rubric"))
            for row in rows
        }
        if baseline_identity is None:
            baseline_identity, baseline_cases = identity, cases
        elif identity != baseline_identity or cases != baseline_cases:
            raise KiteDataError(f"KITE {key} report is not comparable with B1.")
    if runs_dir is not None:
        _validate_provenance_files(reports, repo_root, runs_dir)


def _validate_provenance_files(
    reports: dict[str, dict[str, Any]],
    repo_root: Path,
    runs_dir: Path,
) -> None:
    manifest_path = runs_dir / "manifest.json"
    artifact_path = runs_dir / "parser_artifact.json"
    if not manifest_path.is_file() or not artifact_path.is_file():
        raise KiteDataError("KITE provenance files are incomplete.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    validate_manifest_payload(manifest)
    if (
        manifest.get("upstream_commit") != KITE_COMMIT
        or manifest.get("query_sha256") != KITE_QUERY_SHA256
        or manifest.get("corpus_file_sha256") != KITE_CORPUS_SHA256
        or artifact.get("schema_version") != 1
        or artifact.get("kind") != "kite-parser-artifact"
        or artifact.get("corpus_manifest") != manifest.get("corpus_manifest")
    ):
        raise KiteDataError("KITE provenance files differ from the frozen dataset.")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status_args = ["git", "status", "--porcelain", "--untracked-files=all", "--", "."]
    try:
        runs_relative = runs_dir.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        runs_relative = None
    if runs_relative:
        status_args.append(f":(exclude){runs_relative}")
    if subprocess.run(
        status_args,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip():
        raise KiteDataError("KITE formal report aggregation requires a clean working tree.")
    manifest_sha = sha256_file(manifest_path)
    artifact_sha = sha256_file(artifact_path)
    for key, report in reports.items():
        index_path = runs_dir / "indexes" / key / "manifest.json"
        if not index_path.is_file():
            raise KiteDataError(f"KITE {key} index manifest is missing.")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        provenance = report["provenance"]
        code = provenance["code"]
        if (
            provenance.get("manifest_sha256") != manifest_sha
            or provenance.get("parser_artifact_sha256") != artifact_sha
            or provenance.get("index_manifest_sha256") != sha256_file(index_path)
            or code.get("commit") != head
            or code.get("dirty") is not False
            or index.get("pipeline_key") != key
            or index.get("pipeline_config_hash") != report["pipeline"]["config_hash"]
            or index.get("parser_artifact_sha256") != artifact_sha
            or index.get("embedding") != report["embedding"]
            or index.get("reranker") != report["reranker"]
        ):
            raise KiteDataError(f"KITE {key} provenance does not match the current files.")


def _pairwise_scores(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_rows = {row["case_id"]: row for row in baseline.get("cases", [])}
    candidate_rows = {row["case_id"]: row for row in candidate.get("cases", [])}
    wins = ties = losses = 0
    win_ids: list[str] = []
    tie_ids: list[str] = []
    loss_ids: list[str] = []
    for case_id in sorted(base_rows.keys() & candidate_rows.keys()):
        left = base_rows[case_id].get("judge_score")
        right = candidate_rows[case_id].get("judge_score")
        if not isinstance(left, int) or not isinstance(right, int):
            continue
        if right > left:
            wins += 1
            win_ids.append(case_id)
        elif right == left:
            ties += 1
            tie_ids.append(case_id)
        else:
            losses += 1
            loss_ids.append(case_id)
    return {
        "candidate_wins": wins,
        "ties": ties,
        "candidate_losses": losses,
        "win_case_ids": win_ids,
        "tie_case_ids": tie_ids,
        "loss_case_ids": loss_ids,
    }


def _evidence_contract_ok(report: dict[str, Any]) -> bool:
    for row in report.get("cases", []):
        if row.get("error"):
            return False
        evidence = row.get("evidence")
        if not isinstance(evidence, list):
            return False
        for item in evidence:
            if not isinstance(item, dict) or "retrieval_text" in item:
                return False
            if not str(item.get("source", "")).strip() or not str(item.get("quote", "")).strip():
                return False
            if type(item.get("page")) is not int or not item.get("section_path"):
                return False
    return True


def _write_reports(summary: dict[str, Any], reports: dict[str, Any], repo_root: Path) -> None:
    lines = [
        "# KITE AI Papers Benchmark",
        "",
        "Frozen upstream repository: `" + reports["b1"]["benchmark"].get("upstream_repository", KITE_REPOSITORY) + "`.",
        "",
        "Frozen upstream commit: `" + reports["b1"]["benchmark"]["upstream_commit"] + "`.",
        "",
        "Frozen query SHA-256: `" + reports["b1"]["benchmark"]["query_sha256"] + "`; corpus: `" + str(reports["b1"]["benchmark"]["corpus_file_count"]) + "` PDFs; corpus manifest SHA-256: `" + reports["b1"]["benchmark"]["corpus_file_sha256"] + "`.",
        "",
        "Generation model: `" + reports["b1"]["generation"]["model"] + "`; judge model: `" + reports["b1"]["judge"]["model"] + "`; prompt: `" + reports["b1"]["judge"]["prompt_version"] + "`.",
        "",
        "| Pipeline | Mean score | Valid cases | p95 latency (ms) | Mean context tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in PIPELINES:
        item = summary["pipelines"][key]
        lines.append(
            f"| {key} | {item['mean_score']} | {item['valid_count']}/{item['case_count']} | "
            f"{item['p95_latency_ms']} | {item['mean_context_tokens']} |"
        )
    lines.extend(
        [
            "",
            "Scores are judge outputs on the frozen KITE protocol; per-case evidence and diagnostics remain in the JSON reports.",
            "",
            "## Pairwise results vs B1",
            "",
        ]
    )
    for key, value in summary["pairwise_vs_b1"].items():
        lines.append(
            f"- {key}: {value['candidate_wins']} wins, {value['ties']} ties, "
            f"{value['candidate_losses']} losses; wins=`{', '.join(value['win_case_ids']) or 'none'}`, "
            f"losses=`{', '.join(value['loss_case_ids']) or 'none'}`"
        )
    lines.extend(
        [
            "",
            "## Evidence audit",
            "",
            "Every reported case has an integer score and every public evidence item was canonicalized from the parser artifact by passage ID. Reports contain source, paper, section, page and source-faithful quote fields; `retrieval_text` is not public.",
            "",
            "## Decision gate",
            "",
            f"Promotion candidates before a separate production approval: {', '.join(summary['promotion_candidates']) or 'none'}.",
            "",
            "Known candidate regressions remain visible in the pairwise case lists. This score uses the KITE protocol with the configured local models and is not comparable to the upstream absolute score.",
        ]
    )
    (repo_root / "docs" / "kite_benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision = summary["production_decision"]
    pairwise_text = "; ".join(
        f"{key.upper()} `{value['candidate_wins']}/{value['ties']}/{value['candidate_losses']}`"
        for key, value in summary["pairwise_vs_b1"].items()
    )
    regressions = "; ".join(
        f"{key.upper()} losses: {', '.join(value['loss_case_ids']) or 'none'}"
        for key, value in summary["pairwise_vs_b1"].items()
    )
    candidates = ", ".join(summary["promotion_candidates"]) or "none"
    (repo_root / "docs" / "production_pipeline_decision.md").write_text(
        "# Production Pipeline Decision\n\n"
        f"- Benchmark: `{summary['benchmark_name']}`, upstream commit `{summary['upstream_commit']}`.\n"
        f"- Query SHA-256: `{summary['query_sha256']}`; corpus: `{summary['corpus_file_count']}` PDFs; corpus manifest SHA-256: `{summary['corpus_file_sha256']}`.\n"
        f"- KITE-protocol scores: B0 `{summary['pipelines']['b0']['mean_score']}`, B1 `{summary['pipelines']['b1']['mean_score']}`, B2 `{summary['pipelines']['b2']['mean_score']}`, B3 `{summary['pipelines']['b3']['mean_score']}`.\n"
        f"- Pairwise wins/ties/losses vs B1: {pairwise_text}.\n"
        f"- Latency/context: B1 p95 `{summary['pipelines']['b1']['p95_latency_ms']} ms`, `{summary['pipelines']['b1']['mean_context_tokens']} tokens`; B2 p95 `{summary['pipelines']['b2']['p95_latency_ms']} ms`, `{summary['pipelines']['b2']['mean_context_tokens']} tokens`; B3 p95 `{summary['pipelines']['b3']['p95_latency_ms']} ms`, `{summary['pipelines']['b3']['mean_context_tokens']} tokens`.\n"
        "- Existing internal diagnostic result: M3.2 kept fixed B1 as the frozen default; M4.1 adaptive rechecks did not prove net benefit.\n"
        f"- Known case regressions: {regressions}.\n"
        "- Evidence audit: all 15 cases per pipeline have valid scores; evidence was normalized from retrieval-owned parser records and excludes `retrieval_text`.\n"
        f"- Default: `{decision['default_pipeline']}` (`{decision['default_name']}`).\n"
        f"- Automatic switch: `{decision['auto_switch']}`.\n"
        f"- Promotion candidates: {candidates}.\n"
        f"- Decision: keep B1 active; any candidate ({candidates}) requires separate production approval.\n"
        f"- Reason: {decision['rationale']}\n",
        encoding="utf-8",
    )


def _select_cases(cases: list[KiteCase], selected: object) -> list[KiteCase]:
    if selected is None:
        return cases
    if not isinstance(selected, list) or not selected:
        raise KiteDataError("runtime.case_ids must be a non-empty list.")
    wanted = {str(value) for value in selected}
    result = [case for case in cases if case.id in wanted]
    if {case.id for case in result} != wanted:
        raise KiteDataError("runtime.case_ids contains an unknown KITE case.")
    return result


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise KiteDataError(f"KITE config does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise KiteDataError(f"Invalid KITE config: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise KiteDataError("KITE config schema_version must be 1.")
    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, dict) or benchmark.get("name") != "kite-ai-papers":
        raise KiteDataError("KITE config benchmark.name must be kite-ai-papers.")
    for field in ("query_path", "corpus_root"):
        if not isinstance(benchmark.get(field), str) or not benchmark[field]:
            raise KiteDataError(f"KITE config benchmark.{field} is required.")
    generation = payload.get("generation")
    if isinstance(generation, dict) and generation.get("strategy", "fixed") != "fixed":
        raise KiteDataError("KITE generation.strategy must be fixed.")
    return payload


def _mapping(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise KiteDataError(f"KITE config {name} must be a mapping.")
    return dict(value)


def _resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def _resolve_output_path(value: object, repo_root: Path) -> Path:
    path = _resolve_path(str(value), repo_root)
    artifacts = (repo_root / "artifacts").resolve()
    if not path.is_relative_to(artifacts):
        raise KiteDataError("KITE outputs must stay under artifacts/.")
    return path


def _logical_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return f"external/{path.name}"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the frozen KITE evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("prepare", "validate data and write manifest"),
        ("run", "run one configured pipeline"),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument("--config", required=True, type=Path)
        if command == "run":
            subparser.add_argument(
                "--allow-dirty",
                action="store_true",
                help="allow a non-formal diagnostic run from a dirty tree",
            )
    report = subparsers.add_parser("report", help="compare pipeline reports")
    report.add_argument(
        "--runs",
        "--runs-dir",
        dest="runs_dir",
        default="artifacts/evals/kite",
        type=Path,
    )
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_from_config(args.config.resolve())
    elif args.command == "run":
        run_from_config(args.config.resolve(), allow_dirty=args.allow_dirty)
    else:
        report_from_runs(_resolve_output_path(args.runs_dir, Path(__file__).resolve().parent.parent))


if __name__ == "__main__":
    main()


__all__ = ["JUDGE_PROMPT", "main", "prepare_from_config", "report_from_runs", "run_from_config"]
