"""Reproducible M3 retrieval evaluation runner."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean
import subprocess
from time import perf_counter
from typing import Any

import yaml

from core.persistence import load_bm25_bundle
from core.rag_answer import format_retrieval_only_answer
from core.settings import AppSettings, load_settings
from evals.metrics import ndcg_at_k, reciprocal_rank, recall_at_k
from evals.v2_corpus import (
    artifact_documents,
    artifact_id_sets,
    build_parser_artifact,
    load_parser_artifact,
    sha256_file,
)
from indexing.bm25_index import create_lexical_store
from indexing.indexer import Indexer
from indexing.retrieval_pipeline import (
    RetrievalPipelineConfig,
    document_key,
    get_pipeline_config,
)
from indexing.retriever import FusionRetriever


CATEGORIES = (
    "exact_term_definition",
    "method_section_location",
    "experiment_number_table",
    "cross_paper_or_section",
)


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    case_id: str
    question: str
    category: str
    gold_passage_ids: tuple[str, ...]
    gold_paper_ids: tuple[str, ...]
    gold_section_ids: tuple[str, ...]
    tags: tuple[str, ...]
    notes: str


def run_from_config(config_path: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    config = _load_config(config_path, repo_root=repo_root)
    settings = load_settings(base_dir=repo_root)
    _validate_frozen_runtime(settings, config)

    artifact_path = Path(config["parser_artifact"])
    expected_artifact_sha = str(
        config.get("parser_artifact_sha256") or ""
    ).casefold()
    if not artifact_path.exists():
        _, built_sha = build_parser_artifact(
            settings,
            corpus_dir=Path(config["corpus_dir"]),
            output_path=artifact_path,
            parser_gold_path=Path(config["parser_gold"]),
        )
        if expected_artifact_sha and built_sha != expected_artifact_sha:
            raise ValueError(
                "Fresh parser artifact does not match frozen checksum."
            )
    artifact, artifact_sha = load_parser_artifact(
        artifact_path,
        expected_sha256=expected_artifact_sha or None,
        corpus_dir=Path(config["corpus_dir"]),
    )
    documents = artifact_documents(artifact)
    cases = load_retrieval_cases(
        Path(config["retrieval_dataset"]),
        artifact=artifact,
    )
    answer_cases = load_answer_smoke_cases(
        Path(config["answer_smoke_dataset"])
    )
    dataset_sha = sha256_file(Path(config["retrieval_dataset"]))
    answer_dataset_sha = sha256_file(
        Path(config["answer_smoke_dataset"])
    )
    code_state = _code_state(repo_root)

    run_name = str(config["run_name"])
    run_dir = Path(config["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pipeline_reports: dict[str, Any] = {}
    fresh_index_sources: dict[str, tuple[str, Path]] = {}
    for pipeline_key in config["pipelines"]:
        pipeline = get_pipeline_config(str(pipeline_key))
        index_fingerprint = _config_fingerprint(
            {
                "embedding": _embedding_contract(settings),
                "retrieval": pipeline.index_contract(),
            }
        )
        reusable_source = (
            fresh_index_sources.get(index_fingerprint)
            if str(pipeline_key).startswith("b2_")
            else None
        )
        variant_settings, index_dir = _prepare_variant_settings(
            settings,
            run_dir=run_dir,
            pipeline_key=str(pipeline_key),
        )
        manifest = _build_eval_index(
            variant_settings,
            documents=documents,
            pipeline_key=str(pipeline_key),
            pipeline=pipeline,
            index_dir=index_dir,
            parser_artifact_sha256=artifact_sha,
            corpus_manifest=artifact["corpus_manifest"],
            dataset_sha256=dataset_sha,
            code_state=code_state,
            reranker=config["reranker"],
            force_reindex=bool(config.get("force_reindex", True)),
            reuse_source=reusable_source,
        )
        fresh_index_sources.setdefault(
            index_fingerprint,
            (str(pipeline_key), index_dir),
        )
        retriever = _load_eval_retriever(
            variant_settings,
            pipeline=pipeline,
            pipeline_key=str(pipeline_key),
            index_dir=index_dir,
            manifest=manifest,
            parser_artifact_sha256=artifact_sha,
            reranker=config["reranker"],
        )
        retrieval_report = evaluate_retrieval(
            cases,
            retriever=retriever,
        )
        answer_smoke = evaluate_answer_smoke(
            answer_cases,
            retriever=retriever,
        )
        pipeline_reports[str(pipeline_key)] = {
            "pipeline": asdict(pipeline),
            "pipeline_config_hash": pipeline.config_hash(),
            "index_manifest": manifest,
            "index_manifest_sha256": sha256_file(
                index_dir / "manifest.json"
            ),
            "retrieval": retrieval_report,
            "answer_smoke": answer_smoke,
        }

    report = {
        "schema_version": 1,
        "run_name": run_name,
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "parser_artifact": str(artifact_path),
        "parser_artifact_sha256": artifact_sha,
        "parser_gold_sha256": artifact["parser_gold"]["sha256"],
        "retrieval_dataset": str(config["retrieval_dataset"]),
        "retrieval_dataset_sha256": dataset_sha,
        "answer_smoke_dataset": str(config["answer_smoke_dataset"]),
        "answer_smoke_dataset_sha256": answer_dataset_sha,
        "embedding": _embedding_contract(settings),
        "reranker": dict(config["reranker"]),
        "code": code_state,
        "pipelines": pipeline_reports,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "run_name": run_name,
        "report": str(report_path),
        "parser_artifact_sha256": artifact_sha,
        "dataset_sha256": dataset_sha,
        "pipelines": {
            key: value["retrieval"]["metrics"]
            for key, value in pipeline_reports.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return report


def load_retrieval_cases(
    path: Path,
    *,
    artifact: dict[str, Any],
) -> list[RetrievalCase]:
    paper_ids, section_ids, passage_ids = artifact_id_sets(artifact)
    cases: list[RetrievalCase] = []
    category_counts = {category: 0 for category in CATEGORIES}
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        case = RetrievalCase(
            case_id=str(payload["case_id"]),
            question=str(payload["question"]).strip(),
            category=str(payload["category"]).strip(),
            gold_passage_ids=tuple(
                str(value) for value in payload["gold_passage_ids"]
            ),
            gold_paper_ids=tuple(
                str(value) for value in payload["gold_paper_ids"]
            ),
            gold_section_ids=tuple(
                str(value) for value in payload["gold_section_ids"]
            ),
            tags=tuple(str(value) for value in payload.get("tags", [])),
            notes=str(payload.get("notes", "")).strip(),
        )
        if case.case_id in seen_ids:
            raise ValueError(f"Duplicate case_id at line {line_number}.")
        if case.category not in category_counts:
            raise ValueError(
                f"Unknown retrieval category at line {line_number}."
            )
        if not case.question or not case.gold_passage_ids:
            raise ValueError(
                f"Retrieval case lacks question or passage gold at line "
                f"{line_number}."
            )
        if not set(case.gold_passage_ids).issubset(passage_ids):
            raise ValueError(
                f"Unknown gold passage id at line {line_number}."
            )
        if not set(case.gold_paper_ids).issubset(paper_ids):
            raise ValueError(f"Unknown gold paper id at line {line_number}.")
        if not set(case.gold_section_ids).issubset(section_ids):
            raise ValueError(
                f"Unknown gold section id at line {line_number}."
            )
        seen_ids.add(case.case_id)
        category_counts[case.category] += 1
        cases.append(case)
    if len(cases) != 48 or any(
        count != 12 for count in category_counts.values()
    ):
        raise ValueError(
            "Frozen retrieval test must contain 48 cases, 12 per category; "
            f"found {category_counts}."
        )
    return cases


def load_answer_smoke_cases(path: Path) -> list[dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(cases) != 8:
        raise ValueError("Answer smoke dataset must contain exactly 8 cases.")
    return cases


def evaluate_retrieval(
    cases: list[RetrievalCase],
    *,
    retriever: FusionRetriever,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for case in cases:
        started = perf_counter()
        packed = retriever.retrieve(case.question)
        latency_ms = (perf_counter() - started) * 1000
        latencies.append(latency_ms)
        documents = packed.passages
        passage_ids = [
            str(
                document.metadata.get("passage_id")
                or document.metadata.get("node_id")
                or ""
            )
            for document in documents
        ]
        paper_ids = [
            str(document.metadata.get("paper_id") or "")
            for document in documents
        ]
        section_ids = [
            str(document.metadata.get("section_id") or "")
            for document in documents
        ]
        gold_passages = set(case.gold_passage_ids)
        relevances = [
            1 if passage_id in gold_passages else 0
            for passage_id in passage_ids
        ]
        first_gold_rank = next(
            (
                rank
                for rank, passage_id in enumerate(
                    passage_ids[:10],
                    start=1,
                )
                if passage_id in gold_passages
            ),
            None,
        )
        rows.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "category": case.category,
                "tags": list(case.tags),
                "notes": case.notes,
                "gold_passage_ids": list(case.gold_passage_ids),
                "gold_paper_ids": list(case.gold_paper_ids),
                "gold_section_ids": list(case.gold_section_ids),
                "predicted_passage_ids": passage_ids,
                "predicted_paper_ids": paper_ids,
                "predicted_section_ids": section_ids,
                "first_gold_rank": first_gold_rank,
                "recall_at_5": round(
                    recall_at_k(
                        relevances,
                        len(gold_passages),
                        k=5,
                    ),
                    6,
                ),
                "recall_at_10": round(
                    recall_at_k(
                        relevances,
                        len(gold_passages),
                        k=10,
                    ),
                    6,
                ),
                "mrr_at_10": round(
                    reciprocal_rank(relevances[:10]),
                    6,
                ),
                "ndcg_at_10": round(
                    ndcg_at_k(
                        relevances,
                        len(gold_passages),
                        k=10,
                    ),
                    6,
                ),
                "paper_recall_at_10": round(
                    _set_recall(
                        set(paper_ids[:10]),
                        set(case.gold_paper_ids),
                    ),
                    6,
                ),
                "section_recall_at_10": round(
                    _set_recall(
                        set(section_ids[:10]),
                        set(case.gold_section_ids),
                    ),
                    6,
                ),
                "latency_ms": round(latency_ms, 4),
                "total_tokens": packed.total_tokens,
                "stage_results": packed.debug["stages"],
                "stage_timings_ms": packed.debug["timings_ms"],
            }
        )
    return {
        "metrics": _aggregate_rows(rows, latencies=latencies),
        "subsets": {
            category: _aggregate_rows(
                [row for row in rows if row["category"] == category],
                latencies=[
                    row["latency_ms"]
                    for row in rows
                    if row["category"] == category
                ],
            )
            for category in CATEGORIES
        },
        "cases": rows,
        "bad_cases": [
            row
            for row in rows
            if row["recall_at_10"] < 1.0
        ],
    }


def evaluate_answer_smoke(
    cases: list[dict[str, Any]],
    *,
    retriever: FusionRetriever,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for payload in cases:
        question = str(payload["question"])
        packed = retriever.retrieve(question)
        answer = format_retrieval_only_answer(
            question,
            packed.passages,
        )
        prefix_leaks = [
            document_key(document)
            for document in packed.passages
            if any(
                marker in document.page_content
                for marker in (
                    "[TITLE]",
                    "[AUTHORS]",
                    "[YEAR]",
                    "[SECTION]",
                    "[BLOCK]",
                )
            )
        ]
        rows.append(
            {
                "case_id": str(payload["case_id"]),
                "question": question,
                "answer_preview": answer[:500],
                "evidence_count": len(packed.passages),
                "metadata_prefix_leaks": prefix_leaks,
            }
        )
    return {
        "formal_answer_test": False,
        "case_count": len(rows),
        "metadata_prefix_leak_count": sum(
            len(row["metadata_prefix_leaks"]) for row in rows
        ),
        "cases": rows,
    }


def _load_config(path: Path, *, repo_root: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported V2 eval config schema.")
    for key in (
        "output_dir",
        "corpus_dir",
        "parser_artifact",
        "parser_gold",
        "retrieval_dataset",
        "answer_smoke_dataset",
    ):
        candidate = Path(str(payload[key]))
        payload[key] = str(
            candidate
            if candidate.is_absolute()
            else (repo_root / candidate).resolve()
        )
    return payload


def _validate_frozen_runtime(
    settings: AppSettings,
    config: dict[str, Any],
) -> None:
    expected = dict(config["embedding"])
    actual = _embedding_contract(settings)
    mismatches = [
        key for key, value in expected.items() if actual.get(key) != value
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: config={expected[key]!r}, runtime={actual.get(key)!r}"
            for key in mismatches
        )
        raise ValueError(
            f"Frozen embedding configuration mismatch ({details})."
        )
    if settings.offline_mode:
        raise ValueError("M3 evaluation forbids fake/offline embeddings.")
    if not settings.embedding_api_key or not settings.embedding_api_base:
        raise ValueError(
            "M3 evaluation requires the frozen embedding provider."
        )
    reranker = dict(config["reranker"])
    if reranker.get("backend") != "flashrank":
        raise ValueError("M3 frozen reranker must be flashrank.")


def _prepare_variant_settings(
    settings: AppSettings,
    *,
    run_dir: Path,
    pipeline_key: str,
) -> tuple[AppSettings, Path]:
    index_dir = run_dir / "indexes" / pipeline_key
    prepared = replace(
        settings,
        data_dir=run_dir,
        index_dir=index_dir,
        faiss_dir=index_dir / "faiss",
        bm25_path=index_dir / "bm25.pkl",
        nodes_path=index_dir / "nodes.jsonl",
        doc_trees_path=index_dir / "doc_trees.json",
        retrieval_pipeline=pipeline_key,
        retriever_k=8,
        flashrank_top_n=30,
        max_context_tokens=8000,
        offline_mode=False,
    )
    return prepared, index_dir


def _build_eval_index(
    settings: AppSettings,
    *,
    documents: list[Any],
    pipeline_key: str,
    pipeline: RetrievalPipelineConfig,
    index_dir: Path,
    parser_artifact_sha256: str,
    corpus_manifest: list[dict[str, str]],
    dataset_sha256: str,
    code_state: dict[str, Any],
    reranker: dict[str, Any],
    force_reindex: bool,
    reuse_source: tuple[str, Path] | None,
) -> dict[str, Any]:
    if index_dir.exists() and force_reindex:
        _remove_index_dir(index_dir)
    manifest_path = index_dir / "manifest.json"
    if manifest_path.exists() and not force_reindex:
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    index_dir.mkdir(parents=True, exist_ok=True)
    reused_from: str | None = None
    if reuse_source is not None:
        reused_from, source_dir = reuse_source
        shutil.copytree(source_dir / "faiss", index_dir / "faiss")
        shutil.copy2(source_dir / "bm25.pkl", index_dir / "bm25.pkl")
    else:
        config = settings.indexer_config()
        config["retriever"]["pipeline"] = pipeline_key
        config["retriever"]["reranker_backend"] = reranker["backend"]
        config["retriever"]["flashrank_model"] = reranker["model"]
        config["retriever"]["flashrank_top_n"] = pipeline.rerank_top_n
        indexer = Indexer(config)
        result = indexer.index_documents(documents)
        if result is None:
            raise ValueError(f"Pipeline {pipeline_key} produced no index.")

    manifest = {
        "schema_version": 1,
        "kind": "v2-core-eval-index",
        "pipeline_key": pipeline_key,
        "pipeline": asdict(pipeline),
        "pipeline_config_hash": pipeline.config_hash(),
        "embedding": _embedding_contract(settings),
        "retrieval": pipeline.index_contract(),
        "reranker": reranker,
        "parser_artifact_sha256": parser_artifact_sha256,
        "corpus_manifest": corpus_manifest,
        "retrieval_dataset_sha256": dataset_sha256,
        "document_count": len(documents),
        "fresh_index_reused_from": reused_from,
        "code_commit": code_state["commit"],
        "code_dirty": code_state["dirty"],
        "faiss_sha256": sha256_file(index_dir / "faiss" / "index.faiss"),
        "faiss_metadata_sha256": sha256_file(
            index_dir / "faiss" / "index.pkl"
        ),
        "bm25_sha256": sha256_file(index_dir / "bm25.pkl"),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_eval_retriever(
    settings: AppSettings,
    *,
    pipeline: RetrievalPipelineConfig,
    pipeline_key: str,
    index_dir: Path,
    manifest: dict[str, Any],
    parser_artifact_sha256: str,
    reranker: dict[str, Any],
) -> FusionRetriever:
    _validate_eval_manifest(
        settings,
        pipeline=pipeline,
        manifest=manifest,
        parser_artifact_sha256=parser_artifact_sha256,
    )
    config = settings.indexer_config()
    config["retriever"]["pipeline"] = pipeline_key
    indexer = Indexer(config)
    bundle = load_bm25_bundle(index_dir / "bm25.pkl")
    lexical_store = create_lexical_store(
        settings.lexical_backend,
        bundle=bundle,
        tokenizer=pipeline.tokenizer,
    )
    return FusionRetriever(
        vectorstore=indexer.vector_store,
        lexical_store=lexical_store,
        pipeline=pipeline,
        reranker_backend=str(reranker["backend"]),
        flashrank_model=str(reranker["model"]),
        flashrank_cache_dir=settings.flashrank_cache_dir,
        strict_reranker=pipeline.use_rerank,
    )


def _validate_eval_manifest(
    settings: AppSettings,
    *,
    pipeline: RetrievalPipelineConfig,
    manifest: dict[str, Any],
    parser_artifact_sha256: str,
) -> None:
    if manifest.get("embedding") != _embedding_contract(settings):
        raise ValueError(
            "Query embedding does not match the evaluation index manifest."
        )
    if manifest.get("retrieval") != pipeline.index_contract():
        raise ValueError(
            "Query retrieval contract does not match the index manifest."
        )
    if manifest.get("pipeline_config_hash") != pipeline.config_hash():
        raise ValueError(
            "Query pipeline config does not match the index manifest."
        )
    if manifest.get("parser_artifact_sha256") != parser_artifact_sha256:
        raise ValueError(
            "Parser artifact does not match the index manifest."
        )


def _aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    latencies: list[float],
) -> dict[str, Any]:
    metric_names = (
        "recall_at_5",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "paper_recall_at_10",
        "section_recall_at_10",
    )
    metrics = {
        name: round(mean(row[name] for row in rows), 6)
        if rows
        else 0.0
        for name in metric_names
    }
    metrics.update(
        {
            "case_count": len(rows),
            "recall_at_10_hit_count": sum(
                row["recall_at_10"] == 1.0 for row in rows
            ),
            "p50_latency_ms": round(_percentile(latencies, 0.5), 4),
            "p95_latency_ms": round(_percentile(latencies, 0.95), 4),
        }
    )
    return metrics


def _set_recall(predicted: set[str], gold: set[str]) -> float:
    if not gold:
        return 1.0
    return len(predicted & gold) / len(gold)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _embedding_contract(settings: AppSettings) -> dict[str, Any]:
    return {
        "provider": settings.embedding_provider,
        "model": settings.embedding_model,
        "dimension": settings.embedding_dimensions,
        "batch_size": settings.embedding_batch_size,
        "input_mode": settings.embedding_input_mode,
        "check_embedding_ctx_length": (
            settings.embedding_check_context_length
        ),
        "max_input_chars": settings.embedding_max_input_chars,
    }


def _code_state(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "-uall"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
    }


def _remove_index_dir(index_dir: Path) -> None:
    resolved = index_dir.resolve()
    if "artifacts" not in {part.casefold() for part in resolved.parts}:
        raise ValueError(
            f"Refusing to remove non-artifact index directory: {resolved}"
        )
    shutil.rmtree(resolved)


def _config_fingerprint(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen Agentic RAG V2 Core retrieval evaluation."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_from_config(Path(args.config).resolve())


if __name__ == "__main__":
    main()
