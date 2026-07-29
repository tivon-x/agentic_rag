"""Budgeted, reproducible M3.1 fixed-retrieval experiment runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

import yaml

from core.settings import AppSettings, load_settings
from evals.m3_1_experiments import (
    final_gate,
    pareto_frontier,
    pipeline_from_dict,
    rank_candidates,
    round1_boost_off,
    round1_seed_configs,
    round2_configs,
    round3_blend_configs,
    round3_stability_configs,
)
from evals.run_lock import exclusive_run_lock
from evals.v2_corpus import (
    artifact_documents,
    load_parser_artifact,
    sha256_file,
)
from evals.v2_runner import (
    _build_eval_index,
    _capture_code_state,
    _config_fingerprint,
    _embedding_contract,
    _load_eval_retriever,
    _prepare_run_dir,
    _validate_frozen_runtime,
    evaluate_answer_smoke,
    evaluate_retrieval_protocol,
    load_answer_smoke_cases,
    load_retrieval_cases,
)
from indexing.retrieval_pipeline import RetrievalPipelineConfig


@dataclass
class _ExperimentRuntime:
    repo_root: Path
    run_dir: Path
    settings: AppSettings
    config: dict[str, Any]
    documents: list[Any]
    cases: list[Any]
    answer_cases: list[dict[str, Any]]
    artifact: dict[str, Any]
    artifact_sha: str
    dataset_sha: str
    code_state: dict[str, Any]
    index_dirs: dict[str, Path]

    def evaluate(
        self,
        key: str,
        pipeline: RetrievalPipelineConfig,
    ) -> dict[str, Any]:
        checkpoint_path = self.run_dir / "checkpoints" / f"{key}.json"
        checkpoint_identity = _checkpoint_identity(
            key=key,
            pipeline=pipeline,
            artifact_sha=self.artifact_sha,
            dataset_sha=self.dataset_sha,
            embedding=_embedding_contract(self.settings),
            code_state=self.code_state,
        )
        if bool(self.config.get("resume")) and checkpoint_path.exists():
            checkpoint = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
            if checkpoint.get("identity") == checkpoint_identity:
                return dict(checkpoint["result"])

        index_fingerprint = _config_fingerprint(
            {
                "embedding": _embedding_contract(self.settings),
                "retrieval": pipeline.index_contract(),
            }
        )
        index_dir = self.index_dirs.get(index_fingerprint)
        is_new_index = index_dir is None
        if index_dir is None:
            index_dir = (
                self.run_dir
                / "indexes"
                / "by_contract"
                / index_fingerprint
            )
            self.index_dirs[index_fingerprint] = index_dir
        complete_index_exists = (index_dir / "manifest.json").exists()

        variant_settings, _ = _prepare_content_addressed_settings(
            self.settings,
            run_dir=self.run_dir,
            index_fingerprint=index_fingerprint,
        )
        reranker = {
            "backend": "flashrank",
            "model": pipeline.reranker_model,
            "cache_dir": str(self.config["reranker"]["cache_dir"]),
        }
        content_manifest = _build_eval_index(
            variant_settings,
            documents=self.documents,
            pipeline_key=key,
            pipeline=pipeline,
            index_dir=index_dir,
            parser_artifact_sha256=self.artifact_sha,
            corpus_manifest=self.artifact["corpus_manifest"],
            dataset_sha256=self.dataset_sha,
            code_state=self.code_state,
            reranker=reranker,
            force_reindex=(
                bool(self.config.get("force_reindex", True))
                if is_new_index
                and not (
                    bool(self.config.get("resume"))
                    and complete_index_exists
                )
                else False
            ),
            reuse_source=None,
        )
        pipeline_manifest = _build_pipeline_manifest(
            content_manifest,
            key=key,
            pipeline=pipeline,
            reranker=reranker,
            index_fingerprint=index_fingerprint,
            index_dir=index_dir,
        )
        manifests_dir = self.run_dir / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifests_dir / f"{key}.json"
        manifest_path.write_text(
            json.dumps(
                pipeline_manifest,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        retriever = _load_eval_retriever(
            variant_settings,
            pipeline=pipeline,
            pipeline_key=key,
            index_dir=index_dir,
            manifest=pipeline_manifest,
            parser_artifact_sha256=self.artifact_sha,
            reranker=reranker,
        )
        retrieval_report = evaluate_retrieval_protocol(
            self.cases,
            retriever=retriever,
            repeat_count=int(self.config["latency"]["repeat_count"]),
            random_seed=int(self.config["latency"]["random_seed"]),
        )
        answer_smoke = evaluate_answer_smoke(
            self.answer_cases,
            retriever=retriever,
        )
        result = {
            "pipeline": asdict(pipeline),
            "pipeline_config_hash": pipeline.config_hash(),
            "index_contract": pipeline.index_contract(),
            "retrieval_contract": pipeline.retrieval_contract(),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "retrieval": retrieval_report,
            "answer_smoke": answer_smoke,
        }
        _write_json_atomic(
            checkpoint_path,
            {
                "schema_version": 1,
                "identity": checkpoint_identity,
                "result": result,
            },
        )
        return result


def run_from_config(config_path: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    config = _load_config(config_path, repo_root=repo_root)
    mode = str(config["mode"])
    if mode == "dev":
        return _run_dev(config_path, config=config, repo_root=repo_root)
    if mode == "final":
        return _run_final(config_path, config=config, repo_root=repo_root)
    raise ValueError(f"Unsupported M3.1 mode: {mode}.")


def _run_dev(
    config_path: Path,
    *,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    runtime = _prepare_runtime(
        config,
        repo_root=repo_root,
        dataset_path=Path(config["retrieval_dataset"]),
        expected_dataset_sha=str(config["retrieval_dataset_sha256"]),
    )
    run_dir = runtime.run_dir
    active_index_before = _active_index_snapshot(repo_root)
    if active_index_before != config["active_index_baseline"]:
        raise RuntimeError(
            "Active production index changed after the M3.1 preflight."
        )
    with exclusive_run_lock(run_dir):
        runtime.code_state = _capture_code_state(
            repo_root,
            run_dir=run_dir,
        )
        pipelines: dict[str, dict[str, Any]] = {}
        configs: dict[str, RetrievalPipelineConfig] = {}

        from indexing.retrieval_pipeline import get_pipeline_config

        for key in ("b0", "b1"):
            pipeline = get_pipeline_config(key)
            configs[key] = pipeline
            pipelines[key] = runtime.evaluate(key, pipeline)
        baseline = pipelines["b1"]["retrieval"]

        round1 = round1_seed_configs()
        for key, pipeline in round1.items():
            configs[key] = pipeline
            pipelines[key] = runtime.evaluate(key, pipeline)
        ranked_round1_seed = rank_candidates(
            {
                key: pipelines[key]["retrieval"]
                for key in round1
            },
            baseline=baseline,
        )
        if ranked_round1_seed:
            best_round1_key = ranked_round1_seed[0]
        else:
            best_round1_key = max(
                round1,
                key=lambda key: float(
                    pipelines[key]["retrieval"]["metrics"][
                        "recall_at_10"
                    ]
                ),
            )
        boost_key, boost_pipeline = round1_boost_off(
            best_round1_key,
            configs[best_round1_key],
        )
        configs[boost_key] = boost_pipeline
        pipelines[boost_key] = runtime.evaluate(
            boost_key,
            boost_pipeline,
        )
        round1_keys = [*round1, boost_key]
        ranked_round1 = rank_candidates(
            {
                key: pipelines[key]["retrieval"]
                for key in round1_keys
            },
            baseline=baseline,
        )
        if len(ranked_round1) < 2:
            return _stop_dev_early(
                runtime,
                config_path=config_path,
                pipelines=pipelines,
                candidate_keys=round1_keys,
                baseline=baseline,
                reason=(
                    "Round 1 has fewer than two candidates with "
                    "Recall@10 >= B1."
                ),
            )

        round2_bases = [
            (key, configs[key]) for key in ranked_round1[:2]
        ]
        round2 = round2_configs(round2_bases)
        for key, pipeline in round2.items():
            configs[key] = pipeline
            pipelines[key] = runtime.evaluate(key, pipeline)
        ranked_round2 = rank_candidates(
            {
                key: pipelines[key]["retrieval"]
                for key in round2
            },
            baseline=baseline,
        )
        if len(ranked_round2) < 2:
            return _stop_dev_early(
                runtime,
                config_path=config_path,
                pipelines=pipelines,
                candidate_keys=[*round1_keys, *round2],
                baseline=baseline,
                reason=(
                    "Round 2 has fewer than two candidates with "
                    "Recall@10 >= B1."
                ),
            )

        round3_bases = [
            (key, configs[key]) for key in ranked_round2[:2]
        ]
        round3_blends = round3_blend_configs(round3_bases)
        for key, pipeline in round3_blends.items():
            configs[key] = pipeline
            pipelines[key] = runtime.evaluate(key, pipeline)
        ranked_blends = rank_candidates(
            {
                key: pipelines[key]["retrieval"]
                for key in round3_blends
            },
            baseline=baseline,
        )
        if not ranked_blends:
            return _stop_dev_early(
                runtime,
                config_path=config_path,
                pipelines=pipelines,
                candidate_keys=[
                    *round1_keys,
                    *round2,
                    *round3_blends,
                ],
                baseline=baseline,
                reason=(
                    "Round 3 has no blended candidate with "
                    "Recall@10 >= B1."
                ),
            )
        round3_stability = round3_stability_configs(
            configs[ranked_blends[0]]
        )
        for key, pipeline in round3_stability.items():
            configs[key] = pipeline
            pipelines[key] = runtime.evaluate(key, pipeline)
        round3_keys = [*round3_blends, *round3_stability]
        ranked_round3 = rank_candidates(
            {
                key: pipelines[key]["retrieval"]
                for key in round3_keys
            },
            baseline=baseline,
        )

        candidate_keys = [
            *round1_keys,
            *round2,
            *round3_keys,
        ]
        if len(candidate_keys) != 24 or len(set(candidate_keys)) != 24:
            raise RuntimeError("M3.1 dev matrix must contain exactly 24 candidates.")
        report = {
            "schema_version": 3,
            "mode": "m3_1_dev",
            "generated_at": datetime.now(UTC).isoformat(),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "parser_artifact_sha256": runtime.artifact_sha,
            "retrieval_dataset": str(config["retrieval_dataset"]),
            "retrieval_dataset_sha256": runtime.dataset_sha,
            "holdout_dataset": str(config["holdout_dataset"]),
            "holdout_dataset_sha256": str(
                config["holdout_dataset_sha256"]
            ),
            "holdout_quality_evaluated": False,
            "embedding": _embedding_contract(runtime.settings),
            "code": runtime.code_state,
            "active_index_before": active_index_before,
            "active_index_after": _active_index_snapshot(repo_root),
            "active_index_changed": (
                _active_index_snapshot(repo_root) != active_index_before
            ),
            "latency_protocol": dict(config["latency"]),
            "candidate_budget": 24,
            "candidate_count": len(candidate_keys),
            "rounds": {
                "round1": {
                    "keys": round1_keys,
                    "ranking": ranked_round1,
                    "promoted": ranked_round1[:2],
                },
                "round2": {
                    "keys": list(round2),
                    "ranking": ranked_round2,
                    "promoted": ranked_round2[:2],
                },
                "round3": {
                    "keys": round3_keys,
                    "ranking": ranked_round3,
                },
            },
            "pareto_frontier": pareto_frontier(
                {
                    key: pipelines[key]["retrieval"]
                    for key in candidate_keys
                },
                baseline=baseline,
            ),
            "pipelines": pipelines,
        }
        _write_report(run_dir, report)
    _print_summary(report, run_dir=run_dir)
    return report


def _run_final(
    config_path: Path,
    *,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    final_run_dir = _prepare_run_dir(
        repo_root,
        output_dir=Path(config["output_dir"]),
        run_name=str(config["run_name"]),
    )
    report_path = final_run_dir / "report.json"
    if report_path.exists():
        raise RuntimeError(
            "The formal M3.1 holdout was already evaluated; refusing to rerun it."
        )
    finalist, pipeline = _load_frozen_finalist(config)
    selection_path = Path(str(config["selection_manifest"]))
    if sha256_file(selection_path) != str(
        config["selection_manifest_sha256"]
    ).casefold():
        raise ValueError("Selection manifest does not match frozen SHA-256.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("finalist", {}).get("config_sha256") != (
        finalist["config_sha256"]
    ):
        raise ValueError("Final config does not match the selection manifest.")
    freeze_path = selection_path.parent / "freeze" / "manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("final_config_sha256") != sha256_file(config_path):
        raise ValueError("Final YAML does not match the frozen manifest.")

    from indexing.retrieval_pipeline import get_pipeline_config

    active_index_before = _active_index_snapshot(repo_root)
    if active_index_before != config["active_index_baseline"]:
        raise RuntimeError(
            "Active production index changed after the M3.1 preflight."
        )
    final_run_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_run_lock(final_run_dir):
        code_state = _capture_code_state(repo_root, run_dir=final_run_dir)
        frozen_code = dict(freeze.get("code") or {})
        if any(
            code_state.get(key) != frozen_code.get(key)
            for key in (
                "commit",
                "dirty",
                "working_tree_patch_sha256",
            )
        ):
            raise RuntimeError(
                "Code commit or working-tree patch changed after finalist freeze."
            )
        dataset_reports: dict[str, Any] = {}
        dataset_specs = (
            (
                "holdout",
                Path(config["holdout_dataset"]),
                str(config["holdout_dataset_sha256"]),
            ),
            (
                "dev",
                Path(config["retrieval_dataset"]),
                str(config["retrieval_dataset_sha256"]),
            ),
        )
        for role, dataset_path, dataset_sha in dataset_specs:
            runtime = _prepare_runtime(
                config,
                repo_root=repo_root,
                dataset_path=dataset_path,
                expected_dataset_sha=dataset_sha,
                run_dir_override=final_run_dir / role,
            )
            runtime.code_state = code_state
            b1 = runtime.evaluate("b1", get_pipeline_config("b1"))
            b2_1 = runtime.evaluate("b2_1", pipeline)
            gate = final_gate(
                b2_1["retrieval"],
                baseline=b1["retrieval"],
            )
            dataset_reports[role] = {
                "dataset": str(dataset_path),
                "dataset_sha256": dataset_sha,
                "gate": gate,
                "pipelines": {"b1": b1, "b2_1": b2_1},
            }

        passed = all(
            row["gate"]["passed"] for row in dataset_reports.values()
        )
        prefix_leak_count = sum(
            pipeline_report["answer_smoke"]["metadata_prefix_leak_count"]
            for dataset in dataset_reports.values()
            for pipeline_report in dataset["pipelines"].values()
        )
        passed = passed and prefix_leak_count == 0
        active_index_after = _active_index_snapshot(repo_root)
        active_index_changed = active_index_after != active_index_before
        passed = passed and not active_index_changed
        report = {
            "schema_version": 3,
            "mode": "m3_1_final",
            "generated_at": datetime.now(UTC).isoformat(),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "selection_manifest": str(selection_path),
            "selection_manifest_sha256": str(
                config["selection_manifest_sha256"]
            ),
            "parser_artifact_sha256": str(
                config["parser_artifact_sha256"]
            ),
            "retrieval_dataset_sha256": str(
                config["retrieval_dataset_sha256"]
            ),
            "holdout_dataset_sha256": str(
                config["holdout_dataset_sha256"]
            ),
            "holdout_quality_evaluated": True,
            "formal_holdout_run_count": 1,
            "finalist": finalist,
            "embedding": dict(config["embedding"]),
            "code": code_state,
            "latency_protocol": dict(config["latency"]),
            "datasets": dataset_reports,
            "metadata_prefix_leak_count": prefix_leak_count,
            "active_index_before": active_index_before,
            "active_index_after": active_index_after,
            "active_index_changed": active_index_changed,
            "core_passed": passed,
            "default_pipeline": (
                "v2_fixed_optimized" if passed else "v1_flat_rerank"
            ),
            "m4_entry_ready": passed,
        }
        _write_report(final_run_dir, report)
    _print_summary(report, run_dir=final_run_dir)
    return report


def _prepare_runtime(
    config: dict[str, Any],
    *,
    repo_root: Path,
    dataset_path: Path,
    expected_dataset_sha: str,
    run_dir_override: Path | None = None,
) -> _ExperimentRuntime:
    settings = load_settings(base_dir=repo_root)
    settings = replace(
        settings,
        flashrank_cache_dir=str(config["reranker"]["cache_dir"]),
    )
    _validate_frozen_runtime(settings, config)
    artifact, artifact_sha = load_parser_artifact(
        Path(config["parser_artifact"]),
        expected_sha256=str(config["parser_artifact_sha256"]),
        corpus_dir=Path(config["corpus_dir"]),
    )
    actual_dataset_sha = sha256_file(dataset_path)
    if actual_dataset_sha != expected_dataset_sha.casefold():
        raise ValueError("Retrieval dataset does not match frozen SHA-256.")
    actual_holdout_sha = sha256_file(Path(config["holdout_dataset"]))
    if actual_holdout_sha != str(config["holdout_dataset_sha256"]).casefold():
        raise ValueError("Holdout dataset does not match frozen SHA-256.")
    cases = load_retrieval_cases(dataset_path, artifact=artifact)
    answer_cases = load_answer_smoke_cases(
        Path(config["answer_smoke_dataset"])
    )
    run_dir = run_dir_override or _prepare_run_dir(
        repo_root,
        output_dir=Path(config["output_dir"]),
        run_name=str(config["run_name"]),
    )
    artifacts_root = (repo_root / "artifacts").resolve()
    if not run_dir.resolve().is_relative_to(artifacts_root):
        raise ValueError("M3.1 runtime directory must stay under artifacts.")
    run_dir.mkdir(parents=True, exist_ok=True)
    return _ExperimentRuntime(
        repo_root=repo_root,
        run_dir=run_dir,
        settings=settings,
        config=config,
        documents=artifact_documents(artifact),
        cases=cases,
        answer_cases=answer_cases,
        artifact=artifact,
        artifact_sha=artifact_sha,
        dataset_sha=actual_dataset_sha,
        code_state={},
        index_dirs={},
    )


def _load_frozen_finalist(
    config: dict[str, Any],
) -> tuple[dict[str, Any], RetrievalPipelineConfig]:
    finalist_payload = config.get("finalists")
    if not isinstance(finalist_payload, list) or len(finalist_payload) != 1:
        raise ValueError("M3.1 final accepts exactly one frozen finalist.")
    finalist = dict(finalist_payload[0])
    pipeline = pipeline_from_dict(dict(finalist["config"]))
    if pipeline.name != "v2_fixed_optimized":
        raise ValueError("Frozen finalist must be named v2_fixed_optimized.")
    if pipeline.config_hash() != finalist.get("config_sha256"):
        raise ValueError("Frozen finalist config SHA-256 does not match.")
    return finalist, pipeline


def _build_pipeline_manifest(
    content_manifest: dict[str, Any],
    *,
    key: str,
    pipeline: RetrievalPipelineConfig,
    reranker: dict[str, Any],
    index_fingerprint: str,
    index_dir: Path,
) -> dict[str, Any]:
    return {
        **content_manifest,
        "kind": "v2-m3.1-eval-pipeline",
        "pipeline_key": key,
        "pipeline": asdict(pipeline),
        "pipeline_config_hash": pipeline.config_hash(),
        "retrieval": pipeline.index_contract(),
        "retrieval_contract": pipeline.retrieval_contract(),
        "reranker": reranker,
        "content_index_contract_sha256": index_fingerprint,
        "content_index_dir": str(index_dir),
    }


def _checkpoint_identity(
    *,
    key: str,
    pipeline: RetrievalPipelineConfig,
    artifact_sha: str,
    dataset_sha: str,
    embedding: dict[str, Any],
    code_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pipeline_key": key,
        "pipeline_config_hash": pipeline.config_hash(),
        "parser_artifact_sha256": artifact_sha,
        "retrieval_dataset_sha256": dataset_sha,
        "embedding": embedding,
        "code": {
            name: code_state.get(name)
            for name in (
                "commit",
                "dirty",
                "working_tree_patch_sha256",
            )
        },
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _prepare_content_addressed_settings(
    settings: AppSettings,
    *,
    run_dir: Path,
    index_fingerprint: str,
) -> tuple[AppSettings, Path]:
    index_dir = run_dir / "indexes" / "by_contract" / index_fingerprint
    prepared = replace(
        settings,
        data_dir=run_dir,
        index_dir=index_dir,
        faiss_dir=index_dir / "faiss",
        bm25_path=index_dir / "bm25.pkl",
        nodes_path=index_dir / "nodes.jsonl",
        doc_trees_path=index_dir / "doc_trees.json",
        retriever_k=8,
        flashrank_top_n=30,
        max_context_tokens=8000,
        offline_mode=False,
    )
    return prepared, index_dir


def _load_config(path: Path, *, repo_root: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("Unsupported M3.1 eval config schema.")
    for key in (
        "output_dir",
        "corpus_dir",
        "parser_artifact",
        "parser_gold",
        "retrieval_dataset",
        "holdout_dataset",
        "answer_smoke_dataset",
    ):
        candidate = Path(str(payload[key]))
        payload[key] = str(
            candidate
            if candidate.is_absolute()
            else (repo_root / candidate).resolve()
        )
    if "selection_manifest" in payload:
        selection_path = Path(str(payload["selection_manifest"]))
        payload["selection_manifest"] = str(
            selection_path
            if selection_path.is_absolute()
            else (repo_root / selection_path).resolve()
        )
    cache_dir = Path(str(payload["reranker"]["cache_dir"]))
    payload["reranker"]["cache_dir"] = str(
        cache_dir
        if cache_dir.is_absolute()
        else (repo_root / cache_dir).resolve()
    )
    latency = dict(payload["latency"])
    if latency != {
        "warmup_count": 1,
        "repeat_count": 5,
        "random_seed": 31,
    }:
        raise ValueError("M3.1 latency protocol must remain frozen.")
    if (
        payload["retrieval_dataset_sha256"]
        == payload["holdout_dataset_sha256"]
    ):
        raise ValueError("Old dev and new holdout must have distinct SHAs.")
    artifacts_root = (repo_root / "artifacts").resolve()
    if not Path(payload["reranker"]["cache_dir"]).resolve().is_relative_to(
        artifacts_root
    ):
        raise ValueError("M3.1 model cache must stay under artifacts.")
    return payload


def _stop_dev_early(
    runtime: _ExperimentRuntime,
    *,
    config_path: Path,
    pipelines: dict[str, dict[str, Any]],
    candidate_keys: list[str],
    baseline: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    report = {
        "schema_version": 3,
        "mode": "m3_1_dev",
        "status": "failed",
        "failure_reason": reason,
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "parser_artifact_sha256": runtime.artifact_sha,
        "retrieval_dataset": str(runtime.config["retrieval_dataset"]),
        "retrieval_dataset_sha256": runtime.dataset_sha,
        "holdout_dataset": str(runtime.config["holdout_dataset"]),
        "holdout_dataset_sha256": str(
            runtime.config["holdout_dataset_sha256"]
        ),
        "holdout_quality_evaluated": False,
        "embedding": _embedding_contract(runtime.settings),
        "code": runtime.code_state,
        "latency_protocol": dict(runtime.config["latency"]),
        "candidate_budget": 24,
        "candidate_count": len(candidate_keys),
        "rounds": {},
        "pareto_frontier": pareto_frontier(
            {
                key: pipelines[key]["retrieval"]
                for key in candidate_keys
            },
            baseline=baseline,
        ),
        "pipelines": pipelines,
        "default_pipeline": "v1_flat_rerank",
        "m4_entry_ready": False,
        "active_index_before": runtime.config["active_index_baseline"],
        "active_index_after": _active_index_snapshot(runtime.repo_root),
        "active_index_changed": (
            _active_index_snapshot(runtime.repo_root)
            != runtime.config["active_index_baseline"]
        ),
    }
    _write_report(runtime.run_dir, report)
    _print_summary(report, run_dir=runtime.run_dir)
    return report


def _write_report(run_dir: Path, report: dict[str, Any]) -> None:
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _print_summary(report: dict[str, Any], *, run_dir: Path) -> None:
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "report": str(run_dir / "report.json"),
                "candidate_count": report.get("candidate_count"),
                "holdout_quality_evaluated": report.get(
                    "holdout_quality_evaluated"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _active_index_snapshot(repo_root: Path) -> dict[str, Any]:
    active_path = repo_root / "data" / "indexes" / "active.json"
    active_file = (
        json.loads(active_path.read_text(encoding="utf-8"))
        if active_path.exists()
        else None
    )
    database_path = repo_root / "data" / "api" / "sessions.db"
    database_value: Any = None
    app_state_exists = False
    if database_path.exists():
        with sqlite3.connect(database_path) as connection:
            app_state_exists = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'app_state'
                """
            ).fetchone() is not None
            if app_state_exists:
                row = connection.execute(
                    """
                    SELECT value_json FROM app_state
                    WHERE key = 'active_index_version'
                    """
                ).fetchone()
                if row is not None:
                    database_value = json.loads(row[0])
    return {
        "active_json": active_file,
        "sqlite_app_state_exists": app_state_exists,
        "sqlite_active_index_version": database_value,
    }
