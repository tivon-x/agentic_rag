"""Freeze the single M3.1 finalist selected from the dev experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import yaml

from evals.m3_1_experiments import dev_gate, rank_candidates
from evals.v2_corpus import sha256_file
from evals.v2_runner import _capture_code_state
from indexing.retrieval_pipeline import RetrievalPipelineConfig


SELECTION_METHOD = "frozen_lexicographic_no_composite_score"


def select_candidate(
    run_dir: Path,
    *,
    max_finalists: int = 1,
) -> dict[str, Any]:
    if max_finalists != 1:
        raise ValueError("M3.1 permits exactly one frozen finalist.")
    repo_root = Path(__file__).resolve().parent.parent
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _validate_dev_report(report)

    baseline = report["pipelines"]["b1"]["retrieval"]
    candidate_keys = [
        key
        for round_report in report["rounds"].values()
        for key in round_report["keys"]
    ]
    gates = {
        key: dev_gate(
            report["pipelines"][key]["retrieval"],
            baseline=baseline,
        )
        for key in candidate_keys
    }
    passing = {
        key: report["pipelines"][key]["retrieval"]
        for key in candidate_keys
        if gates[key]["passed"]
    }
    ranking = rank_candidates(passing, baseline=baseline)
    selection_path = run_dir / "selection.json"
    if not ranking:
        selection = {
            "schema_version": 1,
            "status": "failed",
            "reason": "No candidate passed the frozen dev promotion gate.",
            "default_pipeline": "v1_flat_rerank",
            "m4_entry_ready": False,
            "holdout_quality_evaluated": False,
            "dev_report": str(report_path),
            "dev_report_sha256": sha256_file(report_path),
            "gates": gates,
            "pareto_frontier": report["pareto_frontier"],
        }
        _write_json(selection_path, selection)
        _print_result(selection_path, selection)
        return selection

    source_key = ranking[0]
    source_pipeline = RetrievalPipelineConfig(
        **_normalize_pipeline_payload(
            report["pipelines"][source_key]["pipeline"]
        )
    )
    finalist_pipeline = replace(
        source_pipeline,
        name="v2_fixed_optimized",
    )
    finalist_config = _jsonable_pipeline(finalist_pipeline)
    finalist = {
        "source_key": source_key,
        "config": finalist_config,
        "config_sha256": finalist_pipeline.config_hash(),
        "dev_gate": gates[source_key],
    }
    selection = {
        "schema_version": 1,
        "status": "selected",
        "selection_method": SELECTION_METHOD,
        "max_finalists": max_finalists,
        "holdout_quality_evaluated": False,
        "dev_report": str(report_path),
        "dev_report_sha256": sha256_file(report_path),
        "parser_artifact_sha256": report["parser_artifact_sha256"],
        "retrieval_dataset_sha256": report[
            "retrieval_dataset_sha256"
        ],
        "holdout_dataset_sha256": report["holdout_dataset_sha256"],
        "ranking": ranking,
        "gates": gates,
        "finalist": finalist,
    }
    _write_json(selection_path, selection)

    dev_config_path = Path(report["config_path"])
    dev_config = yaml.safe_load(dev_config_path.read_text(encoding="utf-8"))
    final_config_path = repo_root / "evals" / "configs" / "v2_m3_1_final.yaml"
    final_config = {
        **dev_config,
        "mode": "final",
        "run_name": "final",
        "selection_manifest": str(
            selection_path.resolve().relative_to(repo_root)
        ).replace("\\", "/"),
        "selection_manifest_sha256": sha256_file(selection_path),
        "finalists": [finalist],
    }
    final_config_path.write_text(
        yaml.safe_dump(
            final_config,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    freeze_dir = run_dir / "freeze"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    code_state = _capture_code_state(repo_root, run_dir=freeze_dir)
    freeze = {
        "schema_version": 1,
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "final_config": str(final_config_path),
        "final_config_sha256": sha256_file(final_config_path),
        "finalist_config_sha256": finalist["config_sha256"],
        "code": code_state,
    }
    _write_json(freeze_dir / "manifest.json", freeze)
    selection["freeze_manifest"] = str(freeze_dir / "manifest.json")
    _print_result(selection_path, selection)
    return selection


def _validate_dev_report(report: dict[str, Any]) -> None:
    if report.get("mode") != "m3_1_dev":
        raise ValueError("Candidate selection requires an M3.1 dev report.")
    if report.get("candidate_count") != 24:
        raise ValueError("Candidate selection requires exactly 24 candidates.")
    if report.get("holdout_quality_evaluated") is not False:
        raise ValueError("Dev selection must not use holdout quality.")
    candidate_keys = [
        key
        for round_report in report["rounds"].values()
        for key in round_report["keys"]
    ]
    if len(candidate_keys) != 24 or len(set(candidate_keys)) != 24:
        raise ValueError("Dev report candidate matrix is not frozen at 24.")


def _normalize_pipeline_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["metadata_prefix_fields"] = tuple(
        normalized["metadata_prefix_fields"]
    )
    return normalized


def _jsonable_pipeline(
    pipeline: RetrievalPipelineConfig,
) -> dict[str, Any]:
    from dataclasses import asdict

    return json.loads(json.dumps(asdict(pipeline)))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _print_result(path: Path, selection: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "status": selection["status"],
                "selection": str(path),
                "finalist": selection.get("finalist", {}).get("source_key"),
                "holdout_quality_evaluated": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--max-finalists", type=int, default=1)
    args = parser.parse_args()
    select_candidate(
        Path(args.run).resolve(),
        max_finalists=args.max_finalists,
    )


if __name__ == "__main__":
    main()
