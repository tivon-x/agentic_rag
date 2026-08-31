"""M6 KITE prepare-contract regression tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.kite import (
    KITE_CASE_COUNT,
    KITE_COMMIT,
    KITE_CORPUS_FILE_COUNT,
    KITE_CORPUS_SHA256,
    KITE_EMPTY_RUBRIC_COUNT,
    KITE_QUERY_SHA256,
    KITE_REPOSITORY,
    KiteDataError,
    build_corpus_manifest,
    build_kite_manifest,
    load_kite_cases,
    validate_pdf_file,
)
from evals import kite_runner
from indexing.retrieval_pipeline import get_pipeline_config


def _write_query(path: Path, cases: list[dict[str, object]]) -> str:
    path.write_text(json.dumps(cases) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(rubric: str = "rubric") -> dict[str, object]:
    return {"query": "question", "gt_answer": "answer", "rubric": rubric}


def test_kite_cases_preserve_order_and_allow_empty_rubric(tmp_path: Path) -> None:
    path = tmp_path / "queries.json"
    cases = [_case(""), _case("check")]
    digest = _write_query(path, cases)

    loaded = load_kite_cases(
        path,
        expected_sha256=digest,
        expected_case_count=2,
        expected_empty_rubric_count=1,
    )

    assert [case.id for case in loaded] == ["ai-papers-001", "ai-papers-002"]
    assert [case.source_index for case in loaded] == [0, 1]
    assert loaded[0].rubric == ""


@pytest.mark.parametrize("missing", ["query", "gt_answer", "rubric"])
def test_kite_missing_required_field_fails(tmp_path: Path, missing: str) -> None:
    payload = _case()
    del payload[missing]
    path = tmp_path / "queries.json"
    digest = _write_query(path, [payload])

    with pytest.raises(KiteDataError, match=missing):
        load_kite_cases(
            path,
            expected_sha256=digest,
            expected_case_count=1,
            expected_empty_rubric_count=0,
        )


def test_kite_hash_and_duplicate_id_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "queries.json"
    digest = _write_query(path, [{**_case(), "id": "same"}, {**_case(), "id": "same"}])

    with pytest.raises(KiteDataError, match="SHA-256"):
        load_kite_cases(path, expected_sha256="0" * 64)
    with pytest.raises(KiteDataError, match="Duplicate"):
        load_kite_cases(
            path,
            expected_sha256=digest,
            expected_case_count=2,
            expected_empty_rubric_count=0,
        )


def test_kite_corpus_rejects_lfs_pointer_missing_and_empty(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    pointer = corpus / "pointer.pdf"
    pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(KiteDataError, match="git lfs pull"):
        validate_pdf_file(pointer)

    pointer.unlink()
    empty = corpus / "empty.pdf"
    empty.touch()
    with pytest.raises(KiteDataError, match="empty"):
        validate_pdf_file(empty)

    with pytest.raises(KiteDataError, match="does not exist"):
        validate_pdf_file(corpus / "missing.pdf")


def test_kite_corpus_manifest_is_sorted_and_hash_stable(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "nested").mkdir(parents=True)
    (corpus / "z.pdf").write_bytes(b"%PDF-1.7\nz")
    (corpus / "nested" / "a.pdf").write_bytes(b"%PDF-1.7\na")

    first, first_hash = build_corpus_manifest(corpus, expected_file_count=2)
    second, second_hash = build_corpus_manifest(corpus, expected_file_count=2)

    assert [row["file_name"] for row in first] == ["nested/a.pdf", "z.pdf"]
    assert first == second
    assert first_hash == second_hash


def test_kite_manifest_uses_logical_paths(tmp_path: Path) -> None:
    query_path = tmp_path / "queries.json"
    query_digest = _write_query(query_path, [_case()])
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "paper.pdf").write_bytes(b"%PDF-1.7\npaper")

    manifest = build_kite_manifest(
        query_path=query_path,
        corpus_root=corpus,
        upstream_commit=KITE_COMMIT,
        expected_query_sha256=query_digest,
        expected_case_count=1,
        expected_empty_rubric_count=0,
        expected_corpus_file_count=1,
        expected_corpus_sha256=build_corpus_manifest(corpus, expected_file_count=1)[1],
    )

    assert manifest["query_path"] == "queries/ai_papers.json"
    assert manifest["corpus_root"] == "knowledge-base-content/ai-papers"
    assert manifest["case_count"] == 1
    assert manifest["corpus_file_count"] == 1
    assert str(tmp_path).casefold() not in json.dumps(manifest).casefold()


def test_frozen_counts_are_explicit() -> None:
    assert KITE_CASE_COUNT == 15
    assert KITE_EMPTY_RUBRIC_COUNT == 6
    assert KITE_CORPUS_FILE_COUNT == 134
    assert KITE_CORPUS_SHA256 == "f33a3154a0a65d76dbfd10e599a7c5d640ac025ebadb76d80e2a5536c57240c8"


def test_kite_manifest_rejects_wrong_corpus_hash(tmp_path: Path) -> None:
    query_path = tmp_path / "queries.json"
    query_digest = _write_query(query_path, [_case()])
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "paper.pdf").write_bytes(b"%PDF-1.7\npaper")

    with pytest.raises(KiteDataError, match="corpus SHA-256"):
        build_kite_manifest(
            query_path=query_path,
            corpus_root=corpus,
            expected_query_sha256=query_digest,
            expected_case_count=1,
            expected_empty_rubric_count=0,
            expected_corpus_file_count=1,
            expected_corpus_sha256="0" * 64,
        )


def test_kite_prepare_does_not_overwrite_an_existing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_path = tmp_path / "queries.json"
    query_digest = _write_query(query_path, [_case()])
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "paper.pdf").write_bytes(b"%PDF-1.7\npaper")
    corpus_digest = build_corpus_manifest(corpus, expected_file_count=1)[1]
    manifest_path = tmp_path / "manifest.json"
    config_path = tmp_path / "kite.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": {
                    "name": "kite-ai-papers",
                    "query_path": str(query_path),
                    "corpus_root": str(corpus),
                    "query_sha256": query_digest,
                    "case_count": 1,
                    "empty_rubric_count": 0,
                    "corpus_file_count": 1,
                    "corpus_sha256": corpus_digest,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(kite_runner, "_resolve_output_path", lambda *_: manifest_path)

    kite_runner.prepare_from_config(config_path)
    first = manifest_path.read_bytes()
    kite_runner.prepare_from_config(config_path)

    assert manifest_path.read_bytes() == first


def test_kite_config_rejects_non_fixed_generation(tmp_path: Path) -> None:
    config_path = tmp_path / "kite.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": {
                    "name": "kite-ai-papers",
                    "query_path": "query.json",
                    "corpus_root": "corpus",
                },
                "generation": {"strategy": "adaptive"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KiteDataError, match="must be fixed"):
        kite_runner._load_config(config_path)


def test_kite_judge_accepts_only_a_single_integer() -> None:
    assert kite_runner._parse_judge_score("0") == 0
    assert kite_runner._parse_judge_score(" 10\n") == 10
    assert kite_runner._parse_judge_score("1.0") is None
    assert kite_runner._parse_judge_score("score: 9") is None
    assert kite_runner._parse_judge_score("11") is None


def test_kite_judge_retries_one_invalid_response(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(["not a score", SimpleNamespace(content="7")])
    updates: list[dict[str, object]] = []

    class FakeJudge:
        def model_copy(self, *, update: dict[str, object]) -> "FakeJudge":
            updates.append(update)
            return self

        def invoke(self, _: object) -> object:
            return next(responses)

    monkeypatch.setattr(kite_runner, "get_llm_by_type", lambda _: FakeJudge())
    score, attempts, error = kite_runner._judge_answer(
        kite_runner.KiteCase("case", "q", "a", "", 0),
        "candidate",
        {"task_type": "kite_judge", "model": "judge", "timeout_seconds": 60},
    )

    assert (score, attempts, error) == (7, 2, None)
    assert updates == [{"temperature": 0, "request_timeout": 60.0}]


def test_kite_judge_returns_failure_after_two_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingJudge:
        def model_copy(self, *, update: dict[str, object]) -> "FailingJudge":
            return self

        def invoke(self, _: object) -> object:
            raise TimeoutError("timed out")

    monkeypatch.setattr(kite_runner, "get_llm_by_type", lambda _: FailingJudge())
    score, attempts, error = kite_runner._judge_answer(
        kite_runner.KiteCase("case", "q", "a", "", 0),
        "candidate",
        {"task_type": "kite_judge", "model": "judge", "timeout_seconds": 60},
    )

    assert score is None
    assert attempts == 2
    assert error and "TimeoutError" in error


def test_kite_evidence_uses_retrieval_owned_metadata() -> None:
    evidence = kite_runner._public_evidence(
        [
            {
                "node_id": "node-1",
                "source": "model.pdf",
                "page": None,
                "quote": "model text",
            }
        ],
        evidence_lookup={
            "node-1": {
                "doc_id": "canonical-doc",
                "paper_id": "paper-1",
                "paper_title": "Paper title",
                "source": "source.pdf",
                "section_path": ["Methods"],
                "page_start": 4,
                "quote_text": "source-faithful text",
            }
        },
        retrieved_evidence={
            "node-1": {
                "node_id": "node-1",
                "doc_id": "retrieved-doc",
                "score": 0.8,
                "relevance": None,
            }
        },
    )

    assert evidence == [
        {
            "doc_id": "canonical-doc",
            "node_id": "node-1",
            "paper_id": "paper-1",
            "paper_title": "Paper title",
            "source": "source.pdf",
            "section_path": ["Methods"],
            "page": 4,
            "quote": "source-faithful text",
            "score": 0.8,
            "relevance": None,
        }
    ]


def test_kite_evidence_must_come_from_retrieved_parser_records() -> None:
    with pytest.raises(KiteDataError, match="not retrieved"):
        kite_runner._public_evidence(
            [{"node_id": "invented", "quote": "model text"}],
            evidence_lookup={"invented": {"quote_text": "canonical"}},
            retrieved_evidence={"retrieved": {"node_id": "retrieved"}},
        )

    with pytest.raises(KiteDataError, match="parser artifact"):
        kite_runner._public_evidence(
            [{"node_id": "retrieved", "quote": "model text"}],
            evidence_lookup={},
            retrieved_evidence={"retrieved": {"node_id": "retrieved"}},
        )


def test_kite_run_error_is_preserved() -> None:
    class FailingGraph:
        def invoke(self, *_: object, **__: object) -> object:
            raise RuntimeError("generation failed")

    row = kite_runner._run_case(
        FailingGraph(),
        kite_runner.KiteCase("case", "q", "a", "", 0),
        judge={"task_type": "kite_judge", "model": "judge"},
        thread_prefix="test",
        pipeline_name="b1",
        pipeline_config_hash="hash",
        evidence_lookup={},
    )

    assert row["judge_score"] is None
    assert row["run_error"] == "RuntimeError: generation failed"
    assert row["error"] == row["run_error"]


def test_kite_run_rejects_dirty_tree_before_provider_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "kite.yaml"
    config_path.write_text(
        """schema_version: 1
benchmark:
  name: kite-ai-papers
  query_path: query.json
  corpus_root: corpus
output:
  dir: artifacts/evals/kite/test
""",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    monkeypatch.setattr(kite_runner, "_resolve_output_path", lambda *_: run_dir)
    monkeypatch.setattr(
        kite_runner,
        "_capture_code_state",
        lambda *_args, **_kwargs: {
            "dirty": True,
            "working_tree_patch_path": "patch.diff",
        },
    )

    def fail_provider_setup(*_: object) -> object:
        raise AssertionError("provider setup must not run")

    monkeypatch.setattr(kite_runner, "_runtime_settings", fail_provider_setup)
    with pytest.raises(KiteDataError, match="clean working tree"):
        kite_runner.run_from_config(config_path)


def test_kite_pairwise_aggregation_keeps_case_ids_and_skips_invalid_scores() -> None:
    baseline = {
        "cases": [
            {"case_id": "ai-papers-001", "judge_score": 5},
            {"case_id": "ai-papers-002", "judge_score": 5},
            {"case_id": "ai-papers-003", "judge_score": None},
        ]
    }
    candidate = {
        "cases": [
            {"case_id": "ai-papers-001", "judge_score": 7},
            {"case_id": "ai-papers-002", "judge_score": 5},
            {"case_id": "ai-papers-003", "judge_score": 10},
        ]
    }

    assert kite_runner._pairwise_scores(baseline, candidate) == {
        "candidate_wins": 1,
        "ties": 1,
        "candidate_losses": 0,
        "win_case_ids": ["ai-papers-001"],
        "tie_case_ids": ["ai-papers-002"],
        "loss_case_ids": [],
    }


def _formal_reports() -> dict[str, dict[str, object]]:
    rows = [
        {
            "case_id": f"ai-papers-{index:03d}",
            "query": f"q{index}",
            "reference_answer": f"a{index}",
            "rubric": "",
            "judge_score": 5,
            "latency_ms": 10,
            "context": {"total_tokens": 20},
            "error": None,
            "evidence": [
                {
                    "source": "paper.pdf",
                    "quote": "source text",
                    "page": 1,
                    "section_path": ["Methods"],
                }
            ],
        }
        for index in range(1, 16)
    ]
    reports: dict[str, dict[str, object]] = {}
    repo_root = Path(__file__).resolve().parents[1]
    for key in kite_runner.PIPELINES:
        config_path = repo_root / "evals" / "configs" / f"kite_{key}.yaml"
        config = kite_runner._load_config(config_path)
        pipeline = get_pipeline_config(key)
        reports[key] = {
            "schema_version": 1,
            "formal_run": True,
            "benchmark": {
                "name": "kite-ai-papers",
                "upstream_repository": KITE_REPOSITORY,
                "upstream_commit": KITE_COMMIT,
                "query_sha256": KITE_QUERY_SHA256,
                "corpus_file_count": KITE_CORPUS_FILE_COUNT,
                "corpus_file_sha256": KITE_CORPUS_SHA256,
            },
            "pipeline": {
                "key": key,
                "config": asdict(pipeline),
                "config_hash": pipeline.config_hash(),
            },
            "generation": dict(config["generation"]),
            "judge": {
                "task_type": config["judge"]["task_type"],
                "model": config["judge"]["model"],
                "prompt_version": kite_runner.JUDGE_PROMPT_VERSION,
                "temperature": 0,
                "timeout_seconds": 60.0,
            },
            "embedding": dict(config["embedding"]),
            "reranker": dict(config["reranker"]),
            "runtime": dict(config["runtime"]),
            "provenance": {
                "config_path": f"evals/configs/kite_{key}.yaml",
                "config_sha256": kite_runner.sha256_file(config_path),
                "manifest_sha256": "manifest",
                "parser_artifact_sha256": "parser",
                "code": {"commit": "commit", "dirty": False, "config_path": f"{key}.yaml"},
            },
            "metrics": {
                "case_count": 15,
                "valid_count": 15,
                "mean_score": 5,
                "p95_latency_ms": 10,
                "mean_context_tokens": 20,
            },
            "cases": json.loads(json.dumps(rows)),
        }
    return reports


def test_kite_report_validation_recomputes_and_checks_comparability() -> None:
    reports = _formal_reports()
    repo_root = Path(__file__).resolve().parents[1]
    kite_runner._validate_reports(reports, repo_root)

    reports["b2"]["metrics"]["mean_score"] = 9  # type: ignore[index]
    with pytest.raises(KiteDataError, match="metrics"):
        kite_runner._validate_reports(reports, repo_root)

    reports = _formal_reports()
    reports["b3"]["generation"] = {"model": "different"}
    with pytest.raises(KiteDataError, match="frozen config"):
        kite_runner._validate_reports(reports, repo_root)


def test_kite_report_validation_binds_provenance_to_current_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = tmp_path / "kite"
    runs_dir.mkdir()
    manifest = json.loads(
        (repo_root / "artifacts" / "evals" / "kite" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_path = runs_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    artifact_path = runs_dir / "parser_artifact.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "kite-parser-artifact",
                "corpus_manifest": manifest["corpus_manifest"],
            }
        ),
        encoding="utf-8",
    )
    reports = _formal_reports()
    head = kite_runner.subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    real_run = kite_runner.subprocess.run

    def clean_status(command: list[str], **kwargs: object) -> object:
        if command[:2] == ["git", "status"]:
            return SimpleNamespace(stdout="")
        return real_run(command, **kwargs)

    monkeypatch.setattr(kite_runner.subprocess, "run", clean_status)
    for key, report in reports.items():
        index_path = runs_dir / "indexes" / key / "manifest.json"
        index_path.parent.mkdir(parents=True)
        index_path.write_text(
            json.dumps(
                {
                    "pipeline_key": key,
                    "pipeline_config_hash": report["pipeline"]["config_hash"],  # type: ignore[index]
                    "parser_artifact_sha256": kite_runner.sha256_file(artifact_path),
                    "embedding": report["embedding"],
                    "reranker": report["reranker"],
                }
            ),
            encoding="utf-8",
        )
        provenance = report["provenance"]  # type: ignore[assignment]
        provenance["manifest_sha256"] = kite_runner.sha256_file(manifest_path)
        provenance["parser_artifact_sha256"] = kite_runner.sha256_file(artifact_path)
        provenance["index_manifest_sha256"] = kite_runner.sha256_file(index_path)
        provenance["code"] = {"commit": head, "dirty": False}

    kite_runner._validate_reports(reports, repo_root, runs_dir)
    reports["b2"]["provenance"]["index_manifest_sha256"] = "evil"  # type: ignore[index]
    with pytest.raises(KiteDataError, match="current files"):
        kite_runner._validate_reports(reports, repo_root, runs_dir)

    reports = _formal_reports()
    monkeypatch.setattr(
        kite_runner.subprocess,
        "run",
        lambda command, **kwargs: (
            SimpleNamespace(stdout=" M evals/kite_runner.py\n")
            if command[:2] == ["git", "status"]
            else real_run(command, **kwargs)
        ),
    )
    with pytest.raises(KiteDataError, match="clean working tree"):
        kite_runner._validate_provenance_files(reports, repo_root, runs_dir)

    reports = _formal_reports()
    reports["b0"]["formal_run"] = False
    with pytest.raises(KiteDataError, match="not a formal"):
        kite_runner._validate_reports(reports, repo_root)

    reports = _formal_reports()
    for report in reports.values():
        report["benchmark"]["query_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(KiteDataError, match="frozen config"):
        kite_runner._validate_reports(reports, repo_root)
