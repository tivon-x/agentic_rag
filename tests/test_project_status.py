"""Guard the repository's current milestone facts against documentation drift."""

import hashlib
import json
import subprocess
from dataclasses import fields
from pathlib import Path

import yaml

from core.settings import AppSettings
from indexing.retrieval_pipeline import get_pipeline_config


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "docs" / "research" / "v2_upgrade_plan.md"
BASELINE_PATH = ROOT / "artifacts" / "evals" / "v2_m3_2" / "m4_fixed_baseline.json"
KITE_MANIFEST_PATH = ROOT / "artifacts" / "evals" / "kite" / "manifest.json"
KITE_SUMMARY_PATH = ROOT / "artifacts" / "evals" / "kite" / "summary.json"
KITE_REPORT_PATHS = {
    key: ROOT / "artifacts" / "evals" / "kite" / key / "report.json"
    for key in ("b0", "b1", "b2", "b3")
}
SUPERSEDED_DOCS = (
    ROOT / "docs" / "research" / "phase2_goal_prompts.md",
    ROOT / "docs" / "research" / "m6_evaluation_lab_implementation_plan.md",
    ROOT / "tasks.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_included(path: Path) -> None:
    relative_path = path.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", relative_path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert relative_path in result.stdout.splitlines(), f"evidence is ignored: {relative_path}"


def _read_project_status() -> dict[str, object]:
    text = PLAN_PATH.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "v2 plan must start with YAML project status"
    _, frontmatter, _ = text.split("---", maxsplit=2)
    status = yaml.safe_load(frontmatter)
    assert isinstance(status, dict)
    return status


def test_project_status_matches_runtime_contracts() -> None:
    status = _read_project_status()
    defaults = {field.name: field.default for field in fields(AppSettings)}
    _assert_included(BASELINE_PATH)
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    pipeline = status["production_pipeline"]
    assert pipeline == defaults["retrieval_pipeline"]
    assert status["answer_strategy"] == defaults["answer_strategy"]
    assert pipeline == baseline["selected_pipeline_name"]
    assert status["production_pipeline_config_hash"] == baseline["pipeline_config_hash"]
    assert get_pipeline_config(str(pipeline)).config_hash() == baseline["pipeline_config_hash"]


def test_kite_diagnostic_evidence_is_tracked_and_consistent() -> None:
    evidence_paths = (KITE_MANIFEST_PATH, KITE_SUMMARY_PATH, *KITE_REPORT_PATHS.values())
    for path in evidence_paths:
        assert path.is_file(), f"missing clean-checkout evidence: {path.relative_to(ROOT)}"
        _assert_included(path)

    manifest = json.loads(KITE_MANIFEST_PATH.read_text(encoding="utf-8"))
    summary = json.loads(KITE_SUMMARY_PATH.read_text(encoding="utf-8"))
    assert manifest["benchmark_name"] == summary["benchmark_name"] == "kite-ai-papers"
    assert manifest["case_count"] == 15
    assert manifest["corpus_file_count"] == summary["corpus_file_count"] == 134
    assert manifest["query_sha256"] == summary["query_sha256"]
    assert manifest["corpus_file_sha256"] == summary["corpus_file_sha256"]
    manifest_sha256 = _sha256(KITE_MANIFEST_PATH)

    assert summary["production_decision"]["default_pipeline"] == "b1"
    assert summary["production_decision"]["default_name"] == "v1_flat_rerank"
    assert summary["production_decision"]["auto_switch"] is False
    for key, path in KITE_REPORT_PATHS.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        benchmark = report["benchmark"]
        provenance = report["provenance"]
        code = provenance["code"]
        metrics = report["metrics"]
        aggregate = summary["pipelines"][key]

        assert report["formal_run"] is False
        assert code["dirty"] is True
        assert len(code["working_tree_patch_sha256"]) == 64
        assert benchmark["query_sha256"] == manifest["query_sha256"]
        assert benchmark["corpus_file_sha256"] == manifest["corpus_file_sha256"]
        assert benchmark["corpus_file_count"] == manifest["corpus_file_count"]
        assert provenance["manifest_sha256"] == manifest_sha256
        assert provenance["manifest_path"] == "artifacts/evals/kite/manifest.json"
        assert metrics["case_count"] == aggregate["case_count"] == manifest["case_count"]
        assert metrics["valid_count"] == aggregate["valid_count"]
        assert metrics["mean_score"] == aggregate["mean_score"]
        assert metrics["p95_latency_ms"] == aggregate["p95_latency_ms"]
        assert metrics["mean_context_tokens"] == aggregate["mean_context_tokens"]


def test_project_status_references_existing_acceptance_evidence() -> None:
    status = _read_project_status()

    assert status["project_status_schema"] == 1
    assert isinstance(status["status_updated"], str)
    assert status["completed_through"] == "M6A"
    assert status["next_planned_goal"] == "M6B"
    assert status["implementation_authorized"] is False
    assert all(isinstance(goal, str) for goal in status["terminated_goals"])
    assert "docs/implementation/m6a_kite_data_acceptance.md" in status["acceptance_evidence"]
    unfinished_acceptance = (
        "m6b_kite_b1_acceptance.md",
        "m6c_kite_pipeline_acceptance.md",
        "m6d_evaluation_presentation_acceptance.md",
    )
    assert not any(
        path.endswith(name)
        for path in status["acceptance_evidence"]
        for name in unfinished_acceptance
    )
    for name in unfinished_acceptance:
        acceptance = (ROOT / "docs" / "implementation" / name).read_text(encoding="utf-8")
        assert "未通过正式验收" in acceptance
    for relative_path in status["acceptance_evidence"]:
        assert (ROOT / relative_path).is_file(), f"missing acceptance evidence: {relative_path}"


def test_agent_instructions_do_not_embed_stale_milestones() -> None:
    instructions = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    stale_phrases = (
        "Next authorized milestone",
        "M4 state is not implemented",
        "M4.1 must add",
        "M5 owns full trace",
        "defines executable Goal prompts",
        "**Updated:**",
        "**Branch:**",
        "The repository has completed",
    )

    assert "CURRENT STATE PREFLIGHT" in instructions
    assert "docs/research/v2_upgrade_plan.md" in instructions
    for phrase in stale_phrases:
        assert phrase not in instructions


def test_legacy_plans_are_marked_superseded() -> None:
    for path in SUPERSEDED_DOCS:
        heading = "\n".join(path.read_text(encoding="utf-8").splitlines()[:5])
        assert "SUPERSEDED" in heading, f"missing superseded marker: {path.relative_to(ROOT)}"
