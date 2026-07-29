from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

from langchain_core.documents import Document
import pytest
import yaml

from evals.m3_1_experiments import (
    dev_gate,
    final_gate,
    pipeline_from_dict,
    round1_boost_off,
    round1_seed_configs,
    round2_configs,
    round3_blend_configs,
    round3_stability_configs,
)
from core.settings import load_settings
from evals.m3_1_runner import (
    _build_pipeline_manifest,
    _checkpoint_identity,
    _load_frozen_finalist,
    _prepare_content_addressed_settings,
    _write_json_atomic,
)
from evals.run_lock import exclusive_run_lock
from evals.select_candidate import SELECTION_METHOD
from evals.v2_runner import (
    RetrievalCase,
    _capture_code_state,
    _embedding_contract,
    _validate_eval_manifest,
    evaluate_answer_smoke,
    evaluate_retrieval_protocol,
)
from evals.v2_corpus import sha256_file
from indexing.retrieval_pipeline import (
    PIPELINE_REGISTRY,
    PackedContext,
    RetrievalCandidate,
    get_pipeline_config,
    prepare_rerank_document,
    quote_document,
)
from indexing.retriever import merge_rerank_results


class _ProtocolRetriever:
    def __init__(self) -> None:
        self.call_count = 0

    def retrieve(self, query: str) -> PackedContext:
        self.call_count += 1
        passage = Document(
            page_content="Source-faithful quote.",
            metadata={
                "passage_id": "passage-1",
                "paper_id": "paper-1",
                "section_id": "section-1",
                "quote_text": "Source-faithful quote.",
                "retrieval_text": (
                    "[TITLE] Hidden title\n"
                    "[SECTION] Hidden section\n"
                    "Source-faithful quote."
                ),
            },
        )
        return PackedContext(
            passages=[passage],
            total_tokens=4,
            dropped_candidates=0,
            packing_strategy="test",
            debug={
                "stages": {
                    "rerank": [
                        {
                            "passage_id": "passage-1",
                            "paper_id": "paper-1",
                            "section_id": "section-1",
                        }
                    ]
                },
                "timings_ms": {
                    "query_embedding": 1.0,
                    "retrieval_total": 2.0,
                },
            },
        )


def _candidate(passage_id: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        document=Document(
            page_content=f"quote {passage_id}",
            metadata={"passage_id": passage_id},
        ),
        score=1.0,
    )


def test_experiment_overrides_generate_deterministic_pipeline_hashes():
    first = round1_seed_configs()
    second = round1_seed_configs()

    assert list(first) == list(second)
    assert {
        key: pipeline.config_hash() for key, pipeline in first.items()
    } == {
        key: pipeline.config_hash() for key, pipeline in second.items()
    }


def test_experiment_matrix_contains_exactly_24_candidates():
    round1 = round1_seed_configs()
    boost_key, _ = round1_boost_off(
        next(iter(round1)),
        next(iter(round1.values())),
    )
    finalists = list(round1.items())[:2]
    round2 = round2_configs(finalists)
    round3_bases = list(round2.items())[:2]
    blends = round3_blend_configs(round3_bases)
    stability = round3_stability_configs(next(iter(blends.values())))

    keys = [
        *round1,
        boost_key,
        *round2,
        *blends,
        *stability,
    ]
    assert len(keys) == 24
    assert len(set(keys)) == 24


def test_index_contract_changes_for_metadata_and_tokenizer():
    base = get_pipeline_config("b1")
    metadata = replace(
        base,
        use_metadata_prefix=True,
        dense_use_metadata_prefix=True,
        sparse_use_metadata_prefix=True,
        metadata_prefix_fields=("section",),
    )
    tokenizer = replace(base, tokenizer="mixed_v1")

    assert metadata.index_contract() != base.index_contract()
    assert tokenizer.index_contract() != base.index_contract()


def test_legacy_metadata_override_updates_both_index_channels():
    pipeline = PIPELINE_REGISTRY["b2_no_metadata"]

    assert pipeline.channel_uses_metadata_prefix("dense") is False
    assert pipeline.channel_uses_metadata_prefix("sparse") is False


def test_rerank_input_does_not_change_quote_context():
    source = Document(
        page_content="[TITLE] Secret\n[SECTION] Intro\nFaithful quote.",
        metadata={
            "quote_text": "Faithful quote.",
            "retrieval_text": (
                "[TITLE] Secret\n[SECTION] Intro\nFaithful quote."
            ),
        },
    )

    rerank = prepare_rerank_document(
        source,
        rerank_input="title_section_quote",
    )
    context = quote_document(rerank)

    assert rerank.page_content.startswith("[TITLE] Secret")
    assert context.page_content == "Faithful quote."
    assert "[TITLE]" not in context.page_content


def test_weighted_rrf_handles_missing_ranks_ties_and_duplicates():
    fusion = [_candidate("a"), _candidate("b"), _candidate("c")]
    reranked = [
        fusion[2].document,
        fusion[2].document,
        fusion[0].document,
    ]

    first, ranks = merge_rerank_results(
        fusion,
        reranked,
        mode="weighted_rrf",
        rrf_k=60,
        fusion_rank_weight=1.0,
        rerank_rank_weight=1.0,
    )
    second, second_ranks = merge_rerank_results(
        fusion,
        reranked,
        mode="weighted_rrf",
        rrf_k=60,
        fusion_rank_weight=1.0,
        rerank_rank_weight=1.0,
    )

    assert [
        row.document.metadata["passage_id"] for row in first
    ] == ["a", "c", "b"]
    assert ranks == {"c": 0, "a": 2}
    assert ranks == second_ranks
    assert [
        row.document.metadata["passage_id"] for row in second
    ] == ["a", "c", "b"]


def test_boost_policy_is_part_of_config_and_manifest_contract():
    pipeline = replace(get_pipeline_config("b1"), boost_policy="off")
    manifest = _build_pipeline_manifest(
        {},
        key="candidate",
        pipeline=pipeline,
        reranker={"backend": "flashrank"},
        index_fingerprint="abc",
        index_dir=Path("index"),
    )

    assert pipeline.retrieval_contract()["boost_policy"] == "off"
    assert asdict(pipeline)["boost_policy"] == "off"
    assert manifest["pipeline"]["boost_policy"] == "off"
    assert manifest["retrieval_contract"]["boost_policy"] == "off"
    assert pipeline.config_hash() != get_pipeline_config("b1").config_hash()


def test_run_directory_lock_blocks_second_writer(tmp_path):
    run_dir = tmp_path / "run"

    with exclusive_run_lock(run_dir):
        with pytest.raises(RuntimeError, match="already active"):
            with exclusive_run_lock(run_dir):
                pass


def test_dev_config_freezes_distinct_old_and_holdout_hashes():
    repo_root = Path(__file__).resolve().parent.parent
    config = yaml.safe_load(
        (repo_root / "evals" / "configs" / "v2_m3_1_dev.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["retrieval_dataset_sha256"] != (
        config["holdout_dataset_sha256"]
    )
    assert config["retrieval_dataset"].endswith(
        "retrieval_v2_core.jsonl"
    )
    assert config["holdout_dataset"].endswith(
        "retrieval_v2_core_holdout.jsonl"
    )
    assert sha256_file(
        repo_root / config["retrieval_dataset"]
    ) == config["retrieval_dataset_sha256"]
    assert sha256_file(
        repo_root / config["holdout_dataset"]
    ) == config["holdout_dataset_sha256"]


def test_holdout_has_48_valid_stratified_cases():
    repo_root = Path(__file__).resolve().parent.parent
    cases = [
        json.loads(line)
        for line in (
            repo_root
            / "evals"
            / "datasets"
            / "retrieval_v2_core_holdout.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    counts = {
        category: sum(
            case["category"] == category for case in cases
        )
        for category in {
            "exact_term_definition",
            "method_section_location",
            "experiment_number_table",
            "cross_paper_or_section",
        }
    }

    assert len(cases) == 48
    assert set(counts.values()) == {12}
    assert all(case["gold_passage_ids"] for case in cases)
    assert all(case["gold_paper_ids"] for case in cases)
    assert all(case["gold_section_ids"] for case in cases)


def test_candidate_selection_declares_no_composite_score():
    assert SELECTION_METHOD == "frozen_lexicographic_no_composite_score"


def test_dev_and_final_gates_apply_frozen_thresholds():
    baseline = _gate_report(baseline=True)
    candidate = _gate_report(baseline=False)

    assert dev_gate(candidate, baseline=baseline)["passed"] is True
    assert final_gate(candidate, baseline=baseline)["passed"] is True


def test_final_config_accepts_exactly_one_frozen_finalist():
    pipeline = replace(
        get_pipeline_config("b1"),
        name="v2_fixed_optimized",
    )
    finalist = {
        "config": asdict(pipeline),
        "config_sha256": pipeline.config_hash(),
    }

    loaded, restored = _load_frozen_finalist({"finalists": [finalist]})

    assert loaded == finalist
    assert restored == pipeline
    with pytest.raises(ValueError, match="exactly one"):
        _load_frozen_finalist({"finalists": []})
    with pytest.raises(ValueError, match="exactly one"):
        _load_frozen_finalist({"finalists": [finalist, finalist]})


def test_eval_manifest_contract_mismatch_fails_immediately():
    repo_root = Path(__file__).resolve().parent.parent
    settings = load_settings(base_dir=repo_root)
    pipeline = get_pipeline_config("b1")
    manifest = {
        "embedding": _embedding_contract(settings),
        "retrieval": pipeline.index_contract(),
        "pipeline_config_hash": pipeline.config_hash(),
        "parser_artifact_sha256": "artifact",
    }
    manifest["retrieval"] = {"unexpected": True}

    with pytest.raises(ValueError, match="retrieval contract"):
        _validate_eval_manifest(
            settings,
            pipeline=pipeline,
            manifest=manifest,
            parser_artifact_sha256="artifact",
        )


def test_content_addressed_variant_path_matches_index_directory(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    settings = load_settings(base_dir=repo_root)

    prepared, index_dir = _prepare_content_addressed_settings(
        settings,
        run_dir=tmp_path,
        index_fingerprint="hash",
    )

    assert index_dir == tmp_path / "indexes" / "by_contract" / "hash"
    assert prepared.index_dir == index_dir


def test_code_state_captures_dirty_and_untracked_files(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent

    state = _capture_code_state(repo_root, run_dir=tmp_path)

    assert state["commit"]
    assert len(state["working_tree_patch_sha256"]) == 64
    assert Path(state["working_tree_patch_path"]).exists()


def test_pipeline_checkpoint_identity_is_deterministic_and_code_sensitive():
    pipeline = get_pipeline_config("b1")
    common = {
        "key": "b1",
        "pipeline": pipeline,
        "artifact_sha": "artifact",
        "dataset_sha": "dataset",
        "embedding": {"model": "embedding"},
    }
    first = _checkpoint_identity(
        **common,
        code_state={
            "commit": "commit",
            "dirty": True,
            "working_tree_patch_sha256": "patch",
            "working_tree_patch_path": "ignored-path",
        },
    )
    second = _checkpoint_identity(
        **common,
        code_state={
            "commit": "commit",
            "dirty": True,
            "working_tree_patch_sha256": "patch",
            "working_tree_patch_path": "other-ignored-path",
        },
    )
    changed = _checkpoint_identity(
        **common,
        code_state={
            "commit": "commit",
            "dirty": True,
            "working_tree_patch_sha256": "changed",
        },
    )

    assert first == second
    assert first != changed


def test_pipeline_checkpoint_write_is_atomic(tmp_path):
    path = tmp_path / "checkpoints" / "candidate.json"

    _write_json_atomic(path, {"status": "complete"})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "status": "complete"
    }
    assert not path.with_suffix(".json.tmp").exists()


def test_pipeline_from_dict_restores_tuple_fields():
    pipeline = get_pipeline_config("b1")
    payload = asdict(pipeline)
    payload["metadata_prefix_fields"] = list(
        payload["metadata_prefix_fields"]
    )

    assert pipeline_from_dict(payload) == pipeline


def test_latency_warmup_is_excluded_from_formal_samples():
    retriever = _ProtocolRetriever()
    case = RetrievalCase(
        case_id="case-1",
        question="question",
        category="exact_term_definition",
        gold_passage_ids=("passage-1",),
        gold_paper_ids=("paper-1",),
        gold_section_ids=("section-1",),
        tags=(),
        notes="",
    )

    report = evaluate_retrieval_protocol(
        [case],
        retriever=retriever,
        repeat_count=2,
    )

    assert retriever.call_count == 3
    assert report["latency_protocol"]["warmup_excluded"] is True
    assert report["latency_protocol"]["sample_count"] == 2


def test_answer_smoke_has_no_metadata_prefix_leak():
    report = evaluate_answer_smoke(
        [{"case_id": "a1", "question": "question"}],
        retriever=_ProtocolRetriever(),
    )

    assert report["metadata_prefix_leak_count"] == 0


def _gate_report(*, baseline: bool) -> dict:
    categories = (
        "exact_term_definition",
        "method_section_location",
        "experiment_number_table",
        "cross_paper_or_section",
    )
    cases = []
    subset_hits = {category: 12 for category in categories}
    for category_index, category in enumerate(categories):
        for case_index in range(12):
            is_improvement = (
                category_index < 2 and case_index < 5
            )
            rank = None if baseline and is_improvement else 1
            cases.append(
                {
                    "case_id": f"{category}-{case_index:02d}",
                    "category": category,
                    "first_gold_rank": rank,
                    "recall_at_10": 0.0 if rank is None else 1.0,
                }
            )
            if baseline and is_improvement:
                subset_hits[category] -= 1
    hit_count = sum(subset_hits.values())
    return {
        "metrics": {
            "recall_at_10": hit_count / 48,
            "mrr_at_10": hit_count / 48,
            "ndcg_at_10": hit_count / 48,
            "p95_latency_ms": 100.0,
        },
        "subsets": {
            category: {"recall_at_10_hit_count": subset_hits[category]}
            for category in categories
        },
        "cases": cases,
    }
