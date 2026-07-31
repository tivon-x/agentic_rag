"""Read-only B1 authoring snapshot for M4.1.2 candidate selection."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from agent.adaptive import validate_m4_baseline
from core.factory import build_retriever
from core.settings import load_settings
from evals.m4_1_1_runner import _settings_for_eval_index, _validate_index_contract


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "evals/datasets/m4_1_2_authoring_candidates.json"
INDEX_MANIFEST = ROOT / "artifacts/evals/v2_m3_2/old_dev/manifests/b1.json"
OUTPUT = ROOT / "artifacts/evals/v2_m4_1_2/m4_1_2_authoring_snapshot.json"


def run() -> dict[str, Any]:
    """Query frozen B1 once per candidate; never invokes Adaptive or mutates indexes."""
    settings = load_settings(base_dir=ROOT)
    validate_m4_baseline(settings, base_dir=ROOT)
    manifest = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
    _validate_index_contract(settings, manifest)
    retriever = build_retriever(_settings_for_eval_index(settings, manifest))
    if retriever is None:
        raise ValueError("Frozen M3.2 B1 evaluation index is unavailable.")
    rows = []
    for candidate in json.loads(CANDIDATES.read_text(encoding="utf-8")):
        started = perf_counter()
        packed = retriever.retrieve(candidate["query"], query_plan={"subqueries": [candidate["query"]]})
        passages = list(getattr(packed, "passages", packed))
        rows.append({
            **candidate,
            "latency_ms": round((perf_counter() - started) * 1000, 4),
            "evidence": [_evidence(document) for document in passages],
        })
    result = {
        "schema_version": 1,
        "purpose": "M4.1.2 offline candidate authoring only; no Adaptive invocation.",
        "created_at": datetime.now(UTC).isoformat(),
        "baseline_contract": {
            "selected_pipeline_name": "v1_flat_rerank",
            "pipeline_config_hash": "ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17",
            "evaluation_index_manifest": str(INDEX_MANIFEST.relative_to(ROOT)),
        },
        "candidates": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _evidence(document: Any) -> dict[str, Any]:
    metadata = dict(document.metadata)
    return {
        "evidence_id": str(metadata.get("passage_id") or metadata.get("node_id") or metadata.get("id") or ""),
        "paper_id": str(metadata.get("paper_id") or metadata.get("doc_id") or ""),
        "source": str(metadata.get("source") or ""),
        "section_path": metadata.get("section_path") or metadata.get("title_path") or [],
        "page": metadata.get("page"),
        "quote": str(metadata.get("quote_text") or document.page_content or ""),
        "index_version": metadata.get("index_version"),
    }


if __name__ == "__main__":
    run()
