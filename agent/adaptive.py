"""Bounded M4.1 adaptive evidence loop built on the frozen B1 retriever."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TypeVar

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from agent.schemas import AdaptiveAnswer, AdaptivePlan, EvidenceSufficiency
from core.settings import AppSettings
from indexing.index_versions import get_active_version_id
from indexing.retrieval_pipeline import get_pipeline_config
from indexing.token_count import estimate_token_count

BASELINE_PATH = Path("artifacts/evals/v2_m3_2/m4_fixed_baseline.json")
MAX_PLAN_ITEMS = 3
MAX_ROUNDS = 2
MAX_TOOL_CALLS = 4
MAX_EVIDENCE = 12
MAX_CONTEXT_TOKENS = 12_000


@dataclass(frozen=True)
class AdaptiveRunResult:
    strategy: str
    plan_items: list[dict[str, str]]
    evidence: list[dict[str, Any]]
    coverage: dict[str, float]
    final_answer: dict[str, Any]
    rounds: int
    tool_calls: int
    context_tokens: int
    termination_reason: str
    latency_ms: float


Planner = Callable[[str], list[dict[str, str]]]
Assessor = Callable[[list[dict[str, str]], list[dict[str, Any]]], list[dict[str, Any]]]
FollowUp = Callable[[list[dict[str, Any]]], str]
Answerer = Callable[[str, list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]
StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


def validate_m4_baseline(settings: AppSettings, *, base_dir: Path | None = None) -> dict[str, Any]:
    """Fail closed unless runtime retrieval is exactly the frozen M3.2 B1 contract."""
    root = base_dir or settings.base_dir
    path = root / BASELINE_PATH
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"M4 baseline contract is unreadable: {path}") from exc
    expected_name = "v1_flat_rerank"
    expected_hash = "ee7c1306250ba487ee2ca54de776fc70cb584c3bb02d4aca38cf7028e4956c17"
    if contract.get("selected_pipeline_name") != expected_name:
        raise ValueError("M4 baseline does not select v1_flat_rerank.")
    if contract.get("pipeline_config_hash") != expected_hash:
        raise ValueError("M4 baseline pipeline hash does not match the frozen contract.")
    if settings.retrieval_pipeline != expected_name:
        raise ValueError("Adaptive retrieval requires RETRIEVAL_PIPELINE=v1_flat_rerank.")
    if get_pipeline_config(settings.retrieval_pipeline).config_hash() != expected_hash:
        raise ValueError("Runtime B1 pipeline configuration differs from the frozen contract.")
    return contract


class AdaptiveEvidenceLoop:
    """Single-process bounded loop; it owns no persistence or retry worker."""

    def __init__(
        self,
        retriever: Any,
        *,
        expected_index_version: str | None,
        planner: Planner,
        assessor: Assessor,
        follow_up: FollowUp,
        answerer: Answerer,
    ) -> None:
        self.retriever = retriever
        self.expected_index_version = expected_index_version
        self.planner = planner
        self.assessor = assessor
        self.follow_up = follow_up
        self.answerer = answerer

    def run(
        self,
        query: str,
        *,
        scope: list[str] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> AdaptiveRunResult:
        started = perf_counter()
        normalized_scope = tuple(sorted(str(item) for item in (scope or [])))
        if _is_cancelled(cancelled):
            return self._result(
                "refuse", [], [], [], 0, 0, "cancelled", started, query
            )
        try:
            plan_items = _bounded_plan(self.planner(query), query)
        except Exception:
            return self._result(
                "refuse", [], [], [], 0, 0, "model_error", started, query
            )
        evidence: list[dict[str, Any]] = []
        queries: set[tuple[str, tuple[str, ...]]] = set()
        rounds = 0
        tool_calls = 0
        termination = "completed_first_round"
        assessments: list[dict[str, Any]] = []

        if not plan_items:
            return self._result(
                strategy="refuse",
                plan_items=[],
                evidence=[],
                assessments=[],
                rounds=0,
                tool_calls=0,
                termination="empty_plan",
                started=started,
                query=query,
            )

        first_queries = [item["query"] for item in plan_items]
        for retrieval_query in first_queries:
            if _is_cancelled(cancelled):
                return self._result(
                    "refuse",
                    plan_items,
                    evidence,
                    assessments,
                    rounds,
                    tool_calls,
                    "cancelled",
                    started,
                    query,
                )
            if tool_calls >= MAX_TOOL_CALLS:
                termination = "tool_budget_exhausted"
                break
            query_key = (_normalize_query(retrieval_query), normalized_scope)
            if query_key in queries:
                termination = "duplicate_query_scope"
                break
            queries.add(query_key)
            try:
                evidence.extend(self._retrieve(retrieval_query, scope=scope))
            except Exception:
                return self._result(
                    "refuse",
                    plan_items,
                    evidence,
                    assessments,
                    rounds,
                    tool_calls,
                    "retrieval_error",
                    started,
                    query,
                )
            tool_calls += 1
        rounds = 1
        evidence = _limit_evidence(evidence)
        try:
            assessments = _normalize_assessments(
                self.assessor(plan_items, evidence), plan_items
            )
        except Exception:
            return self._result(
                "refuse",
                plan_items,
                evidence,
                [],
                rounds,
                tool_calls,
                "model_error",
                started,
                query,
            )
        if _all_covered(assessments):
            return self._result(
                strategy="fixed",
                plan_items=plan_items,
                evidence=evidence,
                assessments=assessments,
                rounds=rounds,
                tool_calls=tool_calls,
                termination=termination,
                started=started,
                query=query,
            )

        missing = [item for item in assessments if not item["covered"]]
        try:
            follow_up_query = self.follow_up(missing).strip()
        except Exception:
            return self._result(
                "refuse",
                plan_items,
                evidence,
                assessments,
                rounds,
                tool_calls,
                "model_error",
                started,
                query,
            )
        if not follow_up_query:
            termination = "no_follow_up_query"
            return self._result("refuse", plan_items, evidence, assessments, rounds, tool_calls, termination, started, query)
        follow_up_key = (_normalize_query(follow_up_query), normalized_scope)
        if follow_up_key in queries:
            termination = "duplicate_query_scope"
            return self._result("refuse", plan_items, evidence, assessments, rounds, tool_calls, termination, started, query)
        if tool_calls >= MAX_TOOL_CALLS:
            termination = "tool_budget_exhausted"
            return self._result("refuse", plan_items, evidence, assessments, rounds, tool_calls, termination, started, query)

        prior_ids = {item["evidence_id"] for item in evidence}
        prior_coverage = _coverage_total(assessments)
        queries.add(follow_up_key)
        if _is_cancelled(cancelled):
            return self._result("refuse", plan_items, evidence, assessments, rounds, tool_calls, "cancelled", started, query)
        try:
            evidence.extend(self._retrieve(follow_up_query, scope=scope))
        except Exception:
            return self._result("refuse", plan_items, evidence, assessments, rounds, tool_calls, "retrieval_error", started, query)
        tool_calls += 1
        rounds = 2
        evidence = _limit_evidence(evidence)
        try:
            updated = _normalize_assessments(self.assessor(plan_items, evidence), plan_items)
        except Exception:
            return self._result("refuse", plan_items, evidence, assessments, rounds, tool_calls, "model_error", started, query)
        updated_ids = {item["evidence_id"] for item in evidence}
        if updated_ids == prior_ids:
            termination = "evidence_ids_unchanged"
        elif _coverage_total(updated) <= prior_coverage:
            termination = "coverage_not_improved"
        elif _all_covered(updated):
            termination = "completed_second_round"
        else:
            termination = "second_round_incomplete"
        strategy = "adaptive" if _coverage_total(updated) > prior_coverage else "refuse"
        return self._result(strategy, plan_items, evidence, updated, rounds, tool_calls, termination, started, query)

    def _retrieve(self, query: str, *, scope: list[str] | None) -> list[dict[str, Any]]:
        query_plan = {"subqueries": [query]}
        if scope:
            query_plan["scope_ids"] = list(scope)
        packed = self.retriever.retrieve(query, query_plan=query_plan)
        passages = list(getattr(packed, "passages", packed))
        return [
            _to_evidence(document, expected_index_version=self.expected_index_version)
            for document in passages
        ]

    def _result(
        self,
        strategy: str,
        plan_items: list[dict[str, str]],
        evidence: list[dict[str, Any]],
        assessments: list[dict[str, Any]],
        rounds: int,
        tool_calls: int,
        termination: str,
        started: float,
        query: str,
    ) -> AdaptiveRunResult:
        valid_evidence = _limit_evidence(evidence)
        coverage = {item["requirement_id"]: float(item["coverage"]) for item in assessments}
        try:
            final_answer = self.answerer(query, valid_evidence, assessments)
        except Exception:
            final_answer = {"answer": "", "claims": [], "limitations": "Answer generation failed."}
            termination = "model_error"
        final_answer = _validate_claims(final_answer, valid_evidence, assessments)
        if strategy == "refuse" and not final_answer["answer"]:
            final_answer["answer"] = "当前论文库中的证据不足以可靠回答这个问题。"
        return AdaptiveRunResult(
            strategy=strategy,
            plan_items=plan_items,
            evidence=valid_evidence,
            coverage=coverage,
            final_answer=final_answer,
            rounds=rounds,
            tool_calls=tool_calls,
            context_tokens=sum(int(item["token_count"]) for item in valid_evidence),
            termination_reason=termination,
            latency_ms=round((perf_counter() - started) * 1000, 4),
        )


def build_live_loop(settings: AppSettings, retriever: Any) -> AdaptiveEvidenceLoop:
    """Build the synchronous M4.1 loop after validating the baseline contract."""
    validate_m4_baseline(settings)
    return AdaptiveEvidenceLoop(
        retriever,
        expected_index_version=get_active_version_id(settings),
        planner=_live_planner,
        assessor=_live_assessor,
        follow_up=_live_follow_up,
        answerer=_live_answerer,
    )


def _live_planner(query: str) -> list[dict[str, str]]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.prompts import get_adaptive_plan_prompt

    try:
        response = _invoke_structured_json(
            "adaptive_plan",
            AdaptivePlan,
            [
                SystemMessage(content=get_adaptive_plan_prompt()),
                HumanMessage(content=query),
            ],
        )
        planned = [item.model_dump() for item in response.requirements]
        return planned or _fallback_plan(query)
    except Exception:
        return _fallback_plan(query)


def _live_assessor(plan_items: list[dict[str, str]], evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.prompts import get_evidence_sufficiency_prompt

    payload = {"requirements": plan_items, "evidence": [{key: item[key] for key in ("evidence_id", "quote", "page", "section_path")} for item in evidence]}
    try:
        response = _invoke_structured_json(
            "adaptive_sufficiency",
            EvidenceSufficiency,
            [
                SystemMessage(content=get_evidence_sufficiency_prompt()),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ],
        )
        return [item.model_dump() for item in response.items]
    except Exception:
        ids = [item["evidence_id"] for item in evidence]
        return [{"requirement_id": item["id"], "covered": bool(ids), "evidence_ids": ids[:1], "coverage": 1.0 if ids else 0.0, "missing_reason": "Semantic assessor unavailable." if not ids else "", "recommended_follow_up_query": item["query"] if not ids else ""} for item in plan_items]


def _live_follow_up(missing: list[dict[str, Any]]) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.prompts import get_adaptive_follow_up_prompt
    from llms.llm import get_llm_by_type

    try:
        response = get_llm_by_type("adaptive_follow_up").with_config(temperature=0).invoke([SystemMessage(content=get_adaptive_follow_up_prompt()), HumanMessage(content=json.dumps(missing, ensure_ascii=False))])
        return str(response.content).strip()
    except Exception:
        return str(missing[0].get("recommended_follow_up_query", "")) if missing else ""


def _live_answerer(_: str, evidence: list[dict[str, Any]], assessments: list[dict[str, Any]]) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.prompts import get_adaptive_answer_prompt

    payload = {"evidence": evidence, "assessments": assessments}
    try:
        response = _invoke_structured_json(
            "adaptive_answer",
            AdaptiveAnswer,
            [
                SystemMessage(content=get_adaptive_answer_prompt()),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ],
        )
        return response.model_dump()
    except Exception:
        quotes = [item["quote"] for item in evidence[:3]]
        return {"answer": " ".join(quotes), "claims": [{"claim": quote, "evidence_ids": [item["evidence_id"]], "major": True} for quote, item in zip(quotes, evidence, strict=False)], "limitations": "Semantic answer generation was unavailable."}


def _bounded_plan(plan: list[dict[str, str]], fallback_query: str) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for index, item in enumerate(plan[:MAX_PLAN_ITEMS], start=1):
        requirement = str(item.get("requirement", "")).strip()
        query = str(item.get("query", "")).strip()
        if requirement and query:
            result.append({"id": str(item.get("id") or f"requirement-{index}"), "requirement": requirement, "query": query})
    return result or [{"id": "requirement-1", "requirement": fallback_query, "query": fallback_query}]


def _fallback_plan(query: str) -> list[dict[str, str]]:
    """Split only for planning; evidence coverage, never wording, selects follow-up."""
    normalized = " ".join(query.split())
    splitters = (" 与 ", " 和 ", "以及", "与")
    for splitter in splitters:
        if splitter not in normalized:
            continue
        left, right = normalized.split(splitter, maxsplit=1)
        left = left.removeprefix("对比").strip(" ：:，,。")
        right = right.split("分别", maxsplit=1)[0].strip(" ：:，,。")
        if left and right:
            return [
                {"id": "requirement-1", "requirement": left, "query": left},
                {"id": "requirement-2", "requirement": right, "query": right},
            ]
    return [{"id": "requirement-1", "requirement": normalized, "query": normalized}]


def _to_evidence(document: Document, *, expected_index_version: str | None) -> dict[str, Any]:
    metadata = dict(document.metadata)
    evidence_id = str(metadata.get("passage_id") or metadata.get("node_id") or metadata.get("id") or "").strip()
    quote = str(metadata.get("quote_text") or document.page_content or "").strip()
    page = metadata.get("page")
    index_version = metadata.get("index_version")
    if not evidence_id or not quote or not isinstance(page, int) or page <= 0:
        raise ValueError("Retrieved evidence lacks a stable ID, quote, or page location.")
    if expected_index_version is not None and index_version not in {None, expected_index_version}:
        raise ValueError("Retrieved evidence belongs to a different active index version.")
    section_path = metadata.get("section_path") or metadata.get("title_path") or []
    if isinstance(section_path, str):
        section_path = [section_path]
    return {"evidence_id": evidence_id, "paper_id": str(metadata.get("paper_id") or metadata.get("doc_id") or metadata.get("source") or "").strip(), "source": str(metadata.get("source") or "unknown").strip(), "section_path": [str(item) for item in section_path if str(item).strip()], "page": page, "quote": quote, "index_version": expected_index_version, "token_count": estimate_token_count(quote)}


def _limit_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    tokens = 0
    for item in evidence:
        evidence_id = str(item.get("evidence_id", ""))
        if not evidence_id or evidence_id in seen:
            continue
        token_count = int(item.get("token_count", 0))
        if len(result) >= MAX_EVIDENCE or tokens + token_count > MAX_CONTEXT_TOKENS:
            continue
        seen.add(evidence_id)
        result.append(item)
        tokens += token_count
    return result


def _normalize_assessments(assessments: list[dict[str, Any]], plan_items: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_id = {str(item.get("requirement_id", "")): item for item in assessments}
    normalized = []
    for plan in plan_items:
        raw = by_id.get(plan["id"], {})
        normalized.append({"requirement_id": plan["id"], "covered": bool(raw.get("covered", False)), "evidence_ids": [str(value) for value in raw.get("evidence_ids", []) if str(value)], "coverage": max(0.0, min(1.0, float(raw.get("coverage", 0.0)))), "missing_reason": str(raw.get("missing_reason", "")), "recommended_follow_up_query": str(raw.get("recommended_follow_up_query", ""))})
    return normalized


def _validate_claims(answer: dict[str, Any], evidence: list[dict[str, Any]], assessments: list[dict[str, Any]]) -> dict[str, Any]:
    valid_ids = {item["evidence_id"] for item in evidence}
    claims = []
    for claim in answer.get("claims", []) or []:
        if not isinstance(claim, dict):
            continue
        ids = [str(item) for item in claim.get("evidence_ids", []) if str(item) in valid_ids]
        if bool(claim.get("major", True)) and not ids:
            continue
        claims.append({"claim": str(claim.get("claim", "")).strip(), "evidence_ids": ids, "major": bool(claim.get("major", True))})
    limitations = str(answer.get("limitations", "")).strip()
    missing = [item["requirement_id"] for item in assessments if not item["covered"]]
    if missing:
        suffix = "Missing evidence for: " + ", ".join(missing) + "."
        limitations = f"{limitations} {suffix}".strip()
    return {"answer": str(answer.get("answer", "")).strip(), "claims": claims, "limitations": limitations}


def _all_covered(assessments: list[dict[str, Any]]) -> bool:
    return bool(assessments) and all(item["covered"] for item in assessments)


def _coverage_total(assessments: list[dict[str, Any]]) -> float:
    return sum(float(item["coverage"]) for item in assessments)


def _normalize_query(query: str) -> str:
    return " ".join(query.casefold().split())


def _is_cancelled(cancelled: Callable[[], bool] | None) -> bool:
    return bool(cancelled and cancelled())


def invoke_structured_json(
    task_type: str,
    model: type[StructuredModel],
    messages: list[BaseMessage],
) -> StructuredModel:
    """Use provider-neutral JSON text and validate it with the declared schema."""
    from llms.llm import get_llm_by_type

    response = get_llm_by_type(task_type).with_config(temperature=0).invoke(messages)
    content = str(response.content).strip()
    object_start = content.find("{")
    array_start = content.find("[")
    starts = [position for position in (object_start, array_start) if position >= 0]
    if not starts:
        raise ValueError("Structured model response did not contain a JSON object.")
    start = min(starts)
    payload, _ = json.JSONDecoder().raw_decode(content[start:])
    if model is EvidenceSufficiency and isinstance(payload, list):
        payload = {"items": payload}
    return model.model_validate(payload)


_invoke_structured_json = invoke_structured_json
