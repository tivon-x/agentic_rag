import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import get_aggregation_prompt
from agent.schemas import GroundedAnswer
from agent.states import GraphState
from core.corpus_profile import build_answer_style_instruction
from core.rag_answer import render_grounded_answer
from llms.llm import get_llm_by_type


logger = logging.getLogger(__name__)

MAX_AGGREGATE_INPUT_CHARS = 180_000
MAX_AGGREGATE_GROUPS = 3
MAX_AGGREGATE_EVIDENCE_PER_GROUP = 12
MAX_AGGREGATE_TEXT_CHARS = 512
MAX_AGGREGATE_QUOTE_CHARS = 400
_PACKED_CONTEXT_KEYS = {
    "subquery",
    "total_tokens",
    "dropped_candidates",
    "packing_strategy",
    "passage_count",
}


def aggregate_answers(state: GraphState):
    evidence_groups = state.get("evidenceGroups", [])
    if not evidence_groups:
        return {
            "answerGenerationFailed": True,
            "messages": [AIMessage(content="No answers were generated.")],
        }

    payload, sorted_groups = _build_aggregate_payload(state)

    llm = get_llm_by_type("aggregate_answers")
    preferred_answer_style = build_answer_style_instruction(
        state.get("corpusProfileData", {})
    )
    generation_failed = False
    try:
        structured_llm = llm.with_config(temperature=0).with_structured_output(
            GroundedAnswer
        )
        grounded_answer = structured_llm.invoke(
            [
                SystemMessage(
                    content=get_aggregation_prompt(
                        preferred_answer_style=preferred_answer_style,
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ),
            ]
        ).model_dump()
    except Exception as exc:
        generation_failed = True
        logger.exception(
            "Structured aggregate answer generation failed; using evidence fallback "
            "and marking response failed: %s",
            exc,
        )
        collected_evidence: list[dict] = []
        for group in sorted_groups:
            for item in group.get("evidence", []):
                collected_evidence.append(item)

        unique_evidence: list[dict] = []
        seen_keys: set[str] = set()
        for item in collected_evidence:
            key = (
                f"{item.get('doc_id', '')}:{item.get('node_id', '')}:"
                f"{item.get('quote', '')}"
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_evidence.append(item)

        evidence_count = len(unique_evidence)
        answer = (
            "I couldn't find any relevant information in the available sources to answer your question."
            if evidence_count == 0
            else " ".join(
                item.get("quote", "").strip()
                for item in unique_evidence[:3]
                if item.get("quote", "").strip()
            ).strip()
        )
        grounded_answer = {
            "answer": answer
            or "I couldn't find any relevant information in the available sources to answer your question.",
            "reasoning_summary": (
                f"Synthesized from {len(sorted_groups)} evidence group(s) and "
                f"{evidence_count} unique evidence item(s)."
            ),
            "evidence": unique_evidence[:5],
            "confidence": min(0.95, 0.25 + (0.12 * evidence_count)),
            "limitations": (
                "Available evidence is limited to the retrieved passages."
                if evidence_count
                else "No structured evidence was captured from retrieval."
            ),
        }

    content = render_grounded_answer(grounded_answer)
    result: dict[str, Any] = {
        "groundedAnswer": grounded_answer,
        "messages": [AIMessage(content=content)],
    }
    if generation_failed:
        result["answerGenerationFailed"] = True
    return result


def _build_aggregate_payload(
    state: GraphState,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a bounded, citation-ready synthesis payload.

    Retrieval artifacts retain raw passages for tracing, but passing those artifacts
    to the synthesis model duplicates the evidence and can exceed provider limits.
    """
    sorted_groups = sorted(
        (
            group
            for group in state.get("evidenceGroups", [])
            if isinstance(group, dict)
        ),
        key=lambda item: str(item.get("subquery", "")),
    )[:MAX_AGGREGATE_GROUPS]
    compact_groups = [_compact_evidence_group(group) for group in sorted_groups]

    packed_contexts = sorted(
        (
            context
            for context in state.get("packedContexts", [])
            if isinstance(context, dict)
        ),
        key=lambda item: str(item.get("subquery", "")),
    )[:MAX_AGGREGATE_GROUPS]

    payload: dict[str, Any] = {
        "question": _compact_text(state.get("originalQuery", ""), 2_000),
        "query_plan": _compact_query_plan(state.get("queryPlan", {})),
        "packed_context": [_compact_packed_context(item) for item in packed_contexts],
        "evidence_groups": compact_groups,
    }
    serialized_size = len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    if serialized_size > MAX_AGGREGATE_INPUT_CHARS:
        raise ValueError(
            "Aggregate synthesis payload exceeds the bounded input contract: "
            f"{serialized_size} > {MAX_AGGREGATE_INPUT_CHARS} characters."
        )
    return payload, compact_groups


def _compact_query_plan(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    if value.get("intent") is not None:
        result["intent"] = _compact_text(value.get("intent"), 64)
    for key in ("subqueries", "preferred_node_types"):
        raw_values = value.get(key)
        if not isinstance(raw_values, list):
            continue
        result[key] = [_compact_text(item, 1_000) for item in raw_values[:3]]
    return result


def _compact_packed_context(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in _PACKED_CONTEXT_KEYS:
        if key not in value:
            continue
        item = value[key]
        if isinstance(item, str):
            result[key] = _compact_text(item, 128)
        elif isinstance(item, (int, float, bool)) or item is None:
            result[key] = item
    return result


def _compact_evidence_group(value: dict[str, Any]) -> dict[str, Any]:
    raw_items = value.get("evidence", [])
    evidence = raw_items if isinstance(raw_items, list) else []
    compact_items = [
        compact_item
        for item in evidence[:MAX_AGGREGATE_EVIDENCE_PER_GROUP]
        if (compact_item := _compact_evidence_item(item)) is not None
    ]
    return {
        "subquery": _compact_text(value.get("subquery"), 1_000),
        "intent": _compact_text(value.get("intent"), 64),
        "packed_context": _compact_packed_context(
            value.get("packed_context", {})
            if isinstance(value.get("packed_context", {}), dict)
            else {}
        ),
        "evidence": compact_items,
    }


def _compact_evidence_item(value: object) -> dict[str, Any] | None:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict):
        return None

    section_path = value.get("section_path") or value.get("title_path") or []
    if isinstance(section_path, str):
        section_path = [section_path]
    if not isinstance(section_path, list):
        section_path = []
    score = value.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        score = None
    page = value.get("page")
    if type(page) is not int:
        page = None
    return {
        "doc_id": _compact_text(value.get("doc_id"), 256),
        "node_id": _compact_text(value.get("node_id"), 256),
        "paper_id": _compact_text(value.get("paper_id"), 256) or None,
        "paper_title": _compact_text(value.get("paper_title"), 512) or None,
        "source": _compact_text(value.get("source"), 512),
        "section_path": [
            _compact_text(item, 128) for item in section_path[:8] if str(item).strip()
        ],
        "page": page,
        "quote": _compact_text(
            value.get("quote") or value.get("quote_text"),
            MAX_AGGREGATE_QUOTE_CHARS,
        ),
        "score": score,
        "relevance": _compact_text(value.get("relevance"), MAX_AGGREGATE_TEXT_CHARS)
        or None,
    }


def _compact_text(value: object, limit: int) -> str:
    text = "" if value is None else str(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
