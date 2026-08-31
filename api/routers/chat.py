from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sse_starlette.sse import EventSourceResponse

from api.db.database import (
    ChatHistoryLimitError,
    append_chat_session_message,
    get_chat_session,
    get_chat_session_messages,
    list_chat_sessions,
)
from api.dependencies import get_settings
from api.models.chat import (
    ChatEvidence,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatSessionListResponse,
    ChatSessionResponse,
    ChatSessionSummary,
)
from api.services.graph_cache import get_cached_graph
from core.factory import build_retriever
from core.rag_answer import format_retrieval_only_answer, render_grounded_citations
from core.settings import AppSettings


router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)
_ACTIVE_STREAM_SESSIONS: set[str] = set()


@router.get(
    "/chat",
    response_model=ChatSessionListResponse,
    response_model_exclude_none=True,
)
async def list_chat_session_summaries(
    limit: int = Query(default=30, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    settings: AppSettings = Depends(get_settings),
) -> ChatSessionListResponse:
    sessions, total = await list_chat_sessions(
        settings,
        limit=limit,
        offset=offset,
    )
    items = [
        ChatSessionSummary(
            session_id=session["session_id"],
            title=_session_title(session.get("messages")),
            message_count=int(session.get("message_count", 0)),
            created_at=_parse_timestamp(session["created_at"]),
            updated_at=_parse_timestamp(session["updated_at"]),
        )
        for session in sessions
    ]
    return ChatSessionListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    response_model_exclude_none=True,
)
async def create_or_append_chat_message(
    payload: ChatRequest,
    settings: AppSettings = Depends(get_settings),
) -> ChatResponse:
    now = datetime.now(UTC)
    session_id = payload.session_id or str(uuid4())
    user_message = ChatMessage(role="user", content=payload.message.strip())
    try:
        await append_chat_session_message(
            settings,
            session_id=session_id,
            message=user_message.model_dump(exclude_none=True),
            created_at=now.isoformat(),
        )
    except ChatHistoryLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    return ChatResponse(session_id=session_id, created_at=now, message=user_message)


@router.get("/chat/stream")
async def stream_chat_response(
    session_id: str = Query(min_length=1),
    settings: AppSettings = Depends(get_settings),
) -> EventSourceResponse:
    messages = await get_chat_session_messages(settings, session_id=session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    last_message = messages[-1]
    if not isinstance(last_message, dict) or (
        str(last_message.get("role", "user")).strip().lower() != "user"
    ):
        raise HTTPException(
            status_code=409,
            detail="Chat session already has an answer for its latest message.",
        )
    if session_id in _ACTIVE_STREAM_SESSIONS:
        raise HTTPException(
            status_code=409,
            detail="Chat session already has an answer generation in progress.",
        )
    # ponytail: process-local guard; the database compare-and-swap below covers
    # duplicate completions across processes without adding a service lock.
    _ACTIVE_STREAM_SESSIONS.add(session_id)

    async def event_generator():
        history = list(messages)
        model_messages = _to_langchain_messages(history)
        request_id = f"{session_id}:{uuid4().hex}"
        yield {
            "event": "progress",
            "data": _json_payload(
                {
                    "type": "progress",
                    "session_id": session_id,
                    "content": "answering",
                }
            ),
        }

        try:
            graph = None if settings.offline_mode else get_cached_graph(settings)
        except RuntimeError as exc:
            graph = None
            graph_error = str(exc)
        else:
            graph_error = None

        if graph is None:
            try:
                retriever = build_retriever(settings)
            except RuntimeError as exc:
                yield {
                    "event": "stream-error",
                    "data": _json_payload(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "error": str(exc),
                        }
                    ),
                }
                return
            if retriever is None:
                yield {
                    "event": "stream-error",
                    "data": _json_payload(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "error": graph_error or "No index loaded.",
                        }
                    ),
                }
                return

            question = str(history[-1].get("content", "")).strip()
            try:
                documents = await asyncio.to_thread(retriever.invoke, question)
            except Exception:
                logger.exception("Offline retrieval failed for session %s", session_id)
                yield {
                    "event": "stream-error",
                    "data": _json_payload(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "error": "检索失败，回答没有保存，请重试。",
                        }
                    ),
                }
                return
            answer = format_retrieval_only_answer(question, documents)
            evidence = _documents_to_chat_evidence(documents)
            history.append(_assistant_message(answer, evidence))
            if not await _save_history(
                settings,
                session_id=session_id,
                history=history,
            ):
                yield {
                    "event": "stream-error",
                    "data": _json_payload(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "error": "回答保存失败，回答没有保存，请重试。",
                        }
                    ),
                }
                return
            yield {
                "event": "evidence",
                "data": _json_payload(_evidence_payload(session_id, evidence)),
            }
            yield {
                "event": "answer.final",
                "data": _json_payload(
                    {
                        "type": "answer.final",
                        "session_id": session_id,
                        "content": answer,
                        "evidence": _serialize_evidence(evidence),
                    }
                ),
            }
            return

        try:
            result = await asyncio.to_thread(
                graph.invoke,
                {"messages": model_messages},  # type: ignore[arg-type]
                {"configurable": {"thread_id": request_id}},
            )
        except Exception:
            logger.exception("Answer graph failed for session %s", session_id)
            yield {
                "event": "stream-error",
                "data": _json_payload(
                    {
                        "type": "error",
                        "session_id": session_id,
                        "error": "回答生成失败，回答没有保存，请重试。",
                    }
                ),
            }
            return
        final_messages = result.get("messages", []) if isinstance(result, dict) else []
        answer = (
            getattr(final_messages[-1], "content", str(final_messages[-1]))
            if final_messages
            else ""
        )
        if _is_failed_graph_answer(
            result if isinstance(result, dict) else None,
            answer,
        ):
            yield {
                "event": "stream-error",
                "data": _json_payload(
                    {
                        "type": "error",
                        "session_id": session_id,
                        "error": "回答生成失败，回答没有保存，请重试。",
                    }
                ),
            }
            return
        if not answer:
            yield {
                "event": "stream-error",
                "data": _json_payload(
                    {
                        "type": "error",
                        "session_id": session_id,
                        "error": "回答生成失败，回答没有保存，请重试。",
                    }
                ),
            }
            return

        citations = _extract_citations(result if isinstance(result, dict) else None)
        evidence = _extract_chat_evidence(result if isinstance(result, dict) else None)
        history.append(_assistant_message(answer, evidence))
        if not await _save_history(
            settings,
            session_id=session_id,
            history=history,
        ):
            yield {
                "event": "stream-error",
                "data": _json_payload(
                    {
                        "type": "error",
                        "session_id": session_id,
                        "error": "回答保存失败，回答没有保存，请重试。",
                    }
                ),
            }
            return
        yield {
            "event": "evidence",
            "data": _json_payload(
                _evidence_payload(
                    session_id,
                    evidence,
                    citations_markdown=citations,
                )
            ),
        }
        yield {
            "event": "answer.final",
            "data": _json_payload(
                {
                    "type": "answer.final",
                    "session_id": session_id,
                    "content": answer,
                    "evidence": _serialize_evidence(evidence),
                }
            ),
        }

    async def guarded_event_generator():
        try:
            async for event in event_generator():
                yield event
        finally:
            _ACTIVE_STREAM_SESSIONS.discard(session_id)

    return EventSourceResponse(guarded_event_generator())


@router.get(
    "/chat/{session_id}",
    response_model=ChatSessionResponse,
    response_model_exclude_none=True,
)
async def get_chat_session_detail(
    session_id: str,
    settings: AppSettings = Depends(get_settings),
) -> ChatSessionResponse:
    session = await get_chat_session(settings, session_id=session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return ChatSessionResponse.model_validate(session)


def _to_langchain_messages(messages: list[dict[str, Any]]) -> list[Any]:
    converted: list[Any] = []
    for item in messages:
        role = str(item.get("role", "user")).strip().lower()
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if role == "assistant":
            converted.append(AIMessage(content=content))
        elif role == "system":
            converted.append(SystemMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


def _extract_citations(result: dict[str, Any] | None) -> str | None:
    if not isinstance(result, dict):
        return None
    grounded = result.get("groundedAnswer", {})
    if isinstance(grounded, dict) and grounded.get("evidence"):
        return render_grounded_citations(grounded)
    return None


def _extract_chat_evidence(result: dict[str, Any] | None) -> list[ChatEvidence]:
    if not isinstance(result, dict):
        return []
    grounded = result.get("groundedAnswer", {})
    if hasattr(grounded, "model_dump"):
        grounded = grounded.model_dump()
    if not isinstance(grounded, dict):
        return []
    model_evidence = grounded.get("evidence", [])
    retrieval_evidence = _retrieval_evidence(result)
    has_retrieval_source = isinstance(result.get("evidenceGroups"), list) or isinstance(
        result.get("retrievalEvidence"), list
    )
    if not has_retrieval_source:
        # Keep old graph/session payloads readable when no retrieval artifact was
        # persisted. New retrieval answers use the canonical path below.
        return _normalize_evidence_list(model_evidence, trust_paper_id=False)

    if not retrieval_evidence:
        return []

    if not isinstance(model_evidence, list) or not model_evidence:
        return retrieval_evidence

    canonical_by_node = {item.node_id: item for item in retrieval_evidence}
    canonical_by_quote = {item.quote: item for item in retrieval_evidence}
    normalized: list[ChatEvidence] = []
    for raw_item in model_evidence:
        if hasattr(raw_item, "model_dump"):
            raw_item = raw_item.model_dump()
        if not isinstance(raw_item, dict):
            continue
        node_id = _optional_text(
            raw_item.get("node_id")
            or raw_item.get("passage_id")
            or raw_item.get("id")
        )
        quote = _truncate_quote(
            _optional_text(raw_item.get("quote") or raw_item.get("quote_text"))
        )
        canonical = (
            canonical_by_node.get(node_id)
            if node_id
            else canonical_by_quote.get(quote)
        )
        if canonical is None and quote:
            canonical = canonical_by_quote.get(quote)
        if canonical is None:
            # Do not persist evidence (and especially paper IDs) invented by the
            # answer model when it cannot be tied back to retrieved metadata.
            continue
        normalized.append(canonical)
    return _dedupe_evidence(normalized)


def _retrieval_evidence(result: dict[str, Any]) -> list[ChatEvidence]:
    """Return source-owned evidence, never fields invented by the answer model."""
    raw_items: list[object] = []
    groups = result.get("evidenceGroups", [])
    if isinstance(groups, list):
        for group in groups:
            if hasattr(group, "model_dump"):
                group = group.model_dump()
            if isinstance(group, dict):
                raw_items.extend(_evidence_values(group.get("evidence")))

    artifacts = result.get("retrievalEvidence", [])
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if hasattr(artifact, "model_dump"):
                artifact = artifact.model_dump()
            if isinstance(artifact, dict):
                raw_items.extend(_evidence_values(artifact.get("evidence")))

    return _merge_canonical_evidence(_normalize_evidence_list(raw_items))


def _evidence_values(value: object) -> list[object]:
    if not isinstance(value, list):
        return []
    return value


def _merge_canonical_evidence(items: list[ChatEvidence]) -> list[ChatEvidence]:
    """Merge duplicate retrieval records without discarding a catalog ID."""
    merged: dict[str, ChatEvidence] = {}
    for item in items:
        existing = merged.get(item.node_id)
        if existing is None:
            merged[item.node_id] = item
            continue
        updates: dict[str, Any] = {}
        for field in (
            "paper_id",
            "paper_title",
            "page",
            "score",
            "relevance",
        ):
            if getattr(existing, field) is None and getattr(item, field) is not None:
                updates[field] = getattr(item, field)
        if updates:
            merged[item.node_id] = existing.model_copy(update=updates)
    return list(merged.values())


def _is_failed_graph_answer(
    result: dict[str, Any] | None,
    answer: object,
) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("answerGenerationFailed") is True:
        return True

    normalized_answer = str(answer).strip()
    if normalized_answer == "No answers were generated.":
        return True

    if result.get("routingDecision") != "retrieve":
        return False
    evidence_groups = result.get("evidenceGroups")
    return (
        isinstance(evidence_groups, list)
        and not evidence_groups
        and not _extract_chat_evidence(result)
    )


def _documents_to_chat_evidence(documents: list[Any]) -> list[ChatEvidence]:
    normalized: list[ChatEvidence] = []
    for document in documents:
        metadata = dict(getattr(document, "metadata", {}) or {})
        normalized_item = _normalize_evidence_item(
            {
                "node_id": metadata.get("node_id")
                or metadata.get("passage_id")
                or metadata.get("id"),
                "paper_id": metadata.get("paper_id"),
                "paper_title": metadata.get("paper_title"),
                "source": metadata.get("source"),
                "section_path": metadata.get("section_path")
                or metadata.get("title_path")
                or [],
                "page": metadata.get("page") or metadata.get("page_start"),
                "quote": metadata.get("quote_text")
                or getattr(document, "page_content", ""),
                "score": metadata.get("score"),
                "relevance": metadata.get("relevance"),
            }
        )
        if normalized_item is not None:
            normalized.append(normalized_item)
    return _dedupe_evidence(normalized)


def _normalize_evidence_list(
    raw_items: object,
    *,
    trust_paper_id: bool = True,
) -> list[ChatEvidence]:
    if not isinstance(raw_items, list):
        return []
    normalized: list[ChatEvidence] = []
    for item in raw_items:
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if not isinstance(item, dict):
            continue
        evidence = _normalize_evidence_item(item, trust_paper_id=trust_paper_id)
        if evidence is not None:
            normalized.append(evidence)
    return _dedupe_evidence(normalized)


def _normalize_evidence_item(
    item: dict[str, Any],
    *,
    trust_paper_id: bool = True,
) -> ChatEvidence | None:
    node_id = _optional_text(
        item.get("node_id") or item.get("passage_id") or item.get("id")
    )
    quote = _truncate_quote(
        _optional_text(item.get("quote") or item.get("quote_text"))
    )
    if not node_id or not quote:
        return None

    section_path = item.get("section_path") or item.get("title_path") or []
    if isinstance(section_path, str):
        section_path = [section_path]
    if not isinstance(section_path, list):
        section_path = []

    page_value = item.get("page") or item.get("page_start")
    page = page_value if type(page_value) is int and page_value > 0 else None
    score_value = item.get("score")
    score = (
        float(score_value)
        if isinstance(score_value, int | float) and not isinstance(score_value, bool)
        else None
    )
    relevance = _optional_text(item.get("relevance"))
    paper_id = _optional_text(item.get("paper_id")) if trust_paper_id else None
    paper_title = _optional_text(item.get("paper_title"))

    return ChatEvidence(
        node_id=node_id,
        paper_id=paper_id,
        paper_title=paper_title,
        source=_source_label(item.get("source")),
        section_path=[
            str(part).strip() for part in section_path if str(part).strip()
        ],
        page=page,
        quote=quote,
        score=score,
        relevance=relevance,
    )


def _dedupe_evidence(items: list[ChatEvidence]) -> list[ChatEvidence]:
    seen: set[str] = set()
    unique: list[ChatEvidence] = []
    for item in items:
        if item.node_id in seen:
            continue
        seen.add(item.node_id)
        unique.append(item)
    return unique


def _assistant_message(answer: str, evidence: list[ChatEvidence]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": answer,
        "evidence": _serialize_evidence(evidence),
    }


def _serialize_evidence(items: list[ChatEvidence]) -> list[dict[str, Any]]:
    return [item.model_dump() for item in items]


def _evidence_payload(
    session_id: str,
    evidence: list[ChatEvidence],
    *,
    citations_markdown: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "evidence",
        "session_id": session_id,
        "evidence": _serialize_evidence(evidence),
    }
    if citations_markdown:
        payload["citations_markdown"] = citations_markdown
    return payload


async def _save_history(
    settings: AppSettings,
    *,
    session_id: str,
    history: list[dict[str, Any]],
) -> bool:
    if not history:
        return False
    try:
        return await append_chat_session_message(
            settings,
            session_id=session_id,
            message=history[-1],
            created_at=datetime.now(UTC).isoformat(),
            expected_messages=history[:-1],
        )
    except Exception:
        logger.exception("Failed to persist chat session %s", session_id)
        return False


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _truncate_quote(value: str | None, limit: int = 400) -> str:
    if not value:
        return ""
    normalized = value.strip()
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def _source_label(value: object) -> str:
    source = _optional_text(value)
    if not source:
        return "未知来源"
    return PurePosixPath(source.replace("\\", "/")).name or "未知来源"


def _session_title(messages: object, *, limit: int = 60) -> str:
    if not isinstance(messages, list):
        return "未命名会话"
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        normalized = re.sub(r"\s+", " ", content).strip()
        if normalized:
            return normalized[:limit]
    return "未命名会话"


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _json_payload(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)
