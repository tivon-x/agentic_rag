from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sse_starlette.sse import EventSourceResponse

from api.db.database import (
    create_chat_session,
    get_chat_session,
    get_chat_session_messages,
    upsert_chat_session_messages,
)
from api.dependencies import get_settings
from api.models.chat import (
    ChatEvidence,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatSessionResponse,
)
from api.services.graph_cache import get_cached_graph
from core.factory import build_retriever
from core.rag_answer import format_retrieval_only_answer, render_grounded_citations
from core.settings import AppSettings


router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


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
    messages = await get_chat_session_messages(settings, session_id=session_id)

    if messages is None:
        await create_chat_session(
            settings,
            session_id=session_id,
            messages=[user_message.model_dump(exclude_none=True)],
            created_at=now.isoformat(),
        )
    else:
        messages.append(user_message.model_dump(exclude_none=True))
        await upsert_chat_session_messages(
            settings,
            session_id=session_id,
            messages=messages,
            updated_at=now.isoformat(),
        )

    return ChatResponse(session_id=session_id, created_at=now, message=user_message)


@router.get("/chat/stream")
async def stream_chat_response(
    session_id: str = Query(min_length=1),
    settings: AppSettings = Depends(get_settings),
) -> EventSourceResponse:
    messages = await get_chat_session_messages(settings, session_id=session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    async def event_generator():
        history = list(messages)
        model_messages = _to_langchain_messages(history)
        request_id = f"{session_id}:{len(history)}"
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
                            "error": "回答保存失败，请重试。",
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
        if not answer:
            yield {
                "event": "stream-error",
                "data": _json_payload(
                    {
                        "type": "error",
                        "session_id": session_id,
                        "error": "The answer graph returned no final answer.",
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
                        "error": "回答保存失败，请重试。",
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

    return EventSourceResponse(event_generator())


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
    return _normalize_evidence_list(grounded.get("evidence", []))


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


def _normalize_evidence_list(raw_items: object) -> list[ChatEvidence]:
    if not isinstance(raw_items, list):
        return []
    normalized: list[ChatEvidence] = []
    for item in raw_items:
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if not isinstance(item, dict):
            continue
        evidence = _normalize_evidence_item(item)
        if evidence is not None:
            normalized.append(evidence)
    return _dedupe_evidence(normalized)


def _normalize_evidence_item(item: dict[str, Any]) -> ChatEvidence | None:
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
    paper_id = _optional_text(item.get("paper_id"))
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
    try:
        await upsert_chat_session_messages(
            settings,
            session_id=session_id,
            messages=history,
            updated_at=datetime.now(UTC).isoformat(),
        )
    except Exception:
        logger.exception("Failed to persist chat session %s", session_id)
        return False
    return True


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


def _json_payload(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)
