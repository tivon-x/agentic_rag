from __future__ import annotations

import asyncio
from datetime import UTC, datetime
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


@router.post("/chat", response_model=ChatResponse)
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
            messages=[user_message.model_dump()],
            created_at=now.isoformat(),
        )
    else:
        messages.append(user_message.model_dump())
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
            documents = await asyncio.to_thread(retriever.invoke, question)
            answer = format_retrieval_only_answer(question, documents)
            history.append({"role": "assistant", "content": answer})
            await upsert_chat_session_messages(
                settings,
                session_id=session_id,
                messages=history,
                updated_at=datetime.now(UTC).isoformat(),
            )
            yield {
                "event": "answer.final",
                "data": _json_payload(
                    {
                        "type": "answer.final",
                        "session_id": session_id,
                        "content": answer,
                    }
                ),
            }
            return

        result = await asyncio.to_thread(
            graph.invoke,
            {"messages": model_messages},  # type: ignore[arg-type]
            {"configurable": {"thread_id": request_id}},
        )
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
        history.append({"role": "assistant", "content": answer})
        await upsert_chat_session_messages(
            settings,
            session_id=session_id,
            messages=history,
            updated_at=datetime.now(UTC).isoformat(),
        )
        if citations:
            yield {
                "event": "evidence",
                "data": _json_payload(
                    {
                        "type": "evidence",
                        "session_id": session_id,
                        "citations_markdown": citations,
                    }
                ),
            }
        yield {
            "event": "answer.final",
            "data": _json_payload(
                {
                    "type": "answer.final",
                    "session_id": session_id,
                    "content": answer,
                }
            ),
        }

    return EventSourceResponse(event_generator())


@router.get("/chat/{session_id}", response_model=ChatSessionResponse)
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


def _json_payload(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)
