from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agent.research_search_agent import FallbackMiddleware
from api.models.corpus import CorpusProfile
from api.routers import corpus as corpus_router


def _state(
    assistant_message: AIMessage,
    *,
    iteration_count: int = 0,
    tool_call_count: int = 0,
) -> dict:
    return {
        "messages": [HumanMessage(content="question"), assistant_message],
        "question": "question",
        "iteration_count": iteration_count,
        "tool_call_count": tool_call_count,
    }


def _tool_call(call_id: str = "call-1") -> dict:
    return {
        "name": "search_relevant_chunks",
        "args": {"query": "question"},
        "id": call_id,
        "type": "tool_call",
    }


def test_fallback_counts_are_kept_in_invocation_state() -> None:
    model = FakeMessagesListChatModel(responses=[AIMessage(content="fallback")])
    middleware = FallbackMiddleware(model, max_iterations=3, max_tool_calls=3)

    first_update = middleware.after_model(_state(AIMessage(content="first")), None)
    second_update = middleware.after_model(_state(AIMessage(content="second")), None)

    assert first_update == {"iteration_count": 1, "tool_call_count": 0}
    assert second_update == {"iteration_count": 1, "tool_call_count": 0}
    assert not hasattr(middleware, "iteration_count")
    assert not hasattr(middleware, "tool_call_count")


def test_fallback_jumps_before_executing_over_budget_tool_calls() -> None:
    model = FakeMessagesListChatModel(responses=[AIMessage(content="fallback")])
    middleware = FallbackMiddleware(model, max_iterations=10, max_tool_calls=1)

    update = middleware.after_model(
        _state(AIMessage(content="", tool_calls=[_tool_call()],), tool_call_count=1),
        None,
    )

    assert update["jump_to"] == "end"
    assert update["iteration_count"] == 1
    assert "tool_call_count" not in update
    assert update["messages"][-1].content == "fallback"


def test_fallback_jumps_before_executing_tools_after_iteration_limit() -> None:
    model = FakeMessagesListChatModel(responses=[AIMessage(content="fallback")])
    middleware = FallbackMiddleware(model, max_iterations=1, max_tool_calls=10)

    update = middleware.after_model(_state(AIMessage(content="", tool_calls=[_tool_call()])), None)

    assert update["jump_to"] == "end"
    assert update["iteration_count"] == 1
    assert update["messages"][-1].content == "fallback"


def test_fallback_async_counts_are_kept_in_invocation_state() -> None:
    model = FakeMessagesListChatModel(responses=[AIMessage(content="fallback")])
    middleware = FallbackMiddleware(model, max_iterations=3, max_tool_calls=3)

    async def run() -> tuple[dict, dict]:
        first = await middleware.aafter_model(_state(AIMessage(content="first")), None)
        second = await middleware.aafter_model(_state(AIMessage(content="second")), None)
        return first, second

    first_update, second_update = asyncio.run(run())

    assert first_update == {"iteration_count": 1, "tool_call_count": 0}
    assert second_update == {"iteration_count": 1, "tool_call_count": 0}


def test_update_corpus_profile_invalidates_graph_cache(monkeypatch, tmp_path) -> None:
    profile = CorpusProfile(name="updated corpus")
    invalidations: list[bool] = []

    monkeypatch.setattr(corpus_router, "save_corpus_profile", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        corpus_router,
        "load_corpus_profile",
        lambda _index_dir: profile.model_dump(),
    )
    monkeypatch.setattr(
        corpus_router,
        "invalidate_graph_cache",
        lambda: invalidations.append(True),
    )

    result = asyncio.run(
        corpus_router.update_corpus_profile(
            profile,
            SimpleNamespace(index_dir=tmp_path),
        )
    )

    assert result == profile
    assert invalidations == [True]
