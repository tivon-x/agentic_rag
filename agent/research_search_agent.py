from collections.abc import Awaitable
from typing import Any, Callable

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    SummarizationMiddleware,
    after_agent,
    hook_config,
)
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
)
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.prompts import get_fallback_response_prompt, get_research_search_prompt
from agent.schemas import EvidenceGroup
from agent.states import ResearchSearchState
from llms.llm import get_llm_by_type


class QueryPlanMiddleware(AgentMiddleware):
    """Expose graph-level query planning to tool calls via ToolFactory context."""

    def __init__(self, tool_factory):
        self.tool_factory = tool_factory

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        token = self.tool_factory.set_active_query_plan(
            request.state.get("query_plan", {})
        )
        try:
            return handler(request)
        finally:
            self.tool_factory.reset_active_query_plan(token)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        token = self.tool_factory.set_active_query_plan(
            request.state.get("query_plan", {})
        )
        try:
            return await handler(request)
        finally:
            self.tool_factory.reset_active_query_plan(token)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        state = request.state if isinstance(request.state, dict) else {}
        token = self.tool_factory.set_active_query_plan(state.get("query_plan", {}))
        try:
            return handler(request)
        finally:
            self.tool_factory.reset_active_query_plan(token)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        state = request.state if isinstance(request.state, dict) else {}
        token = self.tool_factory.set_active_query_plan(state.get("query_plan", {}))
        try:
            return await handler(request)
        finally:
            self.tool_factory.reset_active_query_plan(token)


class EvidenceCaptureMiddleware(AgentMiddleware):
    """Capture structured retrieval artifacts from tool calls into agent state."""

    def _command_from_tool_message(
        self, request: ToolCallRequest, response: ToolMessage
    ) -> Command[Any] | ToolMessage:
        if request.tool_call.get("name") != "search_relevant_chunks":
            return response

        artifact = response.artifact if isinstance(response.artifact, dict) else None
        if artifact is None:
            return response

        query_plan = artifact.get("query_plan", {}) or {}
        evidence_group = EvidenceGroup(
            subquery=str(artifact.get("subquery", "")).strip(),
            intent=str(query_plan.get("intent", "fact")).strip() or "fact",
            packed_context=dict(artifact.get("packed_context", {}) or {}),
            evidence=list(artifact.get("evidence", []) or []),
            debug=dict(artifact.get("debug", {}) or {}),
        )
        return Command(
            update={
                "messages": [response],
                "retrievalEvidence": [dict(artifact)],
                "packedContexts": [
                    {
                        "subquery": evidence_group.subquery,
                        **evidence_group.packed_context,
                    }
                ],
                "evidenceGroups": [evidence_group.model_dump()],
            }
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        response = handler(request)
        if isinstance(response, ToolMessage):
            return self._command_from_tool_message(request, response)
        return response

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        response = await handler(request)
        if isinstance(response, ToolMessage):
            return self._command_from_tool_message(request, response)
        return response


class FallbackMiddleware(AgentMiddleware):
    """Guardrails for iteration/tool-call limits with a fallback final answer."""

    state_schema = ResearchSearchState

    def __init__(self, model: BaseChatModel, max_iterations: int, max_tool_calls: int):
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.model = model

    @staticmethod
    def _last_ai_message(state: AgentState[Any]) -> AIMessage | None:
        for message in reversed(state.get("messages", [])):
            if isinstance(message, AIMessage):
                return message
        return None

    @staticmethod
    def _current_count(state: AgentState[Any], key: str) -> int:
        return int(state.get(key, 0) or 0)

    def _fallback_prompt(self, state: AgentState[Any]) -> list[SystemMessage | HumanMessage]:
        messages = state.get("messages", [])
        formatted_messages = get_buffer_string(messages)
        user_query = state.get("question")
        prompt_content = (
            "The agent has reached its iteration or tool call limit and cannot continue.\n\n"
            f"User Query: {user_query}\n\n"
            f"Conversation History:\n{formatted_messages}\n\n"
            "INSTRUCTION:\nProvide the best possible answer using only the data above."
        )
        return [
            SystemMessage(content=get_fallback_response_prompt()),
            HumanMessage(content=prompt_content),
        ]

    @hook_config(can_jump_to=["end"])
    def before_model(
        self, state: ResearchSearchState, runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        if self._current_count(state, "iteration_count") < self.max_iterations:
            return None
        return {
            "jump_to": "end",
            "messages": [self.model.invoke(self._fallback_prompt(state))],
        }

    @hook_config(can_jump_to=["end"])
    def after_model(
        self, state: ResearchSearchState, runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        last_ai_message = self._last_ai_message(state)
        tool_calls = list(last_ai_message.tool_calls) if last_ai_message else []
        next_iteration_count = self._current_count(state, "iteration_count") + 1
        next_tool_call_count = self._current_count(state, "tool_call_count") + len(tool_calls)
        updates: dict[str, Any] = {"iteration_count": 1}

        if tool_calls and (
            next_iteration_count >= self.max_iterations
            or next_tool_call_count > self.max_tool_calls
        ):
            updates.update(
                {
                    "jump_to": "end",
                    "messages": [self.model.invoke(self._fallback_prompt(state))],
                }
            )
            return updates

        updates["tool_call_count"] = len(tool_calls)
        return updates

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: ResearchSearchState, runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        if self._current_count(state, "iteration_count") < self.max_iterations:
            return None
        return {
            "jump_to": "end",
            "messages": [await self.model.ainvoke(self._fallback_prompt(state))],
        }

    @hook_config(can_jump_to=["end"])
    async def aafter_model(
        self, state: ResearchSearchState, runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        last_ai_message = self._last_ai_message(state)
        tool_calls = list(last_ai_message.tool_calls) if last_ai_message else []
        next_iteration_count = self._current_count(state, "iteration_count") + 1
        next_tool_call_count = self._current_count(state, "tool_call_count") + len(tool_calls)
        updates: dict[str, Any] = {"iteration_count": 1}

        if tool_calls and (
            next_iteration_count >= self.max_iterations
            or next_tool_call_count > self.max_tool_calls
        ):
            updates.update(
                {
                    "jump_to": "end",
                    "messages": [await self.model.ainvoke(self._fallback_prompt(state))],
                }
            )
            return updates

        updates["tool_call_count"] = len(tool_calls)
        return updates


@after_agent
def collect_answer(state: AgentState, runtime: Runtime) -> dict | None:
    last_message = state["messages"][-1]
    is_valid = (
        isinstance(last_message, AIMessage)
        and last_message.content
        and not last_message.tool_calls
    )
    answer = last_message.content if is_valid else "Unable to generate an answer."
    return {
        "final_answer": answer,
        "agent_answers": [
            {
                "index": state.get("question_index", 0),
                "question": state.get("question", ""),
                "answer": answer,
            }
        ],
    }


def create_research_search_agent(
    tools,
    tool_factory=None,
    *,
    max_context_tokens: int = 5000,
    keep_messages: int = 20,
    max_iterations: int = 10,
    max_tool_calls: int = 8,
):
    llm = get_llm_by_type("research_search")

    summarization_middleware = SummarizationMiddleware(
        model=llm,
        trigger=("tokens", max_context_tokens),
        keep=("messages", keep_messages),
    )
    fallback_middleware = FallbackMiddleware(
        model=llm,
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
    )
    middleware = [summarization_middleware, fallback_middleware, EvidenceCaptureMiddleware()]
    if tool_factory is not None:
        middleware.insert(0, QueryPlanMiddleware(tool_factory))

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=get_research_search_prompt(),
        middleware=[*middleware, collect_answer],
        state_schema=ResearchSearchState,
    )
