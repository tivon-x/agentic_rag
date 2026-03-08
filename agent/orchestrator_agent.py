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

from agent.prompts import get_fallback_response_prompt, get_orchestrator_prompt
from agent.states import OrchestratorState
from core.config import KEEP_MESSAGES, MAX_CONTEXT_TOKENS, MAX_ITERATIONS, MAX_TOOL_CALLS
from llms.llm import get_llm_by_type


class FallbackMiddleware(AgentMiddleware):
    """
    This middleware monitors agent iteration and tool call counts to prevent infinite loops or excessive tool reliance.

    When preset iteration or tool call limits are reached, it triggers a fallback mechanism that generates a final answer,
    informing the user that the agent cannot continue and providing previous research information for manual follow-up.
    """

    def __init__(self, model: BaseChatModel, max_iterations: int, max_tool_calls: int):
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls

        self.iteration_count = 0
        self.tool_call_count = 0

        self.model = model

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        self.iteration_count += 1
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        self.iteration_count += 1
        return await handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        self.tool_call_count += 1
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        self.tool_call_count += 1
        return await handler(request)

    @hook_config(can_jump_to=["end"])
    def after_agent(
        self, state: AgentState[Any], runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        if (
            self.iteration_count >= self.max_iterations
            or self.tool_call_count > self.max_tool_calls
        ):
            messages = state.get("messages", [])
            formatted_messages = get_buffer_string(messages)
            user_query = state.get("question")

            prompt_content = (
                "The agent has reached its iteration or tool call limit and cannot continue.\n\n"
                f"User Query: {user_query}\n\n"
                f"Conversation History:\n{formatted_messages}\n\n"
                f"INSTRUCTION:\nProvide the best possible answer using only the data above."
            )

            response = self.model.invoke(
                [
                    SystemMessage(content=get_fallback_response_prompt()),
                    HumanMessage(content=prompt_content),
                ]
            )

            return {"messages": [response]}

        return None

    @hook_config(can_jump_to=["end"])
    async def aafter_agent(
        self, state: AgentState[Any], runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        if (
            self.iteration_count >= self.max_iterations
            or self.tool_call_count > self.max_tool_calls
        ):
            messages = state.get("messages", [])
            formatted_messages = get_buffer_string(messages)
            user_query = state.get("question")

            prompt_content = (
                "The agent has reached its iteration or tool call limit and cannot continue.\n\n"
                f"User Query: {user_query}\n\n"
                f"Conversation History:\n{formatted_messages}\n\n"
                f"INSTRUCTION:\nProvide the best possible answer using only the data above."
            )

            response = await self.model.ainvoke(
                [
                    SystemMessage(content=get_fallback_response_prompt()),
                    HumanMessage(content=prompt_content),
                ]
            )

            return {"messages": [response]}

        return None


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


def create_orchestrator_agent(tools):
    llm = get_llm_by_type("orchestrate")

    summarization_middleware = SummarizationMiddleware(
        model=llm,
        trigger=("tokens", MAX_CONTEXT_TOKENS),
        keep=("messages", KEEP_MESSAGES),
    )
    fallback_middleware = FallbackMiddleware(
        model=llm, max_iterations=MAX_ITERATIONS, max_tool_calls=MAX_TOOL_CALLS
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=get_orchestrator_prompt(),
        middleware=[summarization_middleware, fallback_middleware, collect_answer],
        state_schema=OrchestratorState,
    )

    return agent
