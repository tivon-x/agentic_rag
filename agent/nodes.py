from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage

from agent.prompts import (
    get_aggregation_prompt,
    get_conversation_summary_prompt,
    get_rewrite_query_prompt,
)
from llms.llm import get_llm_by_type

from .schemas import QueryAnalysis
from .states import GraphState


def summarize_history(state: GraphState):
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}

    relevant_msgs = [
        msg
        for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage))
        and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}

    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    llm = get_llm_by_type("summarize_history")
    summary_response = llm.with_config(temperature=0.2).invoke(
        [
            SystemMessage(content=get_conversation_summary_prompt()),
            HumanMessage(content=conversation),
        ]
    )
    return {
        "conversation_summary": summary_response.content,
        "agent_answers": [{"__reset__": True}],
    }


def rewrite_query(state: GraphState):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (
        f"Conversation Context:\n{conversation_summary}\n"
        if conversation_summary.strip()
        else ""
    ) + f"User Query:\n{last_message.content}\n"

    llm = get_llm_by_type("rewrite_query")
    llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(
        QueryAnalysis
    )

    try:
        response = llm_with_structure.invoke(
            [
                SystemMessage(content=get_rewrite_query_prompt()),
                HumanMessage(content=context_section),
            ]
        )
    except Exception:
        # Some OpenAI-compatible endpoints don't support JSON mode / structured outputs.
        # Fall back to a deterministic rewrite that keeps retrieval functional.
        return {
            "questionIsClear": True,
            "messages": [],
            "originalQuery": last_message.content,
            "rewrittenQuestions": [last_message.content],
        }

    if response.questions and response.is_clear:
        delete_all = []
        for m in state["messages"]:
            if isinstance(m, SystemMessage):
                continue
            msg_id = getattr(m, "id", None)
            if isinstance(msg_id, str) and msg_id:
                delete_all.append(RemoveMessage(id=msg_id))
        return {
            "questionIsClear": True,
            "messages": delete_all,
            "originalQuery": last_message.content,
            "rewrittenQuestions": response.questions,
        }

    clarification = (
        response.clarification_needed
        if response.clarification_needed
        and len(response.clarification_needed.strip()) > 10
        else "I need more information to understand your question."
    )
    return {"questionIsClear": False, "messages": [AIMessage(content=clarification)]}


def request_clarification(state: GraphState):
    return {}


def aggregate_answers(state: GraphState):
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += f"\nAnswer {i}:\n{ans['answer']}\n"

    user_message = HumanMessage(
        content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}"""
    )
    llm = get_llm_by_type("aggregate_answers")
    synthesis_response = llm.invoke(
        [SystemMessage(content=get_aggregation_prompt()), user_message]
    )
    return {"messages": [AIMessage(content=synthesis_response.content)]}
