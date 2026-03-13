from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import get_conversation_summary_prompt
from agent.states import GraphState
from llms.llm import get_llm_by_type


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
        "retrievalEvidence": [{"__reset__": True}],
        "packedContexts": [{"__reset__": True}],
        "evidenceGroups": [{"__reset__": True}],
        "groundedAnswer": {},
    }
