from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import (
    get_aggregation_prompt,
    get_conversation_summary_prompt,
    get_direct_answer_prompt,
    get_out_of_scope_prompt,
    get_retrieval_decision_prompt,
    get_rewrite_query_prompt,
)
from llms.llm import get_llm_by_type

from .schemas import QueryAnalysis, RetrievalDecision
from .states import GraphState



def inject_corpus_profile(corpus_profile: str):
    def _inject(_: GraphState):
        return {"corpusProfile": corpus_profile}

    return _inject



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



def decide_retrieval(state: GraphState):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")
    corpus_profile = state.get("corpusProfile", "")

    sections = []
    if corpus_profile.strip():
        sections.append(f"Knowledge Base Profile:\n{corpus_profile}")
    if conversation_summary.strip():
        sections.append(f"Conversation Summary:\n{conversation_summary}")
    sections.append(f"Latest User Message:\n{last_message.content}")
    decision_input = "\n\n".join(sections) + "\n"

    llm = get_llm_by_type("decide_retrieval")
    structured_llm = llm.with_config(temperature=0).with_structured_output(
        RetrievalDecision
    )

    try:
        response = structured_llm.invoke(
            [
                SystemMessage(content=get_retrieval_decision_prompt()),
                HumanMessage(content=decision_input),
            ]
        )
        decision = response.decision
        reason = response.reason.strip()
    except Exception:
        decision = "retrieve"
        reason = "Fallback to retrieval because the routing decision could not be parsed."

    return {
        "routingDecision": decision,
        "routingReason": reason,
        "originalQuery": last_message.content,
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
        questions = [q.strip() for q in response.questions if q and q.strip()]
    except Exception:
        questions = []

    if not questions:
        questions = [last_message.content]

    return {
        "rewrittenQuestions": questions[:3],
        "originalQuery": last_message.content,
    }



def direct_answer(state: GraphState):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    user_input = (
        f"Conversation Summary:\n{conversation_summary}\n\n"
        if conversation_summary.strip()
        else ""
    ) + f"Latest User Message:\n{last_message.content}"

    llm = get_llm_by_type("direct_answer")
    response = llm.invoke(
        [
            SystemMessage(content=get_direct_answer_prompt()),
            HumanMessage(content=user_input),
        ]
    )
    return {"messages": [AIMessage(content=response.content)]}



def out_of_scope_answer(state: GraphState):
    last_message = state["messages"][-1]
    corpus_profile = state.get("corpusProfile", "")
    routing_reason = state.get("routingReason", "")

    sections = []
    if corpus_profile.strip():
        sections.append(f"Knowledge Base Profile:\n{corpus_profile}")
    if routing_reason.strip():
        sections.append(f"Routing Reason:\n{routing_reason}")
    sections.append(f"Latest User Message:\n{last_message.content}")
    user_input = "\n\n".join(sections)

    llm = get_llm_by_type("out_of_scope_answer")
    response = llm.invoke(
        [
            SystemMessage(content=get_out_of_scope_prompt()),
            HumanMessage(content=user_input),
        ]
    )
    return {"messages": [AIMessage(content=response.content)]}



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
