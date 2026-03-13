from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import get_out_of_scope_prompt
from agent.schemas import OutOfScopeResponse
from agent.states import GraphState
from core.rag_answer import render_out_of_scope_answer
from llms.llm import get_llm_by_type


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
    try:
        structured_llm = llm.with_config(temperature=0).with_structured_output(
            OutOfScopeResponse
        )
        response = structured_llm.invoke(
            [
                SystemMessage(content=get_out_of_scope_prompt()),
                HumanMessage(content=user_input),
            ]
        )
        payload = response.model_dump()
        content = render_out_of_scope_answer(payload)
    except Exception:
        payload = {
            "reason": "This question appears to fall outside the current knowledge base.",
            "boundary": corpus_profile or "No knowledge-base profile is currently available.",
            "suggestion": "Try asking about topics explicitly covered by the uploaded documents.",
            "next_action": "Upload materials related to this topic if you want grounded answers here.",
        }
        content = render_out_of_scope_answer(payload)
    return {"messages": [AIMessage(content=content)]}
