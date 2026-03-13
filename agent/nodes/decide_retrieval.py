from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import get_retrieval_decision_prompt
from agent.schemas import RetrievalDecision
from agent.states import GraphState
from core.corpus_profile import analyze_corpus_profile_match
from llms.llm import get_llm_by_type


def decide_retrieval(state: GraphState):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")
    corpus_profile = state.get("corpusProfile", "")
    corpus_profile_data = state.get("corpusProfileData", {})
    profile_match = analyze_corpus_profile_match(
        str(last_message.content),
        corpus_profile_data,
    )

    if profile_match["force_out_of_scope"]:
        return {
            "routingDecision": "out_of_scope",
            "routingReason": profile_match["reason"]
            or "The query matches the corpus profile's explicit non-coverage boundary.",
            "originalQuery": last_message.content,
        }

    sections = []
    if corpus_profile.strip():
        sections.append(f"Knowledge Base Profile:\n{corpus_profile}")
    if profile_match["reason"]:
        sections.append(f"Corpus Profile Prior:\n{profile_match['reason']}")
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
