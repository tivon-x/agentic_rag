from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import get_plan_query_prompt
from agent.schemas import QueryPlan
from agent.states import GraphState
from core.corpus_profile import apply_profile_query_plan_prior
from llms.llm import get_llm_by_type


def plan_query(state: GraphState):
    original_query = state.get("originalQuery") or state["messages"][-1].content
    conversation_summary = state.get("conversation_summary", "")
    corpus_profile = state.get("corpusProfile", "")
    corpus_profile_data = state.get("corpusProfileData", {})

    sections = []
    if corpus_profile.strip():
        sections.append(f"Knowledge Base Profile:\n{corpus_profile}")
    if conversation_summary.strip():
        sections.append(f"Conversation Summary:\n{conversation_summary}")
    sections.append(f"Latest User Message:\n{original_query}")
    planner_input = "\n\n".join(sections)

    try:
        llm = get_llm_by_type("plan_query")
        structured_llm = llm.with_config(temperature=0).with_structured_output(
            QueryPlan
        )
        plan = structured_llm.invoke(
            [
                SystemMessage(content=get_plan_query_prompt()),
                HumanMessage(content=planner_input),
            ]
        )
    except Exception:
        plan = QueryPlan(
            intent="fact",
            subqueries=[original_query],
            preferred_node_types=["paragraph"],
        )

    return {
        "queryPlan": apply_profile_query_plan_prior(
            plan.model_dump(),
            original_query=str(original_query),
            profile=corpus_profile_data,
        ),
        "originalQuery": original_query,
    }
