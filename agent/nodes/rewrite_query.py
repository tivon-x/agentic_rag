from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import get_rewrite_query_prompt
from agent.schemas import QueryAnalysis
from agent.states import GraphState
from core.corpus_profile import expand_queries_with_corpus_profile
from llms.llm import get_llm_by_type


def rewrite_query(state: GraphState):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")
    query_plan = state.get("queryPlan", {})
    corpus_profile = state.get("corpusProfile", "")
    corpus_profile_data = state.get("corpusProfileData", {})
    seed_queries = [
        str(item).strip()
        for item in query_plan.get("subqueries", [])
        if str(item).strip()
    ]
    if not seed_queries:
        seed_queries = [last_message.content]

    llm = get_llm_by_type("rewrite_query")
    llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(
        QueryAnalysis
    )

    questions: list[str] = []
    for seed_query in seed_queries[:3]:
        sections = []
        if corpus_profile.strip():
            sections.append(f"Knowledge Base Profile:\n{corpus_profile}")
        if conversation_summary.strip():
            sections.append(f"Conversation Context:\n{conversation_summary}")
        sections.append(f"User Query:\n{seed_query}")
        context_section = "\n\n".join(sections) + "\n"
        try:
            response = llm_with_structure.invoke(
                [
                    SystemMessage(content=get_rewrite_query_prompt()),
                    HumanMessage(content=context_section),
                ]
            )
            questions.extend(q.strip() for q in response.questions if q and q.strip())
        except Exception:
            questions.append(seed_query)

    if not questions:
        questions = [last_message.content]

    questions = expand_queries_with_corpus_profile(
        questions,
        original_query=str(last_message.content),
        query_plan=query_plan,
        profile=corpus_profile_data,
    )

    return {
        "rewrittenQuestions": list(dict.fromkeys(questions))[:3],
        "originalQuery": last_message.content,
    }
