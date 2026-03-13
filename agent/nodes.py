from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import (
    get_aggregation_prompt,
    get_conversation_summary_prompt,
    get_direct_answer_prompt,
    get_out_of_scope_prompt,
    get_plan_query_prompt,
    get_retrieval_decision_prompt,
    get_rewrite_query_prompt,
)
from core.rag_answer import render_grounded_answer, render_out_of_scope_answer
from llms.llm import get_llm_by_type

from .schemas import GroundedAnswer, OutOfScopeResponse, QueryAnalysis, QueryPlan, RetrievalDecision
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
        "retrievalEvidence": [{"__reset__": True}],
        "packedContexts": [{"__reset__": True}],
        "evidenceGroups": [{"__reset__": True}],
        "groundedAnswer": {},
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
    query_plan = state.get("queryPlan", {})
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
        context_section = (
            f"Conversation Context:\n{conversation_summary}\n"
            if conversation_summary.strip()
            else ""
        ) + f"User Query:\n{seed_query}\n"
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

    return {
        "rewrittenQuestions": list(dict.fromkeys(questions))[:3],
        "originalQuery": last_message.content,
    }

def plan_query(state: GraphState):
    original_query = state.get("originalQuery") or state["messages"][-1].content
    conversation_summary = state.get("conversation_summary", "")
    corpus_profile = state.get("corpusProfile", "")

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

    return {"queryPlan": plan.model_dump(), "originalQuery": original_query}



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



def aggregate_answers(state: GraphState):
    evidence_groups = state.get("evidenceGroups", [])
    if not evidence_groups:
        return {"messages": [AIMessage(content="No answers were generated.")]}

    packed_contexts = sorted(
        state.get("packedContexts", []),
        key=lambda item: str(item.get("subquery", "")),
    )
    retrieval_evidence = sorted(
        state.get("retrievalEvidence", []),
        key=lambda item: str(item.get("subquery", "")),
    )
    sorted_groups = sorted(
        evidence_groups,
        key=lambda item: str(item.get("subquery", "")),
    )

    payload = {
        "question": state.get("originalQuery", ""),
        "query_plan": state.get("queryPlan", {}),
        "packed_context": packed_contexts,
        "evidence_groups": sorted_groups,
        "retrieval_evidence": retrieval_evidence,
    }

    llm = get_llm_by_type("aggregate_answers")
    try:
        structured_llm = llm.with_config(temperature=0).with_structured_output(
            GroundedAnswer
        )
        grounded_answer = structured_llm.invoke(
            [
                SystemMessage(content=get_aggregation_prompt()),
                HumanMessage(content=str(payload)),
            ]
        ).model_dump()
    except Exception:
        collected_evidence: list[dict] = []
        for group in sorted_groups:
            for item in group.get("evidence", []):
                collected_evidence.append(item)

        unique_evidence: list[dict] = []
        seen_keys: set[str] = set()
        for item in collected_evidence:
            key = f"{item.get('doc_id', '')}:{item.get('node_id', '')}:{item.get('quote', '')}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_evidence.append(item)

        evidence_count = len(unique_evidence)
        answer = (
            "I couldn't find any relevant information in the available sources to answer your question."
            if evidence_count == 0
            else " ".join(
                item.get("quote", "").strip()
                for item in unique_evidence[:3]
                if item.get("quote", "").strip()
            ).strip()
        )
        grounded_answer = {
            "answer": answer
            or "I couldn't find any relevant information in the available sources to answer your question.",
            "reasoning_summary": (
                f"Synthesized from {len(sorted_groups)} evidence group(s) and {evidence_count} unique evidence item(s)."
            ),
            "evidence": unique_evidence[:5],
            "confidence": min(0.95, 0.25 + (0.12 * evidence_count)),
            "limitations": (
                "Available evidence is limited to the retrieved passages."
                if evidence_count
                else "No structured evidence was captured from retrieval."
            ),
        }

    content = render_grounded_answer(grounded_answer)
    return {
        "groundedAnswer": grounded_answer,
        "messages": [AIMessage(content=content)],
    }
