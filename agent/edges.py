from typing import Literal

from langgraph.types import Send

from .states import GraphState



def route_after_decision(
    state: GraphState,
) -> Literal["direct_answer", "plan_query", "out_of_scope_answer"]:
    decision = state.get("routingDecision", "retrieve")
    if decision == "direct_answer":
        return "direct_answer"
    if decision == "out_of_scope":
        return "out_of_scope_answer"
    return "plan_query"



def route_after_rewrite(state: GraphState) -> list[Send]:
    return [
        Send(
            "agent",
            {
                "question": query,
                "question_index": idx,
                "query_plan": state.get("queryPlan", {}),
                "messages": [],
            },
        )
        for idx, query in enumerate(state.get("rewrittenQuestions", []))
    ]
