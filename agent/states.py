import operator
from typing import Annotated

from langchain.agents import AgentState
from langgraph.graph import MessagesState



def accumulate_or_reset(existing: list[dict], new: list[dict]) -> list[dict]:
    if new and any(item.get('__reset__') for item in new):
        return []
    return existing + new



def set_union(a: set[str], b: set[str]) -> set[str]:
    return a | b


class GraphState(MessagesState):
    """State for main agent graph"""

    routingDecision: str
    routingReason: str
    corpusProfile: str
    conversation_summary: str
    originalQuery: str
    queryPlan: dict
    rewrittenQuestions: list[str]
    agent_answers: Annotated[list[dict], accumulate_or_reset]
    retrievalEvidence: Annotated[list[dict], accumulate_or_reset]
    packedContexts: Annotated[list[dict], accumulate_or_reset]
    evidenceGroups: Annotated[list[dict], accumulate_or_reset]
    groundedAnswer: dict


class ResearchSearchState(AgentState):
    """State for individual research-search agent subgraph"""

    question: str
    question_index: int
    query_plan: dict
    context_summary: str
    retrieval_keys: Annotated[set[str], set_union]
    final_answer: str
    agent_answers: list[dict]
    retrievalEvidence: Annotated[list[dict], accumulate_or_reset]
    packedContexts: Annotated[list[dict], accumulate_or_reset]
    evidenceGroups: Annotated[list[dict], accumulate_or_reset]
    tool_call_count: Annotated[int, operator.add]
    iteration_count: Annotated[int, operator.add]
