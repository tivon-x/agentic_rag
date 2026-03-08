from typing import Annotated
import operator

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
    rewrittenQuestions: list[str]
    agent_answers: Annotated[list[dict], accumulate_or_reset]


class ResearchSearchState(AgentState):
    """State for individual research-search agent subgraph"""

    question: str
    question_index: int
    context_summary: str
    retrieval_keys: Annotated[set[str], set_union]
    final_answer: str
    agent_answers: list[dict]
    tool_call_count: Annotated[int, operator.add]
    iteration_count: Annotated[int, operator.add]
