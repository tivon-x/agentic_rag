import logging

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.edges import route_after_decision, route_after_rewrite
from agent.nodes import (
    aggregate_answers,
    decide_retrieval,
    direct_answer,
    inject_corpus_profile,
    out_of_scope_answer,
    plan_query,
    rewrite_query,
    summarize_history,
)
from agent.research_search_agent import create_research_search_agent
from agent.states import GraphState



def create_agent_graph(tools_list, corpus_profile: str = "", tool_factory=None):
    logger = logging.getLogger(__name__)

    agent_subgraph = create_research_search_agent(tools_list, tool_factory=tool_factory)
    checkpointer = InMemorySaver()

    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("inject_corpus_profile", inject_corpus_profile(corpus_profile))
    graph_builder.add_node("summarize_history", summarize_history)
    graph_builder.add_node("decide_retrieval", decide_retrieval)
    graph_builder.add_node("direct_answer", direct_answer)
    graph_builder.add_node("out_of_scope_answer", out_of_scope_answer)
    graph_builder.add_node("plan_query", plan_query)
    graph_builder.add_node("rewrite_query", rewrite_query)
    graph_builder.add_node("agent", agent_subgraph)
    graph_builder.add_node("aggregate_answers", aggregate_answers)

    graph_builder.add_edge(START, "inject_corpus_profile")
    graph_builder.add_edge("inject_corpus_profile", "summarize_history")
    graph_builder.add_edge("summarize_history", "decide_retrieval")
    graph_builder.add_conditional_edges("decide_retrieval", route_after_decision)
    graph_builder.add_edge("plan_query", "rewrite_query")
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph_builder.add_edge(["agent"], "aggregate_answers")
    graph_builder.add_edge("direct_answer", END)
    graph_builder.add_edge("out_of_scope_answer", END)
    graph_builder.add_edge("aggregate_answers", END)

    agent_graph = graph_builder.compile(checkpointer=checkpointer)

    logger.info("Agent graph compiled successfully")
    return agent_graph
