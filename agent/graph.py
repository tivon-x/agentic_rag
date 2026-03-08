import logging

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.edges import route_after_rewrite
from agent.nodes import aggregate_answers, request_clarification, rewrite_query, summarize_history
from agent.orchestrator_agent import create_orchestrator_agent
from agent.states import GraphState


def create_agent_graph(tools_list):
    logger = logging.getLogger(__name__)

    agent_subgraph = create_orchestrator_agent(tools_list)
    checkpointer = InMemorySaver()

    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("summarize_history", summarize_history)
    graph_builder.add_node("rewrite_query", rewrite_query)
    graph_builder.add_node(request_clarification)
    graph_builder.add_node("agent", agent_subgraph)
    graph_builder.add_node("aggregate_answers", aggregate_answers)

    graph_builder.add_edge(START, "summarize_history")
    graph_builder.add_edge("summarize_history", "rewrite_query")
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph_builder.add_edge("request_clarification", "rewrite_query")
    graph_builder.add_edge(["agent"], "aggregate_answers")
    graph_builder.add_edge("aggregate_answers", END)

    agent_graph = graph_builder.compile(
        checkpointer=checkpointer, interrupt_before=["request_clarification"]
    )

    logger.info("Agent graph compiled successfully")
    return agent_graph
