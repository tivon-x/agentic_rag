from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from functools import partial

from agent.orchestrator_agent import create_orchestrator_agent
from agent.edges import route_after_rewrite
from agent.graph_state import State

from agent.nodes import summarize_history, rewrite_query, request_clarification
from agent.nodes import aggregate_answers


def create_agent_graph(llm, tools_list):
    agent_subgraph = create_orchestrator_agent(llm, tools_list)
    checkpointer = InMemorySaver()

    graph_builder = StateGraph(State)
    graph_builder.add_node("summarize_history", partial(summarize_history, llm=llm))
    graph_builder.add_node("rewrite_query", partial(rewrite_query, llm=llm))
    graph_builder.add_node(request_clarification)
    graph_builder.add_node("agent", agent_subgraph)
    graph_builder.add_node("aggregate_answers", partial(aggregate_answers, llm=llm))

    graph_builder.add_edge(START, "summarize_history")
    graph_builder.add_edge("summarize_history", "rewrite_query")
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph_builder.add_edge("request_clarification", "rewrite_query")
    graph_builder.add_edge(["agent"], "aggregate_answers")
    graph_builder.add_edge("aggregate_answers", END)

    agent_graph = graph_builder.compile(
        checkpointer=checkpointer, interrupt_before=["request_clarification"]
    )

    print("✓ Agent graph compiled successfully.")
    return agent_graph
