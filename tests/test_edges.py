from langchain_core.messages import HumanMessage

from agent.edges import route_after_rewrite


def test_route_after_rewrite_dispatches_each_query_as_human_message() -> None:
    query_plan = {"intent": "fact", "subqueries": ["first"]}

    sends = route_after_rewrite(
        {
            "rewrittenQuestions": ["first rewritten query", "second rewritten query"],
            "queryPlan": query_plan,
        }
    )

    assert [send.node for send in sends] == ["agent", "agent"]
    assert [send.arg["question"] for send in sends] == [
        "first rewritten query",
        "second rewritten query",
    ]
    assert [send.arg["question_index"] for send in sends] == [0, 1]
    assert all(send.arg["query_plan"] == query_plan for send in sends)
    assert all(
        len(send.arg["messages"]) == 1
        and isinstance(send.arg["messages"][0], HumanMessage)
        and send.arg["messages"][0].content == send.arg["question"]
        for send in sends
    )
