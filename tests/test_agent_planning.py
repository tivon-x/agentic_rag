from langchain_core.messages import HumanMessage

from agent.nodes import plan_query


class _FakeStructuredPlanner:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        from agent.schemas import QueryPlan

        return QueryPlan(
            intent="summary",
            subqueries=["检索链路总结", "dedupe rerank"],
            preferred_node_types=["section", "paragraph"],
        )


class _FakePlanner:
    def with_config(self, **_kwargs):
        return _FakeStructuredPlanner()


def test_plan_query_uses_llm_when_available(monkeypatch):
    monkeypatch.setattr("agent.nodes.get_llm_by_type", lambda _task: _FakePlanner())
    state = {"messages": [HumanMessage(content="请总结检索链路的工作流程")]}

    result = plan_query(state)

    assert result["queryPlan"]["intent"] == "summary"
    assert result["queryPlan"]["subqueries"] == ["检索链路总结", "dedupe rerank"]


def test_plan_query_falls_back_to_default_plan_on_llm_errors(monkeypatch):
    monkeypatch.setattr(
        "agent.nodes.get_llm_by_type",
        lambda _task: (_ for _ in ()).throw(RuntimeError("llm unavailable")),
    )
    state = {"messages": [HumanMessage(content="请总结检索链路的工作流程")]}

    result = plan_query(state)

    assert result["queryPlan"]["intent"] == "fact"
    assert result["queryPlan"]["subqueries"] == ["请总结检索链路的工作流程"]
    assert result["queryPlan"]["preferred_node_types"] == ["paragraph"]
    assert result["queryPlan"]["profile_hints"] == {
        "matched_domain_keywords": [],
        "matched_primary_entities": [],
    }
