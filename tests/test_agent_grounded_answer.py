from langchain_core.messages import HumanMessage

from agent.nodes import aggregate_answers, out_of_scope_answer


class _FakeStructuredInvoker:
    def __init__(self, response):
        self._response = response

    def invoke(self, _messages):
        return self._response


class _FakeStructuredLLM:
    def __init__(self, response):
        self._response = response

    def with_config(self, **_kwargs):
        return self

    def with_structured_output(self, _schema):
        return _FakeStructuredInvoker(self._response)


def test_aggregate_answers_renders_grounded_payload(monkeypatch):
    from agent.schemas import GroundedAnswer

    response = GroundedAnswer(
        answer="Milestone 3 upgrades answer generation to use structured evidence.",
        reasoning_summary="The evidence groups show the new input and output contracts.",
        evidence=[
            {
                "doc_id": "tasks",
                "node_id": "m3-1",
                "source": "tasks.md",
                "section_path": ["4.3", "4.3.2"],
                "page": None,
                "quote": "aggregate_answers should directly consume structured evidence.",
                "score": 0.91,
                "relevance": "Defines the grounded aggregation requirement.",
            }
        ],
        confidence=0.88,
        limitations="This answer reflects the currently retrieved task spec only.",
    )
    monkeypatch.setattr(
        "agent.nodes.get_llm_by_type", lambda _task: _FakeStructuredLLM(response)
    )
    state = {
        "originalQuery": "Milestone 3 要做什么？",
        "queryPlan": {"intent": "summary", "subqueries": ["Milestone 3"]},
        "packedContexts": [{"subquery": "Milestone 3", "passage_count": 1}],
        "retrievalEvidence": [{"subquery": "Milestone 3", "evidence": []}],
        "evidenceGroups": [
            {
                "subquery": "Milestone 3",
                "intent": "summary",
                "packed_context": {"passage_count": 1},
                "evidence": response.model_dump()["evidence"],
                "debug": {},
            }
        ],
    }

    result = aggregate_answers(state)
    content = result["messages"][0].content

    assert "Milestone 3 upgrades answer generation" in content
    assert "Confidence" in content
    assert "tasks.md" in content
    assert result["groundedAnswer"]["confidence"] == 0.88


def test_out_of_scope_answer_renders_structured_response(monkeypatch):
    from agent.schemas import OutOfScopeResponse

    response = OutOfScopeResponse(
        reason="This question asks about a topic not described in the uploaded corpus.",
        boundary="The corpus only covers the agentic RAG codebase and task spec.",
        suggestion="Ask about retrieval, evidence capture, or answer rendering.",
        next_action="Upload documents about the missing topic.",
    )
    monkeypatch.setattr(
        "agent.nodes.get_llm_by_type", lambda _task: _FakeStructuredLLM(response)
    )
    state = {
        "messages": [HumanMessage(content="请解释量子计算的最新突破")],
        "corpusProfile": "This KB covers an agentic RAG implementation.",
        "routingReason": "The question is outside the codebase domain.",
    }

    result = out_of_scope_answer(state)
    content = result["messages"][0].content

    assert "Current coverage" in content
    assert "Better question" in content
    assert "Upload documents about the missing topic." in content
