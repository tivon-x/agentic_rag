from langchain_core.messages import HumanMessage

from agent.nodes import aggregate_answers, decide_retrieval, plan_query, rewrite_query
from agent.schemas import GroundedAnswer, QueryAnalysis, QueryPlan


class _StructuredLLM:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def with_config(self, **kwargs):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.messages = messages
        return self.response


def test_decide_retrieval_uses_profile_non_coverage_prior():
    result = decide_retrieval(
        {
            "messages": [HumanMessage(content="What is the stock price forecast?")],
            "corpusProfile": "Knowledge Base Name: Agentic RAG",
            "corpusProfileData": {
                "non_coverage": "stock price prediction and finance",
                "forbidden_questions": ["stock price forecast"],
            },
        }
    )

    assert result["routingDecision"] == "out_of_scope"
    assert "stock price forecast" in result["routingReason"]


def test_decide_retrieval_does_not_short_circuit_on_single_generic_overlap(
    monkeypatch,
):
    fake_llm = _StructuredLLM(
        type(
            "Decision",
            (),
            {"decision": "retrieve", "reason": "Coverage keywords match the query."},
        )()
    )
    monkeypatch.setattr("agent.nodes.get_llm_by_type", lambda _: fake_llm)

    result = decide_retrieval(
        {
            "messages": [HumanMessage(content="Please provide a retrieval pipeline analysis")],
            "corpusProfile": "Knowledge Base Name: Agentic RAG",
            "corpusProfileData": {
                "coverage": "retrieval pipeline and rerank design",
                "non_coverage": "financial analysis and stock price prediction",
                "domain_keywords": ["retrieval", "rerank"],
            },
        }
    )

    assert result["routingDecision"] == "retrieve"


def test_plan_query_applies_profile_priors(monkeypatch):
    fake_llm = _StructuredLLM(
        QueryPlan(
            intent="fact",
            subqueries=["how does rerank work"],
            preferred_node_types=["section"],
        )
    )
    monkeypatch.setattr("agent.nodes.get_llm_by_type", lambda _: fake_llm)

    result = plan_query(
        {
            "messages": [HumanMessage(content="how does rerank work in fusionretriever")],
            "corpusProfile": "Knowledge Base Name: Agentic RAG",
            "corpusProfileData": {
                "domain_keywords": ["rerank"],
                "primary_entities": ["FusionRetriever"],
            },
        }
    )

    assert "FusionRetriever" in result["queryPlan"]["subqueries"][0]
    assert "paragraph" in result["queryPlan"]["preferred_node_types"]


def test_rewrite_query_expands_with_profile_priors(monkeypatch):
    fake_llm = _StructuredLLM(
        QueryAnalysis(
            is_clear=True,
            questions=["how does rerank work"],
            clarification_needed="",
        )
    )
    monkeypatch.setattr("agent.nodes.get_llm_by_type", lambda _: fake_llm)

    result = rewrite_query(
        {
            "messages": [HumanMessage(content="how does rerank work in fusionretriever")],
            "queryPlan": {
                "subqueries": ["how does rerank work"],
                "profile_hints": {
                    "matched_domain_keywords": ["rerank"],
                    "matched_primary_entities": ["FusionRetriever"],
                },
            },
            "corpusProfile": "Knowledge Base Name: Agentic RAG",
            "corpusProfileData": {
                "domain_keywords": ["rerank"],
                "primary_entities": ["FusionRetriever"],
            },
        }
    )

    assert any("FusionRetriever" in query for query in result["rewrittenQuestions"])


def test_aggregate_answers_passes_preferred_answer_style(monkeypatch):
    fake_llm = _StructuredLLM(
        GroundedAnswer(
            answer="Conclusion first.",
            reasoning_summary="Supported by one evidence item.",
            evidence=[
                {
                    "doc_id": "doc-1",
                    "node_id": "node-1",
                    "source": "tasks.md",
                    "section_path": ["4.4"],
                    "page": None,
                    "quote": "preferred_answer_style should be used in answer generation.",
                    "score": 0.9,
                    "relevance": "Confirms answer style prior.",
                }
            ],
            confidence=0.9,
            limitations="Test only.",
        )
    )
    monkeypatch.setattr("agent.nodes.get_llm_by_type", lambda _: fake_llm)

    result = aggregate_answers(
        {
            "originalQuery": "How should the answer be formatted?",
            "queryPlan": {"intent": "fact", "subqueries": ["answer format"]},
            "packedContexts": [{"subquery": "answer format"}],
            "retrievalEvidence": [],
            "evidenceGroups": [
                {
                    "subquery": "answer format",
                    "intent": "fact",
                    "packed_context": {},
                    "evidence": [
                        {
                            "doc_id": "doc-1",
                            "node_id": "node-1",
                            "source": "tasks.md",
                            "section_path": ["4.4"],
                            "page": None,
                            "quote": "preferred_answer_style should be used in answer generation.",
                            "score": 0.9,
                            "relevance": "Confirms answer style prior.",
                        }
                    ],
                    "debug": {},
                }
            ],
            "corpusProfileData": {
                "preferred_answer_style": "Lead with the conclusion, then cite the evidence."
            },
        }
    )

    assert "Lead with the conclusion" in fake_llm.messages[0].content
    assert result["groundedAnswer"]["answer"] == "Conclusion first."
