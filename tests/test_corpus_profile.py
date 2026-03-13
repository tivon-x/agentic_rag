from core.corpus_profile import (
    analyze_corpus_profile_match,
    apply_profile_query_plan_prior,
    expand_queries_with_corpus_profile,
    load_corpus_profile,
    save_corpus_profile,
)


def test_load_corpus_profile_normalizes_legacy_shape(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "corpus_profile.json").write_text(
        '{"name":"RAG KB","summary":"Retriever internals","coverage":"retrieval stack"}',
        encoding="utf-8",
    )

    profile = load_corpus_profile(index_dir)

    assert profile["name"] == "RAG KB"
    assert profile["non_coverage"] == ""
    assert profile["domain_keywords"] == []
    assert profile["primary_entities"] == []


def test_save_corpus_profile_persists_extended_fields(tmp_path):
    index_dir = tmp_path / "index"
    save_corpus_profile(
        index_dir,
        name="Agentic RAG",
        summary="Retriever and answer pipeline",
        coverage="planning, rerank, pack",
        non_coverage="finance",
        usage_notes="Prefer implementation questions",
        source_examples=["tasks.md"],
        recommended_questions=["How does rerank work?"],
        forbidden_questions=["What is the stock price?"],
        domain_keywords=["rerank", "packed context"],
        preferred_answer_style="Lead with a concise conclusion and cite evidence.",
        primary_entities=["FusionRetriever"],
    )

    profile = load_corpus_profile(index_dir)

    assert profile["non_coverage"] == "finance"
    assert profile["domain_keywords"] == ["rerank", "packed context"]
    assert profile["preferred_answer_style"].startswith("Lead with")


def test_analyze_corpus_profile_match_flags_forbidden_topic():
    analysis = analyze_corpus_profile_match(
        "What is the stock price forecast?",
        {
            "coverage": "retrieval architecture",
            "non_coverage": "financial analysis and stock price prediction",
            "forbidden_questions": ["stock price forecast"],
            "domain_keywords": ["retrieval", "rerank"],
            "primary_entities": ["FusionRetriever"],
        },
    )

    assert analysis["force_out_of_scope"] is True
    assert analysis["matched_forbidden_questions"] == ["stock price forecast"]


def test_analyze_corpus_profile_match_does_not_overtrigger_non_coverage():
    analysis = analyze_corpus_profile_match(
        "Please give a retrieval pipeline analysis",
        {
            "coverage": "retrieval pipeline and rerank design",
            "non_coverage": "financial analysis and stock price prediction",
            "domain_keywords": ["retrieval", "rerank"],
        },
    )

    assert analysis["matched_non_coverage"] == []
    assert analysis["force_out_of_scope"] is False


def test_apply_profile_query_plan_prior_enriches_subqueries():
    plan = apply_profile_query_plan_prior(
        {
            "intent": "fact",
            "subqueries": ["how does fusion work"],
            "preferred_node_types": ["section"],
        },
        original_query="how does fusion work",
        profile={
            "domain_keywords": ["fusion", "rerank"],
            "primary_entities": ["FusionRetriever"],
        },
    )

    assert "FusionRetriever" in plan["subqueries"][0]
    assert "paragraph" in plan["preferred_node_types"]


def test_expand_queries_with_corpus_profile_adds_matching_priors():
    queries = expand_queries_with_corpus_profile(
        ["how does rerank work"],
        original_query="how does rerank work in fusionretriever",
        query_plan={
            "profile_hints": {
                "matched_domain_keywords": ["rerank"],
                "matched_primary_entities": ["FusionRetriever"],
            }
        },
        profile={
            "domain_keywords": ["rerank"],
            "primary_entities": ["FusionRetriever"],
        },
    )

    assert any("FusionRetriever" in query for query in queries)
