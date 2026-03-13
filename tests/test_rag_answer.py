"""Tests for rag_answer module."""

from __future__ import annotations

from langchain_core.documents import Document

from core.rag_answer import (
    format_retrieval_only_answer,
    render_grounded_citations,
    render_grounded_answer,
    render_out_of_scope_answer,
)


def test_format_retrieval_only_answer_with_docs(sample_documents):
    """Test format_retrieval_only_answer with documents returns formatted answer."""
    question = "What is Python used for?"
    result = format_retrieval_only_answer(question, sample_documents)

    # Verify essential components
    assert "Question:" in result
    assert question in result
    assert "Top excerpts" in result
    assert "Offline mode" in result

    # Verify at least some source content is included
    assert "Python" in result or "programming" in result.lower()

    # Verify sources section exists
    assert "Sources:" in result


def test_format_retrieval_only_answer_no_docs():
    """Test format_retrieval_only_answer with empty docs returns couldn't find message."""
    question = "What is the meaning of life?"
    result = format_retrieval_only_answer(question, [])

    assert "couldn't find" in result.lower()
    assert "available sources" in result.lower()


def test_format_retrieval_only_answer_preserves_question():
    """Test format_retrieval_only_answer includes the original question."""
    question = "How does machine learning work?"
    docs = [
        Document(
            page_content="Machine learning is a subset of AI.",
            metadata={"source": "ml.pdf", "page": 1},
        )
    ]

    result = format_retrieval_only_answer(question, docs)

    assert question in result


def test_format_retrieval_only_answer_includes_source_names(sample_documents):
    """Test format_retrieval_only_answer includes source file names."""
    question = "Test question"
    result = format_retrieval_only_answer(question, sample_documents)

    # Check that source names from sample_documents are included
    assert "doc1.pdf" in result or "doc2.pdf" in result or "doc3.pdf" in result


def test_format_retrieval_only_answer_max_snippets():
    """Test format_retrieval_only_answer respects max_snippets parameter."""
    docs = [
        Document(
            page_content=f"Content {i}",
            metadata={"source": f"doc{i}.pdf", "page": 1},
        )
        for i in range(10)
    ]

    result = format_retrieval_only_answer("test", docs, max_snippets=3)

    # Count number of snippet entries (lines starting with "- **")
    snippet_count = result.count("- **")
    assert snippet_count <= 3


def test_format_retrieval_only_answer_truncates_long_content():
    """Test format_retrieval_only_answer truncates long content."""
    long_content = "A" * 1000
    docs = [
        Document(
            page_content=long_content,
            metadata={"source": "long.pdf", "page": 1},
        )
    ]

    result = format_retrieval_only_answer("test", docs, max_chars_per_snippet=100)

    # Verify content is truncated with ellipsis
    assert "…" in result or len(result) < len(long_content) + 200


def test_format_retrieval_only_answer_groups_by_source():
    """Test format_retrieval_only_answer groups documents by source."""
    docs = [
        Document(page_content="Content 1", metadata={"source": "doc1.pdf"}),
        Document(page_content="Content 2", metadata={"source": "doc1.pdf"}),
        Document(page_content="Content 3", metadata={"source": "doc2.pdf"}),
    ]

    result = format_retrieval_only_answer("test", docs)

    # Both sources should appear
    assert "doc1.pdf" in result
    assert "doc2.pdf" in result


def test_format_retrieval_only_answer_handles_missing_source():
    """Test format_retrieval_only_answer handles documents without source metadata."""
    docs = [
        Document(page_content="Content without source", metadata={}),
        Document(page_content="Content with source", metadata={"source": "known.pdf"}),
    ]

    result = format_retrieval_only_answer("test", docs)

    # Should handle missing source gracefully
    assert "unknown" in result.lower() or "Content without source" in result
    assert "known.pdf" in result


def test_format_retrieval_only_answer_includes_page_numbers():
    """Test format_retrieval_only_answer includes page numbers when available."""
    docs = [
        Document(
            page_content="Content on page 5",
            metadata={"source": "doc.pdf", "page": 5},
        )
    ]

    result = format_retrieval_only_answer("test", docs)

    # Page number should be included
    assert "p.5" in result or "page 5" in result.lower()


def test_render_grounded_answer_includes_evidence_metadata():
    payload = {
        "answer": "The retrieval pipeline now produces grounded answers.",
        "reasoning_summary": "The answer is supported by structured evidence groups.",
        "confidence": 0.82,
        "limitations": "Only the retrieved passages were considered.",
        "evidence": [
            {
                "source": "design.md",
                "doc_id": "doc-1",
                "node_id": "node-9",
                "page": 3,
                "section_path": ["Milestone 3", "Aggregate"],
                "quote": "aggregate_answers should consume structured evidence.",
                "relevance": "Directly describes the new aggregation contract.",
            }
        ],
    }

    result = render_grounded_answer(payload)

    assert "Confidence" in result
    assert "design.md" in result
    assert "node-9" in result
    assert "Milestone 3 > Aggregate" in result
    assert "aggregate_answers should consume structured evidence." in result


def test_render_out_of_scope_answer_sections():
    payload = {
        "reason": "This question is outside the current corpus.",
        "boundary": "The corpus focuses on the agentic RAG implementation.",
        "suggestion": "Ask about retrieval, grounding, or UI behavior.",
        "next_action": "Upload documents about the external topic.",
    }

    result = render_out_of_scope_answer(payload)

    assert "Current coverage" in result
    assert "Better question" in result
    assert "Next step" in result


def test_render_grounded_citations_groups_by_source_and_section():
    payload = {
        "evidence": [
            {
                "source": "tasks.md",
                "doc_id": "doc-a",
                "node_id": "node-1",
                "page": 2,
                "section_path": ["4.3", "4.3.3"],
                "quote": "citation 渲染应直接基于 GroundedAnswer.evidence。",
                "relevance": "Defines citation rendering requirements.",
            },
            {
                "source": "tasks.md",
                "doc_id": "doc-a",
                "node_id": "node-2",
                "page": 2,
                "section_path": ["4.3", "4.3.3"],
                "quote": "每条 citation 至少应可追溯到 source、doc_id、node_id。",
                "relevance": "Defines required citation fields.",
            },
        ]
    }

    result = render_grounded_citations(payload)

    assert "## Citations" in result
    assert "### tasks.md" in result
    assert "Section:** 4.3 > 4.3.3" in result
    assert result.count("- Citation") == 2
