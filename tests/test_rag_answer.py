"""Tests for rag_answer module."""

from __future__ import annotations

from langchain_core.documents import Document
from core.rag_answer import format_retrieval_only_answer


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
