"""Tests for agent schemas module."""

from agent.schemas import QueryAnalysis


def test_query_analysis_valid():
    """Test QueryAnalysis model with valid data."""
    analysis = QueryAnalysis(
        is_clear=True,
        questions=["What is Python?", "How does machine learning work?"],
        clarification_needed="",
    )

    assert analysis.is_clear is True
    assert len(analysis.questions) == 2
    assert "What is Python?" in analysis.questions
    assert "How does machine learning work?" in analysis.questions
    assert analysis.clarification_needed == ""


def test_query_analysis_unclear():
    """Test QueryAnalysis model with unclear query."""
    analysis = QueryAnalysis(
        is_clear=False,
        questions=[],
        clarification_needed="The question is too vague. Please provide more details.",
    )

    assert analysis.is_clear is False
    assert len(analysis.questions) == 0
    assert "vague" in analysis.clarification_needed


def test_query_analysis_field_types():
    """Test QueryAnalysis field types and validation."""
    # Valid construction
    analysis = QueryAnalysis(
        is_clear=True, questions=["Single question"], clarification_needed=""
    )

    # Verify field types
    assert isinstance(analysis.is_clear, bool)
    assert isinstance(analysis.questions, list)
    assert isinstance(analysis.clarification_needed, str)


def test_query_analysis_empty_questions():
    """Test QueryAnalysis with empty questions list."""
    analysis = QueryAnalysis(
        is_clear=False, questions=[], clarification_needed="Need more information"
    )

    assert len(analysis.questions) == 0
    assert analysis.is_clear is False


def test_query_analysis_serialization():
    """Test QueryAnalysis can be serialized to dict."""
    analysis = QueryAnalysis(
        is_clear=True, questions=["Q1", "Q2"], clarification_needed=""
    )

    data = analysis.model_dump()

    assert data["is_clear"] is True
    assert data["questions"] == ["Q1", "Q2"]
    assert data["clarification_needed"] == ""


def test_query_analysis_multiple_questions():
    """Test QueryAnalysis with multiple rewritten questions."""
    questions = [
        "What are the main features of Python?",
        "How is Python used in data science?",
        "What libraries does Python have for machine learning?",
    ]

    analysis = QueryAnalysis(
        is_clear=True, questions=questions, clarification_needed=""
    )

    assert len(analysis.questions) == 3
    assert all(q in analysis.questions for q in questions)
