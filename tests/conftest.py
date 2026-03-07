"""Shared fixtures for pytest test suite."""

import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_documents():
    """Sample documents for testing retrievers and indexing."""
    return [
        Document(
            page_content="Python is a programming language used for AI.",
            metadata={"source": "doc1.pdf", "page": 1},
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "doc1.pdf", "page": 2},
        ),
        Document(
            page_content="FAISS is a library for efficient similarity search.",
            metadata={"source": "doc2.pdf", "page": 1},
        ),
        Document(
            page_content="BM25 is a ranking function used in information retrieval.",
            metadata={"source": "doc2.pdf", "page": 2},
        ),
        Document(
            page_content="LangChain provides tools for building LLM applications.",
            metadata={"source": "doc3.pdf", "page": 1},
        ),
    ]


@pytest.fixture
def sample_text():
    """Sample text for testing chunkers."""
    return (
        "Python is a versatile programming language. "
        "It supports multiple programming paradigms. "
        "Python is widely used in data science and machine learning. "
        "The language has a rich ecosystem of libraries."
    )


@pytest.fixture
def tmp_env_file(tmp_path):
    """Create a temporary .env file for testing settings."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=test-key-123\n"
        "OPENAI_API_BASE=https://api.test.com/v1\n"
        'LLM_MODEL="test-model"\n'
        "LOG_LEVEL=DEBUG\n"
        "CHUNK_SIZE=256\n"
        "CHUNK_OVERLAP=32\n"
        "RETRIEVER_K=5\n"
    )
    return env_file
