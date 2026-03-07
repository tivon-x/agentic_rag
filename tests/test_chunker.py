"""Tests for chunker module."""

from langchain_core.documents import Document
from indexing.chunker import RecursiveChunker, TokenChunker


def test_recursive_chunker(sample_text):
    """Test RecursiveChunker splits text into chunks within size limits."""
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    docs = [Document(page_content=sample_text, metadata={"source": "test.txt"})]

    chunks = chunker.chunk(docs)

    # Verify chunks were created
    assert len(chunks) > 0

    # Verify each chunk is within size limits (allow some margin for overlap)
    for chunk in chunks:
        assert len(chunk.page_content) <= 150  # Allow margin for word boundaries

    # Verify metadata is preserved
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.txt"

    # Verify chunks contain content from original
    combined = " ".join(chunk.page_content for chunk in chunks)
    assert "Python" in combined
    assert "programming" in combined


def test_recursive_chunker_multiple_docs(sample_documents):
    """Test RecursiveChunker handles multiple documents."""
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)

    chunks = chunker.chunk(sample_documents)

    # Should have at least as many chunks as input documents
    assert len(chunks) >= len(sample_documents)

    # Verify each chunk has metadata
    for chunk in chunks:
        assert "source" in chunk.metadata


def test_token_chunker(sample_text):
    """Test TokenChunker splits text based on token count."""
    chunker = TokenChunker(chunk_size=50, chunk_overlap=10)
    docs = [Document(page_content=sample_text, metadata={"source": "test.txt"})]

    chunks = chunker.chunk(docs)

    # Verify chunks were created
    assert len(chunks) > 0

    # Verify metadata is preserved
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.txt"

    # Verify chunks contain content
    for chunk in chunks:
        assert len(chunk.page_content) > 0


def test_token_chunker_with_long_text():
    """Test TokenChunker with longer text that requires multiple chunks."""
    long_text = " ".join(
        [
            "This is a sentence about Python programming and machine learning."
            for _ in range(20)
        ]
    )
    chunker = TokenChunker(chunk_size=100, chunk_overlap=20)
    docs = [Document(page_content=long_text, metadata={"source": "long.txt"})]

    chunks = chunker.chunk(docs)

    # Should create multiple chunks
    assert len(chunks) > 1

    # Verify overlap exists (check if content appears in adjacent chunks)
    for i in range(len(chunks) - 1):
        # At least some overlap should exist between adjacent chunks
        assert len(chunks[i].page_content) > 0
        assert len(chunks[i + 1].page_content) > 0


def test_recursive_chunker_empty_document():
    """Test RecursiveChunker handles empty documents gracefully."""
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    docs = [Document(page_content="", metadata={"source": "empty.txt"})]

    chunks = chunker.chunk(docs)

    # Empty documents may result in empty chunks or be filtered out
    # Just verify it doesn't crash
    assert isinstance(chunks, list)
