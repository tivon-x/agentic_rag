"""Tests for BM25 index module."""

from langchain_core.documents import Document
from indexing.bm25_index import create_bm25_bundle, BM25Bundle


def test_create_bm25_bundle(sample_documents):
    """Test create_bm25_bundle creates bundle from documents."""
    bundle = create_bm25_bundle(sample_documents)

    assert isinstance(bundle, BM25Bundle)
    assert len(bundle.documents) == len(sample_documents)
    assert len(bundle.tokenized_corpus) == len(sample_documents)
    assert bundle._bm25 is not None

    # Verify documents are stored correctly
    for i, doc in enumerate(sample_documents):
        assert bundle.documents[i].page_content == doc.page_content
        assert bundle.documents[i].metadata == doc.metadata


def test_bm25_query(sample_documents):
    """Test BM25 query returns relevant documents."""
    bundle = create_bm25_bundle(sample_documents)

    # Query for Python-related content
    results = bundle.query("Python programming", k=3)

    assert len(results) <= 3
    assert len(results) > 0

    # Verify results contain relevant content
    results_text = " ".join([doc.page_content for doc in results])
    assert "Python" in results_text or "programming" in results_text.lower()


def test_bm25_topk_with_scores(sample_documents):
    """Test BM25 topk_with_scores returns documents with scores."""
    bundle = create_bm25_bundle(sample_documents)

    results = bundle.topk_with_scores("artificial intelligence", k=3)

    assert len(results) <= 3
    assert len(results) > 0

    # Verify results are tuples of (Document, score)
    for doc, score in results:
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        assert score >= 0.0

    # Verify results are sorted by score (descending)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_empty_corpus():
    """Test BM25 handles empty corpus gracefully."""
    empty_docs = []
    bundle = create_bm25_bundle(empty_docs)

    assert len(bundle.documents) == 0
    assert len(bundle.tokenized_corpus) == 0
    # Empty corpus has None as bm25 index
    assert bundle._bm25 is None


def test_bm25_bundle_rebuild_index(sample_documents):
    """Test BM25Bundle rebuild_index functionality."""
    bundle = create_bm25_bundle(sample_documents)

    # Clear the index
    bundle._bm25 = None

    # Rebuild should restore the index
    bundle.rebuild_index()
    assert bundle._bm25 is not None

    # Should still be able to query
    results = bundle.query("Python", k=2)
    assert len(results) > 0


def test_bm25_with_empty_strings():
    """Test BM25 filters out empty documents."""
    docs = [
        Document(page_content="Python is great", metadata={"source": "doc1.txt"}),
        Document(page_content="", metadata={"source": "doc2.txt"}),
        Document(
            page_content="   ", metadata={"source": "doc3.txt"}
        ),  # whitespace only
        Document(
            page_content="Machine learning rocks", metadata={"source": "doc4.txt"}
        ),
    ]

    bundle = create_bm25_bundle(docs)

    # Empty/whitespace documents should be filtered
    assert len(bundle.documents) == 2
    assert bundle.documents[0].page_content == "Python is great"
    assert bundle.documents[1].page_content == "Machine learning rocks"


def test_bm25_query_returns_correct_k(sample_documents):
    """Test BM25 query respects k parameter."""
    bundle = create_bm25_bundle(sample_documents)

    # Request more results than available
    results = bundle.query("test", k=100)
    assert len(results) <= len(sample_documents)

    # Request specific k
    results = bundle.query("test", k=2)
    assert len(results) == 2
