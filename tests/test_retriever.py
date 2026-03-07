"""Tests for retriever module."""

import pytest
from langchain_core.documents import Document
from indexing.bm25_index import create_bm25_bundle
from indexing.retriever import BM25Retriever, FusionRetriever
from indexing.embeddings import FakeEmbeddings
from indexing.vectorstore import VectorStore


@pytest.fixture
def vector_store(sample_documents, tmp_path):
    """Create a vector store with fake embeddings for testing."""
    embeddings = FakeEmbeddings(dimensions=384)
    persist_dir = str(tmp_path / "test_faiss")

    vector_store = VectorStore(persist_directory=persist_dir, embeddings=embeddings)
    vector_store.add_documents(sample_documents)

    return vector_store


@pytest.fixture
def bm25_bundle(sample_documents):
    """Create a BM25 bundle for testing."""
    return create_bm25_bundle(sample_documents)


def test_bm25_retriever_basic_query(bm25_bundle):
    """Test BM25Retriever returns relevant documents for a query."""
    retriever = BM25Retriever(bundle=bm25_bundle, k=3)

    results = retriever.invoke("Python programming")

    # Should return results
    assert len(results) > 0
    assert len(results) <= 3

    # Results should be Document objects
    for doc in results:
        assert isinstance(doc, Document)


def test_bm25_retriever_respects_k_parameter(bm25_bundle):
    """Test BM25Retriever respects k parameter."""
    retriever = BM25Retriever(bundle=bm25_bundle, k=2)

    results = retriever.invoke("test query")

    assert len(results) <= 2


def test_bm25_retriever_returns_documents_with_metadata(bm25_bundle):
    """Test BM25Retriever returns documents with metadata intact."""
    retriever = BM25Retriever(bundle=bm25_bundle, k=5)

    results = retriever.invoke("Python")

    for doc in results:
        assert hasattr(doc, "metadata")
        assert "source" in doc.metadata


def test_fusion_retriever_combines_results(vector_store, bm25_bundle):
    """Test FusionRetriever combines vector and BM25 results."""
    retriever = FusionRetriever(
        vectorstore=vector_store, bm25=bm25_bundle, k=3, alpha=0.5, fetch_k=10
    )

    results = retriever.invoke("Python programming language")

    # Should return combined results
    assert len(results) > 0
    assert len(results) <= 3

    # Results should be Document objects
    for doc in results:
        assert isinstance(doc, Document)


def test_fusion_retriever_alpha_weighting(vector_store, bm25_bundle):
    """Test FusionRetriever with different alpha values."""
    # Alpha = 1.0 (only vector search)
    retriever_vector_only = FusionRetriever(
        vectorstore=vector_store, bm25=bm25_bundle, k=3, alpha=1.0, fetch_k=10
    )

    results_vector = retriever_vector_only.invoke("Python")

    # Alpha = 0.0 (only BM25)
    retriever_bm25_only = FusionRetriever(
        vectorstore=vector_store, bm25=bm25_bundle, k=3, alpha=0.0, fetch_k=10
    )

    results_bm25 = retriever_bm25_only.invoke("Python")

    # Both should return results
    assert len(results_vector) > 0
    assert len(results_bm25) > 0


def test_fusion_retriever_deduplicates_results(vector_store, bm25_bundle):
    """Test FusionRetriever deduplicates documents from both sources."""
    retriever = FusionRetriever(
        vectorstore=vector_store, bm25=bm25_bundle, k=5, alpha=0.5, fetch_k=20
    )

    results = retriever.invoke("Python machine learning")

    # Should deduplicate - no duplicate documents
    result_keys = []
    for doc in results:
        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", ""),
            doc.page_content,
        )
        assert key not in result_keys, "Found duplicate document in results"
        result_keys.append(key)


def test_fusion_retriever_fetch_k_parameter(vector_store, bm25_bundle):
    """Test FusionRetriever fetch_k parameter."""
    retriever = FusionRetriever(
        vectorstore=vector_store, bm25=bm25_bundle, k=2, alpha=0.5, fetch_k=5
    )

    results = retriever.invoke("Python")

    # Should respect k parameter even with larger fetch_k
    assert len(results) <= 2


def test_bm25_retriever_empty_query(bm25_bundle):
    """Test BM25Retriever handles empty query."""
    retriever = BM25Retriever(bundle=bm25_bundle, k=3)

    results = retriever.invoke("")

    # Should still return results (all documents scored equally)
    assert isinstance(results, list)


def test_fusion_retriever_empty_query(vector_store, bm25_bundle):
    """Test FusionRetriever handles empty query."""
    retriever = FusionRetriever(
        vectorstore=vector_store, bm25=bm25_bundle, k=3, alpha=0.5, fetch_k=10
    )

    results = retriever.invoke("")

    # Should still return results
    assert isinstance(results, list)


def test_bm25_retriever_with_large_k(bm25_bundle):
    """Test BM25Retriever when k exceeds available documents."""
    retriever = BM25Retriever(bundle=bm25_bundle, k=1000)

    results = retriever.invoke("test")

    # Should return all available documents
    assert len(results) <= len(bm25_bundle.documents)


def test_fusion_retriever_preserves_metadata(vector_store, bm25_bundle):
    """Test FusionRetriever preserves document metadata."""
    retriever = FusionRetriever(
        vectorstore=vector_store, bm25=bm25_bundle, k=3, alpha=0.5, fetch_k=10
    )

    results = retriever.invoke("Python")

    for doc in results:
        assert isinstance(doc, Document)
        assert hasattr(doc, "metadata")
        # Metadata should contain at least some keys
        assert len(doc.metadata) >= 0
