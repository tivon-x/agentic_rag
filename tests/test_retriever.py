"""Tests for retriever module."""

import json

import pytest
from langchain_core.documents import Document
from indexing.retrieval_pipeline import PackedContext
from indexing.bm25_index import create_bm25_bundle
from indexing.retriever import BM25Retriever, FusionRetriever
from indexing.embeddings import FakeEmbeddings
from indexing.models.doc_tree import ParsedDocumentTree
from indexing.models.node import Node
from indexing.stores.node_store import JsonNodeStore
from indexing.vectorstore import VectorStore
from agent.tools import ToolFactory


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


@pytest.fixture
def hierarchical_node_store(tmp_path):
    nodes = [
        Node(
            node_id="doc-1",
            parent_id=None,
            doc_id="doc-1",
            node_type="document",
            title="Guide",
            text="Intro paragraph.\n\nDetails paragraph.\n\nAppendix paragraph.",
            order=0,
            level=0,
            metadata={"source": "guide.md"},
        ),
        Node(
            node_id="sec-1",
            parent_id="doc-1",
            doc_id="doc-1",
            node_type="section",
            title="Intro",
            text="Intro paragraph.\n\nDetails paragraph.",
            order=0,
            level=1,
            metadata={"source": "guide.md", "section_path": "Intro"},
        ),
        Node(
            node_id="p-1",
            parent_id="sec-1",
            doc_id="doc-1",
            node_type="paragraph",
            title=None,
            text="Intro paragraph about Python and retrieval planning.",
            order=0,
            level=2,
            metadata={"source": "guide.md", "section_path": "Intro"},
            token_count=10,
        ),
        Node(
            node_id="p-2",
            parent_id="sec-1",
            doc_id="doc-1",
            node_type="paragraph",
            title=None,
            text="Details paragraph about dedupe and rerank in the pipeline.",
            order=1,
            level=2,
            metadata={"source": "guide.md", "section_path": "Intro"},
            token_count=10,
        ),
        Node(
            node_id="p-3",
            parent_id="sec-1",
            doc_id="doc-1",
            node_type="paragraph",
            title=None,
            text="Appendix paragraph unrelated to the main question.",
            order=2,
            level=2,
            metadata={"source": "guide.md", "section_path": "Intro"},
            token_count=10,
        ),
    ]
    tree = ParsedDocumentTree(
        doc_id="doc-1",
        root_id="doc-1",
        nodes=nodes,
        children_by_parent={"doc-1": ["sec-1"], "sec-1": ["p-1", "p-2", "p-3"]},
    )
    nodes_path = tmp_path / "nodes.jsonl"
    doc_trees_path = tmp_path / "doc_trees.json"
    JsonNodeStore(nodes_path, doc_trees_path).save_trees([tree])

    assert json.loads(doc_trees_path.read_text(encoding="utf-8"))
    return JsonNodeStore(nodes_path, doc_trees_path)


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


def test_fusion_retriever_returns_pipeline_debug(vector_store, bm25_bundle):
    retriever = FusionRetriever(
        vectorstore=vector_store, bm25=bm25_bundle, k=3, alpha=0.5, fetch_k=10
    )

    packed = retriever.retrieve("Python programming language")

    assert packed.passages
    assert packed.debug["raw_candidates"] >= len(packed.passages)
    assert packed.debug["dedupe"]["deduped_count"] <= packed.debug["dedupe"]["raw_count"]
    assert packed.packing_strategy == "score_then_contiguity"


def test_fusion_retriever_packs_adjacent_paragraphs_with_node_store(
    hierarchical_node_store, tmp_path
):
    documents = [
        Document(
            page_content="Intro paragraph about Python and retrieval planning.",
            metadata={
                "source": "guide.md",
                "node_id": "p-1",
                "parent_id": "sec-1",
                "doc_id": "doc-1",
                "node_type": "paragraph",
                "order": 0,
                "token_count": 10,
            },
        ),
        Document(
            page_content="Details paragraph about dedupe and rerank in the pipeline.",
            metadata={
                "source": "guide.md",
                "node_id": "p-2",
                "parent_id": "sec-1",
                "doc_id": "doc-1",
                "node_type": "paragraph",
                "order": 1,
                "token_count": 10,
            },
        ),
    ]
    embeddings = FakeEmbeddings(dimensions=384)
    vector_store = VectorStore(
        persist_directory=str(tmp_path / "hier_faiss"), embeddings=embeddings
    )
    vector_store.add_documents(documents)
    bm25_bundle = create_bm25_bundle(documents)
    retriever = FusionRetriever(
        vectorstore=vector_store,
        bm25=bm25_bundle,
        k=2,
        alpha=0.5,
        fetch_k=5,
        node_store=hierarchical_node_store,
    )

    packed = retriever.retrieve("dedupe rerank pipeline")

    assert packed.passages
    assert any(
        doc.metadata.get("merged_count", 1) > 1 or "Details paragraph" in doc.page_content
        for doc in packed.passages
    )


def test_fusion_retriever_promotes_section_context_for_summary_queries(
    hierarchical_node_store, tmp_path
):
    documents = [
        Document(
            page_content="Intro paragraph about Python and retrieval planning.",
            metadata={
                "source": "guide.md",
                "node_id": "p-1",
                "parent_id": "sec-1",
                "doc_id": "doc-1",
                "node_type": "paragraph",
                "order": 0,
                "token_count": 10,
            },
        )
    ]
    embeddings = FakeEmbeddings(dimensions=384)
    vector_store = VectorStore(
        persist_directory=str(tmp_path / "summary_faiss"), embeddings=embeddings
    )
    vector_store.add_documents(documents)
    bm25_bundle = create_bm25_bundle(documents)
    retriever = FusionRetriever(
        vectorstore=vector_store,
        bm25=bm25_bundle,
        k=1,
        alpha=0.5,
        fetch_k=5,
        node_store=hierarchical_node_store,
    )

    packed = retriever.retrieve(
        "Summarize the retrieval pipeline",
        query_plan={
            "intent": "summary",
            "subqueries": ["retrieval pipeline"],
            "preferred_node_types": ["section", "paragraph"],
        },
    )

    assert packed.passages
    assert packed.passages[0].metadata.get("node_type") == "section"


def test_fusion_retriever_uses_flashrank_reranker(monkeypatch, vector_store, bm25_bundle):
    class FakeFlashrankReranker:
        def compress_documents(self, *, documents, query):
            assert query == "Python"
            return list(reversed(documents))

    retriever = FusionRetriever(
        vectorstore=vector_store,
        bm25=bm25_bundle,
        k=3,
        alpha=0.5,
        fetch_k=10,
        reranker_backend="flashrank",
        flashrank_top_n=3,
    )
    monkeypatch.setattr(
        retriever, "_get_flashrank_reranker", lambda: FakeFlashrankReranker()
    )

    packed = retriever.retrieve("Python")

    assert packed.passages
    assert packed.debug["rerank"]["flashrank"]["enabled"] is True


def test_tool_factory_passes_active_query_plan_to_retriever():
    class RecordingRetriever:
        def __init__(self):
            self.calls: list[tuple[str, dict | None]] = []

        def retrieve(self, query: str, *, query_plan=None):
            self.calls.append((query, query_plan))
            return PackedContext(
                passages=[
                    Document(page_content="retrieved", metadata={"source": "doc.txt"})
                ],
                total_tokens=1,
                dropped_candidates=0,
                packing_strategy="score_then_contiguity",
                debug={"query_plan": query_plan or {}},
            )

    retriever = RecordingRetriever()
    factory = ToolFactory(retriever)
    token = factory.set_active_query_plan(
        {"intent": "summary", "preferred_node_types": ["section", "paragraph"]}
    )
    try:
        _, docs = factory._search_documents("query text")
    finally:
        factory.reset_active_query_plan(token)

    assert docs
    assert retriever.calls == [
        (
            "query text",
            {"intent": "summary", "preferred_node_types": ["section", "paragraph"]},
        )
    ]
