from __future__ import annotations

import pytest

from langchain_core.documents import Document

from indexing.bm25_index import BM25Bundle, create_lexical_store
from indexing.embeddings import FakeEmbeddings
from indexing.models.node import Node
from indexing.stores.node_store import JsonNodeStore, create_node_store
from indexing.vectorstore import create_vector_store


def test_json_node_store_save_nodes_supports_direct_node_access(tmp_path):
    node_store = JsonNodeStore(tmp_path / "nodes.jsonl", tmp_path / "doc_trees.json")
    nodes = [
        Node(
            node_id="doc-1",
            parent_id=None,
            doc_id="doc-1",
            node_type="document",
            title="Doc",
            text="Root",
            order=0,
            level=0,
            metadata={"source": "doc.md"},
        ),
        Node(
            node_id="p-1",
            parent_id="doc-1",
            doc_id="doc-1",
            node_type="paragraph",
            title=None,
            text="Paragraph",
            order=0,
            level=1,
            metadata={"source": "doc.md"},
        ),
    ]

    node_store.save_nodes(nodes)

    assert node_store.get_node("p-1") is not None
    assert [child.node_id for child in node_store.get_children("doc-1")] == ["p-1"]


def test_create_vector_store_rejects_unsupported_backend(tmp_path):
    embeddings = FakeEmbeddings(dimensions=8)

    with pytest.raises(ValueError, match="Unsupported vector backend"):
        create_vector_store(
            "unknown",
            embeddings=embeddings,
            persist_directory=str(tmp_path / "index"),
        )


def test_create_vector_store_reserves_sqlite_vec_backend(tmp_path):
    embeddings = FakeEmbeddings(dimensions=8)

    with pytest.raises(NotImplementedError, match="sqlite-vec backend"):
        create_vector_store(
            "sqlite_vec",
            embeddings=embeddings,
            persist_directory=str(tmp_path / "index"),
        )


def test_create_lexical_store_returns_bm25_bundle():
    documents = [Document(page_content="alpha beta", metadata={"source": "doc.txt"})]

    lexical_store = create_lexical_store("bm25", documents=documents)

    assert isinstance(lexical_store, BM25Bundle)
    assert lexical_store.query("alpha", k=1)


def test_create_node_store_returns_json_store(tmp_path):
    node_store = create_node_store(
        "json",
        nodes_path=tmp_path / "nodes.jsonl",
        doc_trees_path=tmp_path / "doc_trees.json",
    )

    assert isinstance(node_store, JsonNodeStore)


def test_create_node_store_reserves_sqlite_backend(tmp_path):
    with pytest.raises(NotImplementedError, match="SQLite node store"):
        create_node_store(
            "sqlite",
            nodes_path=tmp_path / "nodes.jsonl",
            doc_trees_path=tmp_path / "doc_trees.json",
        )
