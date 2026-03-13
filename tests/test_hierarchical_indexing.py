from __future__ import annotations

import json

import tiktoken

from indexing.indexer import Indexer
from indexing.parsers.txt_parser import TxtHierarchicalParser
from indexing.stores.node_store import JsonNodeStore
from indexing.token_count import estimate_token_count


def test_hierarchical_indexer_builds_tree_and_persists_nodes(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text(
        "# Intro\n\nThis is the first paragraph.\n\n## Details\n\nThis is the second paragraph.",
        encoding="utf-8",
    )

    config = {
        "embedding": {"type": "fake", "dimensions": 16},
        "index_mode": "hierarchical",
        "leaf_node_type": "paragraph",
        "parent_embed_pooling": "mean",
        "chunker": {"type": "recursive", "params": {}},
        "vectorstore": {"persist_directory": str(tmp_path / "faiss")},
        "bm25_path": str(tmp_path / "bm25.pkl"),
        "nodes_path": str(tmp_path / "nodes.jsonl"),
        "doc_trees_path": str(tmp_path / "doc_trees.json"),
    }

    indexer = Indexer(config)
    result = indexer.index(str(source))

    assert result is not None
    _, bm25_bundle = result
    assert bm25_bundle.documents

    node_store = JsonNodeStore(tmp_path / "nodes.jsonl", tmp_path / "doc_trees.json")
    nodes = node_store.load_nodes()
    assert any(node.node_type == "document" for node in nodes)
    assert any(node.node_type == "section" for node in nodes)
    paragraph_nodes = [node for node in nodes if node.node_type == "paragraph"]
    assert len(paragraph_nodes) == 2
    assert all(node.embedding is None for node in paragraph_nodes)

    trees_payload = json.loads((tmp_path / "doc_trees.json").read_text(encoding="utf-8"))
    assert len(trees_payload) == 1
    tree = next(iter(trees_payload.values()))
    assert tree["root_id"]
    assert tree["children_by_parent"]
    assert "nodes" not in tree

    retrieved_parent = node_store.get_parent(paragraph_nodes[0].node_id)
    assert retrieved_parent is not None
    assert retrieved_parent.node_type == "section"


def test_hierarchical_indexer_falls_back_to_flat_mode_when_requested(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")

    config = {
        "embedding": {"type": "fake", "dimensions": 16},
        "index_mode": "flat",
        "chunker": {"type": "recursive", "params": {"chunk_size": 32, "chunk_overlap": 0}},
        "vectorstore": {"persist_directory": str(tmp_path / "faiss")},
        "bm25_path": str(tmp_path / "bm25.pkl"),
        "nodes_path": str(tmp_path / "nodes.jsonl"),
        "doc_trees_path": str(tmp_path / "doc_trees.json"),
    }

    indexer = Indexer(config)
    result = indexer.index(str(source))

    assert result is not None
    assert not (tmp_path / "nodes.jsonl").exists()
    assert not (tmp_path / "doc_trees.json").exists()


def test_hierarchical_indexer_supports_section_leaf_nodes(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text(
        "# Intro\n\nAlpha paragraph.\n\n## Details\n\nBeta paragraph.",
        encoding="utf-8",
    )

    config = {
        "embedding": {"type": "fake", "dimensions": 16},
        "index_mode": "hierarchical",
        "leaf_node_type": "section",
        "parent_embed_pooling": "none",
        "chunker": {"type": "recursive", "params": {}},
        "vectorstore": {"persist_directory": str(tmp_path / "faiss")},
        "bm25_path": str(tmp_path / "bm25.pkl"),
        "nodes_path": str(tmp_path / "nodes.jsonl"),
        "doc_trees_path": str(tmp_path / "doc_trees.json"),
    }

    indexer = Indexer(config)
    result = indexer.index(str(source))

    assert result is not None
    docs = result[0].get_all_documents()
    assert any(doc.metadata["node_type"] == "section" for doc in docs)
    assert all(doc.metadata["node_type"] != "paragraph" for doc in docs)


def test_estimate_token_count_handles_cjk_text():
    encoding = tiktoken.get_encoding("cl100k_base")
    cjk_text = "这是一个中文句子。"
    english_text = "plain english words"

    assert estimate_token_count(cjk_text) == len(encoding.encode(cjk_text))
    assert estimate_token_count(english_text) == len(encoding.encode(english_text))


def test_txt_parser_avoids_false_positive_colon_lines(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text(
        "We observed:\nThe system remained stable throughout the run.\n",
        encoding="utf-8",
    )

    tree = TxtHierarchicalParser().parse(str(source))
    assert not any(node.node_type == "section" for node in tree.nodes)


def test_json_node_store_uses_cache_after_initial_load(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text("# Intro\n\nParagraph one.", encoding="utf-8")

    config = {
        "embedding": {"type": "fake", "dimensions": 16},
        "index_mode": "hierarchical",
        "leaf_node_type": "paragraph",
        "parent_embed_pooling": "mean",
        "chunker": {"type": "recursive", "params": {}},
        "vectorstore": {"persist_directory": str(tmp_path / "faiss")},
        "bm25_path": str(tmp_path / "bm25.pkl"),
        "nodes_path": str(tmp_path / "nodes.jsonl"),
        "doc_trees_path": str(tmp_path / "doc_trees.json"),
    }

    Indexer(config).index(str(source))
    node_store = JsonNodeStore(tmp_path / "nodes.jsonl", tmp_path / "doc_trees.json")

    first_nodes = node_store.load_nodes()
    original_cache_mtime = node_store._nodes_cache_mtime_ns
    second_nodes = node_store.load_nodes()

    assert first_nodes
    assert second_nodes
    assert node_store._nodes_cache_mtime_ns == original_cache_mtime
    assert first_nodes[0].node_id == second_nodes[0].node_id
