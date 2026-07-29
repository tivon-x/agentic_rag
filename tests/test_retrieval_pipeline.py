from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
import pytest

from indexing.models.node import Node
from indexing.retrieval_pipeline import (
    PIPELINE_REGISTRY,
    PackedContext,
    RetrievalCandidate,
    get_pipeline_config,
    prepare_index_documents,
)
from indexing.retriever import FusionRetriever


@dataclass
class _VectorStoreStub:
    results: list[tuple[Document, float]]

    def add_documents(self, documents: list[Document]) -> None:
        self.results = [(document, 0.0) for document in documents]

    def add_embeddings(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, object]] | None = None,
    ) -> None:
        return None

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        filter: dict[str, object] | None = None,
        fetch_k: int = 20,
    ) -> list[Document]:
        return [document for document, _ in self.search_with_score(query, k=k)]

    def search_with_score(
        self,
        query: str,
        *,
        k: int = 10,
        filter: dict[str, object] | None = None,
        fetch_k: int = 20,
    ) -> list[tuple[Document, float]]:
        return self.results[:k]

    def get_all_documents(self) -> list[Document]:
        return [document for document, _ in self.results]

    def save(self, persist_directory: str) -> None:
        return None

    def get_retriever(self, **search_kwargs):
        raise NotImplementedError


@dataclass
class _LexicalStoreStub:
    results: list[tuple[Document, float]]

    def build(self, documents: list[Document]) -> None:
        self.results = [(document, 0.0) for document in documents]

    def query(self, query: str, *, k: int = 10) -> list[Document]:
        return [document for document, _ in self.topk_with_scores(query, k=k)]

    def topk_with_scores(
        self,
        query: str,
        *,
        k: int = 10,
    ) -> list[tuple[Document, float]]:
        return self.results[:k]


class _NodeStoreStub:
    def __init__(self, nodes: list[Node]):
        self._nodes = {node.node_id: node for node in nodes}
        self._children: dict[str, list[Node]] = {}
        for node in nodes:
            if node.parent_id is None:
                continue
            self._children.setdefault(node.parent_id, []).append(node)
        for siblings in self._children.values():
            siblings.sort(key=lambda item: item.order)

    def save_nodes(self, nodes: list[Node]) -> None:
        raise NotImplementedError

    def save_trees(self, trees: list[Any]) -> None:
        raise NotImplementedError

    def load_nodes(self) -> list[Node]:
        return list(self._nodes.values())

    def load_trees(self) -> dict[str, Any]:
        raise NotImplementedError

    def get_node(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    def get_children(self, node_id: str) -> list[Node]:
        return list(self._children.get(node_id, []))

    def get_parent(self, node_id: str) -> Node | None:
        node = self._nodes.get(node_id)
        if node is None or node.parent_id is None:
            return None
        return self._nodes.get(node.parent_id)


def _doc(
    node_id: str,
    text: str,
    *,
    score: float = 0.2,
    source: str = "guide.md",
    parent_id: str = "",
    node_type: str = "paragraph",
    title: str = "",
    order: int = 0,
    token_count: int = 12,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        document=Document(
            page_content=text,
            metadata={
                "node_id": node_id,
                "source": source,
                "parent_id": parent_id,
                "node_type": node_type,
                "title": title,
                "order": order,
                "token_count": token_count,
            },
        ),
        score=score,
    )


def _make_retriever(
    *,
    vector_results: list[tuple[Document, float]] | None = None,
    lexical_results: list[tuple[Document, float]] | None = None,
    node_store: _NodeStoreStub | None = None,
    corpus_profile: dict[str, Any] | None = None,
    k: int = 3,
    fetch_k: int = 6,
    token_budget: int = 60,
    reranker_backend: str = "none",
    pipeline_name: str = "b1",
    strict_reranker: bool = False,
) -> FusionRetriever:
    return FusionRetriever(
        vectorstore=_VectorStoreStub(vector_results or []),
        lexical_store=_LexicalStoreStub(lexical_results or []),
        k=k,
        fetch_k=fetch_k,
        token_budget=token_budget,
        node_store=node_store,
        corpus_profile=corpus_profile,
        reranker_backend=reranker_backend,
        strict_reranker=strict_reranker,
        pipeline=get_pipeline_config(pipeline_name),
    )


def test_dedupe_candidates_removes_duplicate_text_and_keeps_highest_score():
    retriever = _make_retriever()
    candidates = [
        _doc("node-a", "Same passage text", score=0.15),
        _doc("node-b", "Same passage text", score=0.42),
        _doc("node-c", "Unique passage text", score=0.3),
    ]

    deduped, debug = retriever._dedupe_candidates(candidates)

    assert len(deduped) == 2
    assert {item.document.metadata["node_id"] for item in deduped} == {"node-a", "node-c"}
    assert next(item for item in deduped if item.document.metadata["node_id"] == "node-a").score == 0.42
    assert debug["merge_log"] == ["text:node-b->node-a"]


def test_dedupe_candidates_keeps_adjacent_siblings_available_for_window_merge():
    retriever = _make_retriever()
    candidates = [
        _doc("p-1", "Paragraph one", parent_id="sec-1", order=0),
        _doc("p-2", "Paragraph two", parent_id="sec-1", order=1),
    ]

    deduped, debug = retriever._dedupe_candidates(candidates)

    assert len(deduped) == 2
    assert debug["deduped_count"] == 2
    assert not debug["merge_log"]


def test_dedupe_candidates_handles_empty_input():
    retriever = _make_retriever()

    deduped, debug = retriever._dedupe_candidates([])

    assert deduped == []
    assert debug == {"raw_count": 0, "deduped_count": 0, "merge_log": []}


def test_rerank_candidates_applies_title_and_node_type_boosts():
    retriever = _make_retriever()
    candidates = [
        _doc(
            "section-1",
            "Detailed explanation of the pipeline.",
            node_type="section",
            title="Retrieval Pipeline Overview",
            score=0.2,
        ),
        _doc(
            "paragraph-1",
            "Background details.",
            node_type="paragraph",
            title="Background",
            score=0.2,
        ),
    ]

    reranked, _ = retriever._rerank_candidates(
        "retrieval pipeline",
        candidates,
        {"preferred_node_types": ["section"]},
    )

    assert reranked[0].document.metadata["node_id"] == "section-1"
    assert "title_match" in reranked[0].boosts
    assert reranked[0].boosts["node_type_match"] == 0.08


def test_rerank_candidates_uses_corpus_profile_priors():
    retriever = _make_retriever(
        corpus_profile={
            "coverage": "agentic rag retrieval pipeline",
            "domain_keywords": ["rerank"],
            "primary_entities": ["FusionRetriever"],
            "non_coverage": "stock price finance",
        }
    )
    candidates = [
        _doc(
            "good",
            "FusionRetriever applies rerank to the retrieval pipeline.",
            score=0.2,
        ),
        _doc(
            "bad",
            "Stock price finance outlook for next quarter.",
            score=0.2,
        ),
    ]

    reranked, debug = retriever._rerank_candidates(
        "How does rerank work in FusionRetriever?",
        candidates,
        {"preferred_node_types": ["paragraph"]},
    )

    assert reranked[0].document.metadata["node_id"] == "good"
    assert reranked[0].boosts["domain_keyword"] > 0
    assert reranked[0].boosts["primary_entity"] > 0
    assert reranked[1].boosts["non_coverage_penalty"] < 0
    assert debug["top_candidates"][0]["node_id"] == "good"


def test_rerank_candidates_gracefully_falls_back_when_flashrank_unavailable(monkeypatch):
    retriever = _make_retriever(reranker_backend="flashrank")
    candidates = [_doc("node-1", "retrieval pipeline", score=0.3)]
    monkeypatch.setattr(
        retriever,
        "_get_flashrank_reranker",
        lambda: (_ for _ in ()).throw(RuntimeError("flashrank unavailable")),
    )

    reranked, debug = retriever._rerank_candidates(
        "retrieval pipeline",
        candidates,
        {"preferred_node_types": ["paragraph"]},
    )

    assert reranked == candidates
    assert debug["flashrank"]["enabled"] is False
    assert "flashrank unavailable" in debug["flashrank"]["error"]


def test_rerank_candidates_fails_when_strict_flashrank_unavailable(
    monkeypatch,
):
    retriever = _make_retriever(
        reranker_backend="flashrank",
        strict_reranker=True,
    )
    monkeypatch.setattr(
        retriever,
        "_get_flashrank_reranker",
        lambda: (_ for _ in ()).throw(RuntimeError("flashrank unavailable")),
    )

    with pytest.raises(RuntimeError, match="Required reranker is unavailable"):
        retriever._rerank_candidates(
            "query",
            [_doc("node-a", "candidate")],
            {"preferred_node_types": ["paragraph"]},
        )


def test_flashrank_cache_dir_is_passed_to_ranker(monkeypatch):
    captured: dict[str, Any] = {}

    class _Ranker:
        def __init__(self, **kwargs):
            captured["ranker"] = kwargs

    class _Compressor:
        def __init__(self, **kwargs):
            captured["compressor"] = kwargs

    monkeypatch.setattr("flashrank.Ranker", _Ranker)
    monkeypatch.setattr(
        "langchain_community.document_compressors.FlashrankRerank",
        _Compressor,
    )
    retriever = FusionRetriever(
        vectorstore=_VectorStoreStub([]),
        lexical_store=_LexicalStoreStub([]),
        reranker_backend="flashrank",
        flashrank_cache_dir="artifact-cache",
        pipeline=get_pipeline_config("b1"),
    )

    compressor = retriever._get_flashrank_reranker()

    assert isinstance(compressor, _Compressor)
    assert captured["ranker"] == {
        "model_name": "ms-marco-TinyBERT-L-2-v2",
        "cache_dir": "artifact-cache",
    }
    assert captured["compressor"]["client"].__class__ is _Ranker


def test_pack_context_respects_token_budget_and_prefers_higher_scores():
    retriever = _make_retriever(k=3, token_budget=210)
    candidates = [
        _doc("high", "High priority", score=0.9, token_count=100),
        _doc("mid", "Mid priority", score=0.5, token_count=100),
        _doc("low", "Low priority", score=0.3, token_count=100),
    ]

    packed = retriever._pack_context(
        candidates,
        {"intent": "fact", "preferred_node_types": ["paragraph"]},
        {"raw_candidates": 3},
        {"raw_count": 3, "deduped_count": 3, "merge_log": []},
        {"top_candidates": []},
    )

    assert [doc.metadata["node_id"] for doc in packed.passages] == ["high", "mid"]
    assert packed.total_tokens == 200
    assert packed.dropped_candidates == 1


def test_pack_context_expands_paragraph_to_parent_section_for_summary_queries():
    nodes = [
        Node(
            node_id="sec-1",
            parent_id="doc-1",
            doc_id="doc-1",
            node_type="section",
            title="Intro",
            text="Paragraph one.\n\nParagraph two.",
            order=0,
            level=1,
            metadata={"source": "guide.md"},
            token_count=20,
        ),
        Node(
            node_id="p-1",
            parent_id="sec-1",
            doc_id="doc-1",
            node_type="paragraph",
            title=None,
            text="Paragraph one.",
            order=0,
            level=2,
            metadata={"source": "guide.md"},
            token_count=8,
        ),
    ]
    retriever = _make_retriever(
        node_store=_NodeStoreStub(nodes),
        token_budget=50,
        pipeline_name="b3",
    )
    candidates = [_doc("p-1", "Paragraph one.", parent_id="sec-1", token_count=8)]

    packed = retriever._pack_context(
        candidates,
        {"intent": "summary", "preferred_node_types": ["section", "paragraph"]},
        {"raw_candidates": 1},
        {"raw_count": 1, "deduped_count": 1, "merge_log": []},
        {"top_candidates": []},
    )

    assert len(packed.passages) == 2
    assert packed.passages[0].metadata["node_id"] == "p-1"
    assert packed.passages[1].metadata["node_type"] == "section"
    assert packed.passages[1].metadata["is_parent_context"] is True


def test_fusion_retriever_retrieve_returns_packed_context_with_debug():
    paragraph = Document(
        page_content="Fusion retrieval pipeline overview.",
        metadata={
            "node_id": "p-1",
            "source": "guide.md",
            "parent_id": "sec-1",
            "node_type": "paragraph",
            "title": "Pipeline",
            "order": 0,
            "token_count": 8,
        },
    )
    retriever = _make_retriever(
        vector_results=[(paragraph, 0.1)],
        lexical_results=[(paragraph, 1.0)],
        k=1,
        fetch_k=2,
        token_budget=20,
    )

    packed = retriever.retrieve(
        "fusion retrieval pipeline",
        query_plan={
            "intent": "fact",
            "subqueries": ["fusion retrieval pipeline"],
            "preferred_node_types": ["paragraph"],
        },
    )

    assert isinstance(packed, PackedContext)
    assert len(packed.passages) == 1
    assert packed.debug["query_plan"]["subqueries"] == ["fusion retrieval pipeline"]
    assert packed.debug["dedupe"]["raw_count"] >= packed.debug["dedupe"]["deduped_count"]
    assert "rerank" in packed.debug


def test_registry_freezes_core_and_ablation_contracts():
    assert PIPELINE_REGISTRY["b0"].use_rerank is False
    assert PIPELINE_REGISTRY["b1"].fusion_method == "minmax"
    assert PIPELINE_REGISTRY["b2"].rrf_k == 60
    assert PIPELINE_REGISTRY["b2"].neighbor_window == 0
    assert PIPELINE_REGISTRY["b3"].neighbor_window == 1
    assert PIPELINE_REGISTRY["b2_no_metadata"].use_metadata_prefix is False
    assert PIPELINE_REGISTRY["b2_no_sparse"].use_sparse is False
    assert PIPELINE_REGISTRY["b2_no_dense"].use_dense is False
    assert PIPELINE_REGISTRY["b2_minmax"].fusion_method == "minmax"
    assert PIPELINE_REGISTRY["b2_no_rerank"].use_rerank is False


def test_metadata_prefix_is_index_only_and_quote_remains_source_faithful():
    document = Document(
        page_content=(
            "[TITLE] Attention Is All You Need\n"
            "[SECTION] 3.2 Attention\n"
            "[BLOCK] paragraph\n"
            "The source quote."
        ),
        metadata={
            "retrieval_text": (
                "[TITLE] Attention Is All You Need\n"
                "[SECTION] 3.2 Attention\n"
                "[BLOCK] paragraph\n"
                "The source quote."
            ),
            "quote_text": "The source quote.",
        },
    )

    b1_document = prepare_index_documents(
        [document],
        get_pipeline_config("b1"),
    )[0]
    b2_document = prepare_index_documents(
        [document],
        get_pipeline_config("b2"),
    )[0]

    assert b1_document.page_content == "The source quote."
    assert "[TITLE] Attention Is All You Need" in b2_document.page_content
    assert b2_document.metadata["quote_text"] == "The source quote."
