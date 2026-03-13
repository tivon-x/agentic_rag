"""BM25 index module.

Provides:
- Persistable BM25Bundle (containing original Document list + tokenized_corpus)
- Bundle-based retrieval
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from indexing.stores.lexical_store import LexicalStore


@dataclass
class BM25Bundle:
    """BM25-backed lexical store adapter."""

    documents: list[Document]
    tokenized_corpus: list[list[str]]
    _bm25: BM25Okapi | None = None

    def build(self, documents: list[Document]) -> None:
        self.documents = [doc for doc in documents if (doc.page_content or "").strip()]
        self.tokenized_corpus = [doc.page_content.split() for doc in self.documents]
        self.rebuild_index()

    def rebuild_index(self) -> None:
        if not self.tokenized_corpus:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(self.tokenized_corpus)

    @property
    def bm25_index(self) -> BM25Okapi:
        if self._bm25 is None:
            self.rebuild_index()
        assert self._bm25 is not None
        return self._bm25

    def query(self, query: str, *, k: int = 10) -> list[Document]:
        scores = self.bm25_index.get_scores(query.split())
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
        return [self.documents[i] for i in ranked]

    def topk_with_scores(
        self, query: str, *, k: int = 10
    ) -> list[tuple[Document, float]]:
        scores = self.bm25_index.get_scores(query.split())
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
        return [(self.documents[i], float(scores[i])) for i in ranked]


def create_bm25_bundle(documents: list[Document]) -> BM25Bundle:
    bundle = BM25Bundle(documents=[], tokenized_corpus=[])
    bundle.build(documents)
    return bundle


def create_bm25_index(documents: list[Document]) -> BM25Okapi:
    """Backward-compatible helper: return only BM25Okapi."""
    return create_bm25_bundle(documents).bm25_index


def create_lexical_store(
    backend: str,
    *,
    documents: list[Document] | None = None,
    bundle: BM25Bundle | None = None,
) -> LexicalStore:
    normalized_backend = backend.strip().lower()
    if normalized_backend != "bm25":
        raise ValueError(f"Unsupported lexical backend: {backend}")
    if bundle is not None:
        return bundle
    if documents is None:
        raise ValueError("documents are required when creating a BM25 lexical store")
    return create_bm25_bundle(documents)
