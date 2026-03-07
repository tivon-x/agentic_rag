"""BM25 index module.

Provides:
- Persistable BM25Bundle (containing original Document list + tokenized_corpus)
- Bundle-based retrieval
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


@dataclass
class BM25Bundle:
    documents: list[Document]
    tokenized_corpus: list[list[str]]
    _bm25: BM25Okapi | None = None

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
    corpus = [doc.page_content for doc in documents]
    tokenized_corpus = [text.split() for text in corpus if (text or "").strip()]
    kept_docs: list[Document] = [
        doc for doc in documents if (doc.page_content or "").strip()
    ]
    bundle = BM25Bundle(documents=kept_docs, tokenized_corpus=tokenized_corpus)
    bundle.rebuild_index()
    return bundle


def create_bm25_index(documents: list[Document]) -> BM25Okapi:
    """Backward-compatible helper: return only BM25Okapi."""
    return create_bm25_bundle(documents).bm25_index
