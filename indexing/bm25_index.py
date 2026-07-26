"""Persistable BM25 index with deterministic English/Chinese tokenization."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from indexing.stores.lexical_store import LexicalStore


_SEGMENT_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]+|"
    r"[A-Za-z]+(?:[-.][A-Za-z0-9]+)*|"
    r"\d+(?:\.\d+)*(?:%|[A-Za-z]+)?",
)


def tokenize_whitespace(text: str) -> list[str]:
    return [token.casefold() for token in text.split() if token.strip()]


def tokenize_mixed(text: str) -> list[str]:
    """Tokenize Chinese with jieba search mode while preserving scholarly terms."""
    normalized = unicodedata.normalize("NFKC", text or "")
    tokens: list[str] = []
    for segment in _SEGMENT_RE.findall(normalized):
        if any("\u3400" <= char <= "\u9fff" for char in segment):
            tokens.extend(
                token.casefold()
                for token in jieba.cut_for_search(segment)
                if token.strip()
            )
        else:
            tokens.append(segment.casefold())
    return tokens


def tokenize(text: str, tokenizer: str) -> list[str]:
    if tokenizer == "whitespace_v1":
        return tokenize_whitespace(text)
    if tokenizer == "mixed_v1":
        return tokenize_mixed(text)
    raise ValueError(f"Unsupported BM25 tokenizer: {tokenizer}")


@dataclass
class BM25Bundle:
    """BM25-backed lexical store adapter."""

    documents: list[Document]
    tokenized_corpus: list[list[str]]
    tokenizer: str = "whitespace_v1"
    _bm25: BM25Okapi | None = None

    def build(self, documents: list[Document]) -> None:
        self.documents = [
            doc for doc in documents if (doc.page_content or "").strip()
        ]
        self.tokenized_corpus = [
            tokenize(doc.page_content, self.tokenizer)
            for doc in self.documents
        ]
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
        return [
            document
            for document, _ in self.topk_with_scores(query, k=k)
        ]

    def topk_with_scores(
        self, query: str, *, k: int = 10
    ) -> list[tuple[Document, float]]:
        scores = self.bm25_index.get_scores(tokenize(query, self.tokenizer))
        ranked = sorted(
            range(len(scores)),
            key=scores.__getitem__,
            reverse=True,
        )[:k]
        return [
            (self.documents[index], float(scores[index]))
            for index in ranked
        ]


def create_bm25_bundle(
    documents: list[Document],
    *,
    tokenizer: str = "whitespace_v1",
) -> BM25Bundle:
    bundle = BM25Bundle(
        documents=[],
        tokenized_corpus=[],
        tokenizer=tokenizer,
    )
    bundle.build(documents)
    return bundle


def create_bm25_index(documents: list[Document]) -> BM25Okapi:
    """Backward-compatible helper returning only BM25Okapi."""
    return create_bm25_bundle(documents).bm25_index


def create_lexical_store(
    backend: str,
    *,
    documents: list[Document] | None = None,
    bundle: BM25Bundle | None = None,
    tokenizer: str = "whitespace_v1",
) -> LexicalStore:
    normalized_backend = backend.strip().lower()
    if normalized_backend != "bm25":
        raise ValueError(f"Unsupported lexical backend: {backend}")
    if bundle is not None:
        bundle_tokenizer = getattr(bundle, "tokenizer", "whitespace_v1")
        if bundle_tokenizer != tokenizer:
            raise ValueError(
                "BM25 tokenizer is incompatible "
                f"(index={bundle_tokenizer!r}, query={tokenizer!r}); "
                "rebuild the index."
            )
        return bundle
    if documents is None:
        raise ValueError(
            "documents are required when creating a BM25 lexical store"
        )
    return create_bm25_bundle(documents, tokenizer=tokenizer)
