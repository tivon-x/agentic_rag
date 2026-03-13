from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


@runtime_checkable
class VectorStore(Protocol):
    def add_documents(self, documents: list[Document]) -> None: ...
    def add_embeddings(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, object]] | None = None,
    ) -> None: ...
    def search(
        self,
        query: str,
        *,
        k: int = 10,
        filter: dict[str, object] | None = None,
        fetch_k: int = 20,
    ) -> list[Document]: ...
    def search_with_score(
        self,
        query: str,
        *,
        k: int = 10,
        filter: dict[str, object] | None = None,
        fetch_k: int = 20,
    ) -> list[tuple[Document, float]]: ...
    def get_all_documents(self) -> list[Document]: ...
    def save(self, persist_directory: str) -> None: ...
    def get_retriever(self, **search_kwargs) -> BaseRetriever: ...
