from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.documents import Document


@runtime_checkable
class LexicalStore(Protocol):
    def build(self, documents: list[Document]) -> None: ...
    def query(self, query: str, *, k: int = 10) -> list[Document]: ...
    def topk_with_scores(
        self,
        query: str,
        *,
        k: int = 10,
    ) -> list[tuple[Document, float]]: ...
