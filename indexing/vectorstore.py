from __future__ import annotations

import logging
import os
from typing import Literal
from uuid import uuid4

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.base import VectorStoreRetriever

from indexing.stores.sqlite_vec_store import SqliteVecVectorStore


class FaissVectorStore:
    """FAISS-backed vector store adapter."""

    def __init__(self, embeddings: Embeddings, persist_directory: str | None = None):
        logger = logging.getLogger(__name__)
        loaded = False
        if persist_directory:
            index_path = os.path.join(persist_directory, "index.faiss")
            pkl_path = os.path.join(persist_directory, "index.pkl")
            if os.path.isfile(index_path) and os.path.isfile(pkl_path):
                try:
                    self._vectorstore = FAISS.load_local(
                        persist_directory,
                        embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    loaded = True
                except Exception as exc:
                    logger.warning(
                        "Failed to load FAISS index from %s: %s; creating a new empty index",
                        persist_directory,
                        str(exc),
                    )

        if not loaded:
            index = faiss.IndexFlatL2(len(embeddings.embed_query("dimension_probe")))
            self._vectorstore = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={},
            )

    def add_documents(self, documents: list[Document]) -> None:
        ids = [str(uuid4()) for _ in documents]
        self._vectorstore.add_documents(documents=documents, ids=ids)

    def add_embeddings(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, object]] | None = None,
    ) -> None:
        ids = [str(uuid4()) for _ in texts]
        payloads = metadatas or [{} for _ in texts]
        self._vectorstore.add_embeddings(
            text_embeddings=list(zip(texts, embeddings, strict=False)),
            metadatas=payloads,
            ids=ids,
        )

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        filter: dict[str, object] | None = None,
        fetch_k: int = 20,
    ) -> list[Document]:
        return self._vectorstore.similarity_search(
            query,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: dict[str, object] | None = None,
        fetch_k: int = 20,
    ) -> list[Document]:
        return self.search(query, k=k, filter=filter, fetch_k=fetch_k)

    def search_with_score(
        self,
        query: str,
        *,
        k: int = 10,
        filter: dict[str, object] | None = None,
        fetch_k: int = 20,
    ) -> list[tuple[Document, float]]:
        return self._vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        filter: dict[str, object] | None = None,
        fetch_k: int = 20,
    ) -> list[tuple[Document, float]]:
        return self.search_with_score(query, k=k, filter=filter, fetch_k=fetch_k)

    def get_all_documents(self) -> list[Document]:
        docstore = getattr(self._vectorstore, "docstore", None)
        index_to_id = getattr(self._vectorstore, "index_to_docstore_id", None)
        if docstore is None or index_to_id is None:
            return []

        if hasattr(docstore, "_dict"):
            try:
                return list(docstore._dict.values())  # type: ignore[attr-defined]
            except Exception:
                logging.getLogger(__name__).debug("Failed to read docstore._dict")

        docs: list[Document] = []
        for doc_id in index_to_id.values():
            try:
                doc = docstore.search(doc_id)
            except Exception:
                doc = None
            if isinstance(doc, Document):
                docs.append(doc)
        return docs

    def save(self, persist_directory: str) -> None:
        os.makedirs(persist_directory, exist_ok=True)
        self._vectorstore.save_local(persist_directory)

    def save_local(self, persist_directory: str) -> None:
        self.save(persist_directory)

    def get_retriever(
        self,
        search_type: Literal[
            "similarity", "mmr", "similarity_score_threshold"
        ] = "similarity",
        **search_kwargs,
    ) -> VectorStoreRetriever:
        return self._vectorstore.as_retriever(
            search_type=search_type,
            **search_kwargs,
        )


def create_vector_store(
    backend: str,
    *,
    embeddings: Embeddings,
    persist_directory: str | None = None,
):
    normalized_backend = backend.strip().lower()
    if normalized_backend == "faiss":
        return FaissVectorStore(
            embeddings=embeddings,
            persist_directory=persist_directory,
        )
    if normalized_backend == "sqlite_vec":
        return SqliteVecVectorStore(
            embeddings=embeddings,
            persist_directory=persist_directory,
        )
    raise ValueError(f"Unsupported vector backend: {backend}")
