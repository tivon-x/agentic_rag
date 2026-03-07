import os
from typing import Literal
import faiss
import logging
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from uuid import uuid4

from langchain_core.vectorstores.base import VectorStoreRetriever


class VectorStore:
    def __init__(self, embeddings: Embeddings, persist_directory: str | None = None):
        """
        Initialize vector store.

        Args:
            embeddings (Embeddings): Embedding model instance.
            persist_directory (str): Persistence directory; loads FAISS index if exists.
        """
        logger = logging.getLogger(__name__)
        loaded = False
        if persist_directory:
            # langchain FAISS persistence expects index.faiss + index.pkl
            index_path = os.path.join(persist_directory, "index.faiss")
            pkl_path = os.path.join(persist_directory, "index.pkl")
            if os.path.isfile(index_path) and os.path.isfile(pkl_path):
                try:
                    self.__vectorstore = FAISS.load_local(
                        persist_directory,
                        embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    loaded = True
                except Exception as e:
                    logger.warning(
                        "Failed to load FAISS index from %s: %s; creating a new empty index",
                        persist_directory,
                        str(e),
                    )

        if not loaded:
            index = faiss.IndexFlatL2(len(embeddings.embed_query("dimension_probe")))
            self.__vectorstore = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={},
            )

    def add_documents(self, documents: list[Document]) -> None:
        """
        Add documents to vector store.

        Args:
            documents (list[Document]): Document list.
        """
        ids = [str(uuid4()) for _ in documents]
        self.__vectorstore.add_documents(documents=documents, ids=ids)

    def similarity_search(
        self, query: str, k: int = 10, filter: dict | None = None, fetch_k: int = 20
    ) -> list[Document]:
        """
        Similarity search based on cosine similarity.

        Args:
            query (str): Query string.
            k (int): Number of top results to return.
            filter (dict): Optional filter conditions based on vector store metadata.
            fetch_k (int): Number of documents to fetch before filtering, defaults to 20.

        Returns:
            list[Document]: Retrieved document list.
        """
        return self.__vectorstore.similarity_search(
            query, k=k, filter=filter, fetch_k=fetch_k
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        filter: dict | None = None,
        fetch_k: int = 20,
    ) -> list[tuple[Document, float]]:
        return self.__vectorstore.similarity_search_with_score(
            query, k=k, filter=filter, fetch_k=fetch_k
        )

    def get_all_documents(self) -> list[Document]:
        """
        Get all documents.

        Returns:
            list[Document]: All document list.
        """
        docstore = getattr(self.__vectorstore, "docstore", None)
        index_to_id = getattr(self.__vectorstore, "index_to_docstore_id", None)
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

    def save_local(self, persist_directory: str) -> None:
        os.makedirs(persist_directory, exist_ok=True)
        self.__vectorstore.save_local(persist_directory)

    def get_retriever(
        self,
        search_type: Literal[
            "similarity", "mmr", "similarity_score_threshold"
        ] = "similarity",
        **search_kwargs,
    ) -> VectorStoreRetriever:
        """
        Get retriever.

        Args:
            search_type (Literal["similarity", "mmr", "similarity_score_threshold"]): Retrieval type, defaults to "similarity".
            **search_kwargs: Other optional parameters.

        Returns:
            FAISS: Vector store retriever instance.
        """
        return self.__vectorstore.as_retriever(search_type=search_type, **search_kwargs)
