from __future__ import annotations

from langchain_core.embeddings import Embeddings


class SqliteVecVectorStore:
    """Reserved adapter slot for a future sqlite-vec backend."""

    def __init__(self, embeddings: Embeddings, persist_directory: str | None = None):
        raise NotImplementedError(
            "sqlite-vec backend is not implemented yet. Set VECTOR_BACKEND=faiss."
        )
