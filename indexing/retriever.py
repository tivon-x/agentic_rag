from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from indexing.vectorstore import VectorStore
from indexing.bm25_index import BM25Bundle
import hashlib


def get_similarity_retriever(
    vectorstore: VectorStore, k: int, filter: dict | None = None
) -> BaseRetriever:
    """
    Get cosine similarity-based retriever.

    Args:
        vectorstore (VectorStore): Vector store instance.
        k (int): Number of results to return.
        filter (dict | None): Optional filter conditions based on vector store metadata.

    Returns:
        BaseRetriever: Cosine similarity-based retriever.
    """
    return vectorstore.get_retriever(search_type="similarity", k=k, filter=filter)


class BM25Retriever(BaseRetriever):
    """
    BM25-based retriever.
    """

    bundle: BM25Bundle
    k: int = 10

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.bundle.query(query, k=self.k)


class FusionRetriever(BaseRetriever):
    """
    Fusion retriever combining cosine similarity and BM25 retrievers.
    """

    vectorstore: VectorStore
    bm25: BM25Bundle
    alpha: float = 0.5
    k: int = 10
    fetch_k: int = 40

    def model_post_init(self, __context) -> None:  # pydantic v2 hook
        if self.fetch_k <= 0:
            self.fetch_k = max(40, self.k * 4)
        else:
            self.fetch_k = max(self.fetch_k, self.k)

    @staticmethod
    def _doc_key(doc: Document) -> str:
        src = str(doc.metadata.get("source", ""))
        page = str(doc.metadata.get("page", ""))
        raw = (src + "|" + page + "|" + doc.page_content).encode(
            "utf-8", errors="ignore"
        )
        return hashlib.sha1(raw).hexdigest()

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        epsilon = 1e-8

        # Vector search: FAISS returns (doc, distance); smaller distance is better.
        vec_results = self.vectorstore.similarity_search_with_score(
            query, k=self.fetch_k
        )
        vec_sims: list[float] = [1.0 / (dist + epsilon) for _, dist in vec_results]
        if vec_sims:
            vmin, vmax = float(min(vec_sims)), float(max(vec_sims))
        else:
            vmin = vmax = 0.0

        vec_score_by_key: dict[str, float] = {}
        doc_by_key: dict[str, Document] = {}
        for (doc, dist), sim in zip(vec_results, vec_sims, strict=False):
            key = self._doc_key(doc)
            doc_by_key.setdefault(key, doc)
            if vmax - vmin <= epsilon:
                vec_score_by_key[key] = 0.0
            else:
                vec_score_by_key[key] = (sim - vmin) / (vmax - vmin)

        bm_results = self.bm25.topk_with_scores(query, k=self.fetch_k)
        bm_scores = [s for _, s in bm_results]
        if bm_scores:
            bmin, bmax = float(min(bm_scores)), float(max(bm_scores))
        else:
            bmin = bmax = 0.0

        bm_score_by_key: dict[str, float] = {}
        for doc, score in bm_results:
            key = self._doc_key(doc)
            doc_by_key.setdefault(key, doc)
            if bmax - bmin <= epsilon:
                bm_score_by_key[key] = 0.0
            else:
                bm_score_by_key[key] = (score - bmin) / (bmax - bmin)

        combined: list[tuple[str, float]] = []
        for key in doc_by_key.keys():
            vs = vec_score_by_key.get(key, 0.0)
            bs = bm_score_by_key.get(key, 0.0)
            combined.append((key, self.alpha * vs + (1.0 - self.alpha) * bs))
        combined.sort(key=lambda x: x[1], reverse=True)

        return [doc_by_key[key] for key, _ in combined[: self.k]]
