from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
import numpy as np
from indexing.vectorstore import VectorStore
from rank_bm25 import BM25Okapi


def get_similarity_retriever(vectorstore: VectorStore, k: int, filter: dict | None = None) -> BaseRetriever:
    """
    获取基于余弦相似度的检索器。

    Args:
        vectorstore (VectorStore): 向量存储实例。
        k (int): 返回的结果数量。
        filter (dict | None): 可选的过滤条件，基于向量存储的元数据。

    Returns:
        BaseRetriever: 基于余弦相似度的检索器。
    """
    return vectorstore.get_retriever(search_type="similarity", k=k, filter=filter)


class BM25Retriever(BaseRetriever):
    """
    基于 BM25 的检索器。
    """
    def __init__(self, bm25_index: BM25Okapi, corpus: list[str], k: int = 10):
        self.bm25_index = bm25_index
        self.corpus = corpus
        self.k = k

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        scores = self.bm25_index.get_scores(query.split())
        ranked_indices = scores.argsort()[::-1][:self.k]
        return [Document(page_content=self.corpus[i]) for i in ranked_indices]
    

class FusionRetriever(BaseRetriever):
    """
    融合检索器，结合了余弦相似度和 BM25 检索器。
    """

    def __init__(self, vectorstore: VectorStore, bm25_index: BM25Okapi, corpus: list[str], alpha: float = 0.5, k: int = 10):
        self.vectorstore = vectorstore
        self.bm25_index = bm25_index
        self.corpus = corpus
        self.alpha = alpha
        self.k = k

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:

        epsilon = 1e-8
        
        # 第一步，获取所有文档
        all_docs = self.vectorstore.get_all_documents()

        # 第二步，使用 BM25 检索器获取相关文档
        bm25_scores = self.bm25_index.get_scores(query.split())

        # 第三步，使用余弦相似度检索器获取相关文档
        vector_results = self.vectorstore.similarity_search(query, k=len(all_docs))

        # 第四步：归一化分数
        vector_scores = np.array([score for _, score in vector_results])
        vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)

        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) -  np.min(bm25_scores) + epsilon)

        #  第五步：融合分数
        combined_scores = self.alpha * vector_scores + (1 - self.alpha) * bm25_scores

        # 第六步：排序文档
        sorted_indices = np.argsort(combined_scores)[::-1]

        # 第七步：返回排序后的文档
        return [all_docs[i] for i in sorted_indices[:self.k]]