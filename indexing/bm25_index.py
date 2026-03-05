"""
BM25 索引模块
"""

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


def create_bm25_index(documents: list[Document]) -> BM25Okapi:
    """
    创建 BM25 索引。

    Args:
        documents (list[Document]): 文档列表。

    Returns:
        BM25Okapi: 返回 BM25 索引实例。
    """
    # 提取文档内容
    corpus = [doc.page_content for doc in documents]
    
    # 分词处理
    tokenized_corpus = [doc.split() for doc in corpus]
    
    # 创建 BM25 索引
    bm25_index = BM25Okapi(tokenized_corpus)
    
    return bm25_index