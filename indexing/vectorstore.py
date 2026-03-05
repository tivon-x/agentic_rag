import os
from typing import Literal
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from uuid import uuid4

from langchain_core.vectorstores.base import VectorStoreRetriever


class VectorStore:
    def __init__(self, embeddings: Embeddings, persist_directory: str | None = None):
        """
        初始化向量存储。

        Args:
            embeddings (Embeddings): 嵌入模型实例。
            persist_directory (str): 持久化目录，如果存在加载 FAISS 索引。
        """
        if persist_directory and os.path.exists(persist_directory):
            self.__vectorstore = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
        else:
            index = faiss.IndexFlatL2(len(embeddings.embed_query("test")))
            self.__vectorstore = FAISS(
                embedding_function=embeddings.embed_query,
                index=index,
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={}
            )

    def add_documents(self, documents: list[Document]) -> None:
        """
        添加文档到向量存储。

        Args:
            documents (list[Document]): 文档列表。
        """
        ids = [str(uuid4()) for _ in documents]
        self.__vectorstore.add_documents(documents=documents, ids=ids)

    
    def similarity_search(self, query: str, k: int = 10, filter: dict | None = None, fetch_k: int = 20) -> list[Document]:
        """
        基于余弦相似度搜索。

        Args:
            query (str): 查询字符串。
            k (int): 返回的顶部结果数量。
            filter (dict): 可选的过滤条件，基于向量存储的元数据。
            fetch_k (int): 在过滤之前需要获取的文档数量, 默认为 20。

        Returns:
            list[Document]: 检索到的文档列表。
        """
        return self.__vectorstore.similarity_search(query, k=k, filter=filter, fetch_k=fetch_k)
    
    def get_all_documents(self) -> list[Document]:
        """
        获取所有文档。

        Returns:
            list[Document]: 所有文档列表。
        """
        return self.__vectorstore.similarity_search("", k=self.__vectorstore.index.ntotal)
    
    def get_retriever(self, search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = "similarity", **search_kwargs) -> VectorStoreRetriever:
        """
        获取检索器。

        Args:
            search_type (Literal["similarity", "mmr", "similarity_score_threshold"]): 检索类型，默认为 "similarity"。
            **search_kwargs: 其他可选参数。

        Returns:
            FAISS: 向量存储的检索器实例。
        """
        return self.__vectorstore.as_retriever(search_type=search_type, **search_kwargs)