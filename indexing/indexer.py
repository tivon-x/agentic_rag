#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List
import os
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from indexing.bm25_index import create_bm25_index
from indexing.embeddings import get_embeddings
from indexing.vectorstore import VectorStore
from indexing.chunker import Chunker
from mappers import LOADER_MAPPING, CHUNER_MAPPING


class Indexer:
    def __init__(self, config: dict):
        self.config = config
        self._init_components()

    def _init_components(self):
        """
        初始化索引器的组件，包括分块器和文档嵌入器。
        """

        # 初始化分块器
        self.Chunker = self.__get_chunker()

        # 初始化向量数据库
        self.vector_store = self.__get_vectorstore(self.config)

    def __get_data_processor(self, file_path: str) -> List[tuple[list[Document], str]]:
        """
        根据文件路径获取数据处理器，返回处理后的数据和文件类型。
        Args:
          file_path (str): 文件路径

        Returns:
          List[tuple[list[Document], str]]: 处理后的数据和文件类型的列表
        """

        # 递归处理文件夹
        if os.path.isdir(file_path):
            results = []
            for filename in os.listdir(file_path):
                # 跳过隐藏文件
                if filename.startswith("."):
                    continue
                full_path = os.path.join(file_path, filename)
                try:
                    # 如果是文件，则表明接下来要处理
                    if os.path.isfile(full_path):
                        print("处理文件，路径为:", full_path)
                    results += self.__get_data_processor(full_path)
                except ValueError as e:
                    print(f"跳过 {full_path}: {str(e)}")
                    continue
            return results
        # 如果是文件，则根据后缀名选择处理器
        else:
            ext = Path(file_path).suffix.lower()  # 获取文件后缀名
            # 根据后缀名选择处理器
            loader_mapping = LOADER_MAPPING.get(ext)
            if loader_mapping is None:
                return []
            processor, loader_args = loader_mapping
            # 返回处理后的文件 + 后缀用来表示是图像还是文本
            return [
                (processor(**loader_args).process(file_path), file_path.split(".")[-1])
            ]

    def __get_chunker(self) -> Chunker:
        """
        获取分块器实例。

        Returns:
            Chunker: 分块器实例
        """
        chunker_config = self.config.get("chunker", {})
        # 获取分块器类型和参数
        chunker_type = chunker_config.get("type", "recursive")  # 默认使用递归分块器
        params = chunker_config.get("params", {})
        chunker = CHUNER_MAPPING.get(chunker_type)
        if chunker is None:
            raise ValueError(f"Indexer_get_chunker -> 未知分块器类型: {chunker_type}")
        # 实例化
        return chunker(**params)

    def __get_vectorstore(self, config: dict) -> VectorStore:
        """
        获取向量存储实例。

        Returns:
            VectorStore: 向量存储实例
        """
        # 使用统一的嵌入获取函数，优先云端其次 HuggingFace
        embeddings = get_embeddings(config)

        vectorstore_config = self.config.get("vectorstore", {})
        return VectorStore(
            embeddings=embeddings,
            persist_directory=vectorstore_config.get("persist_directory", None),
        )

    def index(self, file_path: str) -> None | tuple[VectorStore, BM25Okapi]:
        """
        索引文件
        Args:
            file_path (str): 文件路径

        Returns:
            None | tuple[VectorStore, BM25Okapi]: 返回向量存储和BM25索引，如果没有可分块的数据则返回None
        """
        datas = self.__get_data_processor(file_path)

        chunks: list[Document] = []
        for data, type in datas:
            if type in ["jpg", "jpeg", "png"]:
                pass  # 多模态信息除了图像就是文本, 先不处理图像
            else:
                chunks += self.Chunker.chunk(data)

        if not chunks:
            print("没有可分块的数据")
            return None
        # 构造向量存储
        self.vector_store.add_documents(chunks)

        # 构造BM25索引
        bm25_index = create_bm25_index(chunks)

        return self.vector_store, bm25_index
