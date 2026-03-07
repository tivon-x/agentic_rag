#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List
import os
import logging
from langchain_core.documents import Document

from indexing.bm25_index import BM25Bundle, create_bm25_bundle
from indexing.embeddings import get_embeddings
from indexing.vectorstore import VectorStore
from indexing.chunker import Chunker
from indexing.mappers import LOADER_MAPPING, CHUNKER_MAPPING

from core.persistence import save_bm25_bundle


class Indexer:
    def __init__(self, config: dict):
        self.config = config
        self._logger = logging.getLogger(__name__)
        self._init_components()

    def _init_components(self):
        """
        Initialize indexer components including chunker and embeddings.
        """

        # Initialize chunker
        self.Chunker = self.__get_chunker()

        # Initialize vector store
        self.vector_store = self.__get_vectorstore(self.config)

    def __get_data_processor(self, file_path: str) -> List[tuple[list[Document], str]]:
        """
        Get data processor based on file path, return processed data and file type.
        Args:
          file_path (str): File path

        Returns:
          List[tuple[list[Document], str]]: List of processed data and file type tuples
        """

        # Recursively process directories
        if os.path.isdir(file_path):
            results = []
            for filename in os.listdir(file_path):
                # Skip hidden files
                if filename.startswith("."):
                    continue
                full_path = os.path.join(file_path, filename)
                try:
                    # If it's a file, log that we're about to process it
                    if os.path.isfile(full_path):
                        self._logger.info("Processing file: %s", full_path)
                    results += self.__get_data_processor(full_path)
                except ValueError as e:
                    self._logger.warning("Skipping %s: %s", full_path, str(e))
                    continue
            return results
        # If it's a file, select processor based on extension
        else:
            ext = Path(file_path).suffix.lower()  # Get file extension
            # Select processor based on extension
            loader_mapping = LOADER_MAPPING.get(ext)
            if loader_mapping is None:
                return []
            processor, loader_args = loader_mapping
            # Return processed file + extension to indicate image vs text
            return [
                (processor(**loader_args).process(file_path), file_path.split(".")[-1])
            ]

    def __get_chunker(self) -> Chunker:
        """
        Get chunker instance.

        Returns:
            Chunker: Chunker instance
        """
        chunker_config = self.config.get("chunker", {})
        # Get chunker type and parameters
        chunker_type = chunker_config.get(
            "type", "recursive"
        )  # Default to recursive chunker
        params = chunker_config.get("params", {})
        mapping_val = CHUNKER_MAPPING.get(chunker_type)
        if mapping_val is None:
            raise ValueError(
                f"Indexer_get_chunker -> Unknown chunker type: {chunker_type}"
            )

        # Back-compat: mapping may be a class OR (class, default_kwargs)
        if isinstance(mapping_val, tuple) and len(mapping_val) == 2:
            chunker_cls, default_kwargs = mapping_val
            if not isinstance(default_kwargs, dict):
                default_kwargs = {}
            if not callable(chunker_cls):
                raise ValueError(
                    f"Indexer_get_chunker -> Invalid chunker mapping for {chunker_type}"
                )
            merged = {**default_kwargs, **params}
            return chunker_cls(**merged)

        if callable(mapping_val):
            return mapping_val(**params)

        raise ValueError(
            f"Indexer_get_chunker -> Invalid chunker mapping for {chunker_type}"
        )

    def __get_vectorstore(self, config: dict) -> VectorStore:
        """
        Get vector store instance.

        Returns:
            VectorStore: Vector store instance
        """
        # Use unified embeddings function (cloud-based or fallback)
        embeddings = get_embeddings(config)

        vectorstore_config = self.config.get("vectorstore", {})
        return VectorStore(
            embeddings=embeddings,
            persist_directory=vectorstore_config.get("persist_directory", None),
        )

    def index(self, file_path: str) -> None | tuple[VectorStore, BM25Bundle]:
        """
        Index files
        Args:
            file_path (str): File path

        Returns:
            None | tuple[VectorStore, BM25Okapi]: Returns vector store and BM25 index, or None if no chunks generated
        """
        datas = self.__get_data_processor(file_path)

        chunks: list[Document] = []
        for data, file_type in datas:
            if file_type in ["jpg", "jpeg", "png"]:
                pass  # Multimodal info: skip images for now, process text only
            else:
                chunks += self.Chunker.chunk(data)

        if not chunks:
            self._logger.warning("No chunks generated from input: %s", file_path)
            return None
        # Build vector store
        self.vector_store.add_documents(chunks)

        vectorstore_config = self.config.get("vectorstore", {})
        persist_directory = vectorstore_config.get("persist_directory")
        if persist_directory:
            self.vector_store.save_local(persist_directory)

        # Build and persist BM25 (based on all indexed documents for consistency)
        all_docs = self.vector_store.get_all_documents()
        bm25_bundle = create_bm25_bundle(all_docs)

        bm25_path = self.config.get("bm25_path")
        if bm25_path:
            save_bm25_bundle(bm25_path, bm25_bundle)

        return self.vector_store, bm25_bundle
