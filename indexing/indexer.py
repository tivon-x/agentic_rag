#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_core.documents import Document

from core.persistence import save_bm25_bundle
from indexing.bm25_index import BM25Bundle, create_lexical_store
from indexing.builders.hierarchical_index_builder import HierarchicalIndexBuilder
from indexing.chunker import Chunker
from indexing.embeddings import get_embeddings
from indexing.mappers import CHUNKER_MAPPING, LOADER_MAPPING
from indexing.parsers.base import SUPPORTED_HIERARCHICAL_SUFFIXES, build_parser
from indexing.stores.lexical_store import LexicalStore
from indexing.stores.node_store import create_node_store
from indexing.stores.vector_store import VectorStore as VectorStoreProtocol
from indexing.vectorstore import create_vector_store


class Indexer:
    def __init__(self, config: dict):
        self.config = config
        self._logger = logging.getLogger(__name__)
        self._init_components()

    def _init_components(self) -> None:
        self.chunker = self._get_chunker()
        self.embeddings = get_embeddings(self.config)
        self.vector_store = self._get_vectorstore()

        self.index_mode = str(self.config.get("index_mode", "flat")).strip().lower()
        self.leaf_node_type = str(
            self.config.get("leaf_node_type", "paragraph")
        ).strip()
        self.parent_embed_pooling = str(
            self.config.get("parent_embed_pooling", "mean")
        ).strip()

        nodes_path = self.config.get("nodes_path")
        doc_trees_path = self.config.get("doc_trees_path")
        node_backend = str(self.config.get("node_backend", "json"))
        self.node_store = (
            create_node_store(
                node_backend,
                nodes_path=nodes_path,
                doc_trees_path=doc_trees_path,
            )
            if nodes_path and doc_trees_path
            else None
        )

    def _iter_supported_files(self, file_path: str) -> list[str]:
        if os.path.isdir(file_path):
            results: list[str] = []
            for entry in sorted(os.listdir(file_path)):
                if entry.startswith("."):
                    continue
                results.extend(self._iter_supported_files(os.path.join(file_path, entry)))
            return results

        suffix = Path(file_path).suffix.lower()
        if suffix not in SUPPORTED_HIERARCHICAL_SUFFIXES:
            return []
        return [file_path]

    def _get_flat_documents(self, file_path: str) -> list[tuple[list[Document], str]]:
        if os.path.isdir(file_path):
            results: list[tuple[list[Document], str]] = []
            for filename in os.listdir(file_path):
                if filename.startswith("."):
                    continue
                full_path = os.path.join(file_path, filename)
                try:
                    if os.path.isfile(full_path):
                        self._logger.info("Processing file: %s", full_path)
                    results.extend(self._get_flat_documents(full_path))
                except ValueError as exc:
                    self._logger.warning("Skipping %s: %s", full_path, str(exc))
            return results

        ext = Path(file_path).suffix.lower()
        loader_mapping = LOADER_MAPPING.get(ext)
        if loader_mapping is None:
            return []
        processor, loader_args = loader_mapping
        return [(processor(**loader_args).process(file_path), file_path.split(".")[-1])]

    def _get_chunker(self) -> Chunker:
        chunker_config = self.config.get("chunker", {})
        chunker_type = chunker_config.get("type", "recursive")
        params = chunker_config.get("params", {})
        mapping_val = CHUNKER_MAPPING.get(chunker_type)
        if mapping_val is None:
            raise ValueError(f"Indexer_get_chunker -> Unknown chunker type: {chunker_type}")

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

    def _get_vectorstore(self) -> VectorStoreProtocol:
        vectorstore_config = self.config.get("vectorstore", {})
        vector_backend = str(self.config.get("vector_backend", "faiss"))
        return create_vector_store(
            vector_backend,
            embeddings=self.embeddings,
            persist_directory=vectorstore_config.get("persist_directory"),
        )

    def _index_flat(self, file_path: str) -> list[Document]:
        datas = self._get_flat_documents(file_path)

        chunks: list[Document] = []
        for data, file_type in datas:
            if file_type in ["jpg", "jpeg", "png"]:
                continue
            chunks.extend(self.chunker.chunk(data))
        return chunks

    def _index_hierarchical(self, file_path: str) -> list[Document]:
        files = self._iter_supported_files(file_path)
        if not files:
            return []

        trees = []
        for supported_file in files:
            try:
                parser = build_parser(supported_file)
                trees.append(parser.parse(supported_file))
            except Exception as exc:
                self._logger.warning(
                    "Hierarchical parse failed for %s, skipping file: %s",
                    supported_file,
                    str(exc),
                )

        if not trees:
            return []

        builder = HierarchicalIndexBuilder(
            embeddings=self.embeddings,
            leaf_node_type=self.leaf_node_type,
            parent_embed_pooling=self.parent_embed_pooling,
        )
        enriched_trees = builder.enrich_trees(trees)
        if self.node_store is not None:
            self.node_store.save_trees(enriched_trees)
        return builder.to_documents(enriched_trees)

    def index(self, file_path: str) -> None | tuple[VectorStoreProtocol, LexicalStore]:
        if self.index_mode == "hierarchical":
            chunks = self._index_hierarchical(file_path)
        else:
            chunks = self._index_flat(file_path)

        if not chunks:
            self._logger.warning("No chunks generated from input: %s", file_path)
            return None

        self.vector_store.add_documents(chunks)

        vectorstore_config = self.config.get("vectorstore", {})
        persist_directory = vectorstore_config.get("persist_directory")
        if persist_directory:
            self.vector_store.save(persist_directory)

        all_docs = self.vector_store.get_all_documents()
        lexical_backend = str(self.config.get("lexical_backend", "bm25"))
        lexical_store = create_lexical_store(
            lexical_backend,
            documents=all_docs,
        )

        bm25_path = self.config.get("bm25_path")
        if bm25_path:
            if not isinstance(lexical_store, BM25Bundle):
                raise TypeError("Only BM25Bundle persistence is currently supported.")
            save_bm25_bundle(bm25_path, lexical_store)

        return self.vector_store, lexical_store
