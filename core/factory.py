"""Shared factory functions for building retriever and agent graph."""

from __future__ import annotations

import logging
from pathlib import Path

from core.corpus_profile import build_corpus_profile_context, load_corpus_profile
from core.persistence import load_bm25_bundle
from core.settings import AppSettings
from indexing.bm25_index import create_lexical_store
from indexing.indexer import Indexer
from indexing.retriever import FusionRetriever
from indexing.stores.lexical_store import LexicalStore
from indexing.stores.node_store import NodeStore, create_node_store

logger = logging.getLogger(__name__)



def build_retriever(settings: AppSettings) -> FusionRetriever | None:
    """Build a FusionRetriever from settings. Returns None if no index is available."""
    cfg = settings.indexer_config()
    indexer = Indexer(cfg)

    bm25_path = Path(cfg.get("bm25_path", str(settings.bm25_path)))
    lexical_backend = str(cfg.get("lexical_backend", settings.lexical_backend))
    if bm25_path.exists():
        lexical_store: LexicalStore = create_lexical_store(
            lexical_backend,
            bundle=load_bm25_bundle(bm25_path),
        )
    else:
        docs = indexer.vector_store.get_all_documents()
        if not docs:
            return None
        lexical_store = create_lexical_store(
            lexical_backend,
            documents=docs,
        )

    retr_cfg = cfg.get("retriever", {})
    k = int(retr_cfg.get("k", settings.retriever_k))
    alpha = float(retr_cfg.get("alpha", settings.fusion_alpha))
    node_store: NodeStore | None = None
    if settings.nodes_path.exists() and settings.doc_trees_path.exists():
        node_store = create_node_store(
            settings.node_backend,
            nodes_path=settings.nodes_path,
            doc_trees_path=settings.doc_trees_path,
        )
    corpus_profile = load_corpus_profile(settings.index_dir)
    return FusionRetriever(
        vectorstore=indexer.vector_store,
        lexical_store=lexical_store,
        alpha=alpha,
        k=k,
        reranker_backend=str(retr_cfg.get("reranker_backend", settings.reranker_backend)),
        flashrank_model=str(retr_cfg.get("flashrank_model", settings.flashrank_model)),
        flashrank_cache_dir=str(
            retr_cfg.get("flashrank_cache_dir", settings.flashrank_cache_dir)
        ),
        flashrank_top_n=int(retr_cfg.get("flashrank_top_n", settings.flashrank_top_n)),
        node_store=node_store,
        corpus_profile=corpus_profile,
    )



def build_graph(settings: AppSettings):
    """Build the agent graph. Raises RuntimeError if no index is loaded."""
    from agent.graph import create_agent_graph
    from agent.tools import ToolFactory
    from llms.llm import configure_llm_router

    configure_llm_router(settings.llm_config())

    retriever = build_retriever(settings)
    if retriever is None:
        raise RuntimeError(
            "No index loaded. Run `python main.py index <path>` first or use the UI to index documents."
        )

    corpus_profile = load_corpus_profile(settings.index_dir)
    corpus_profile_context = build_corpus_profile_context(corpus_profile)

    tool_factory = ToolFactory(retriever)
    tools = tool_factory.create_tools()
    return create_agent_graph(
        tools,
        corpus_profile=corpus_profile_context,
        corpus_profile_data=corpus_profile,
        tool_factory=tool_factory,
    )
