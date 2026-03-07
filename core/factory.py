"""Shared factory functions for building retriever and agent graph."""

from __future__ import annotations

import logging
from pathlib import Path

from indexing.indexer import Indexer
from indexing.retriever import FusionRetriever
from core.persistence import load_bm25_bundle
from core.settings import AppSettings

logger = logging.getLogger(__name__)


def build_retriever(settings: AppSettings) -> FusionRetriever | None:
    """Build a FusionRetriever from settings. Returns None if no index is available."""
    cfg = settings.indexer_config()
    indexer = Indexer(cfg)

    bm25_path = Path(cfg.get("bm25_path", str(settings.bm25_path)))
    if bm25_path.exists():
        bm25 = load_bm25_bundle(bm25_path)
    else:
        docs = indexer.vector_store.get_all_documents()
        if not docs:
            return None
        from indexing.bm25_index import create_bm25_bundle

        bm25 = create_bm25_bundle(docs)

    retr_cfg = cfg.get("retriever", {})
    k = int(retr_cfg.get("k", settings.retriever_k))
    alpha = float(retr_cfg.get("alpha", settings.fusion_alpha))
    return FusionRetriever(
        vectorstore=indexer.vector_store, bm25=bm25, alpha=alpha, k=k
    )


def build_graph(settings: AppSettings):
    """Build the agent graph. Raises RuntimeError if no index is loaded."""
    from agent.graph import create_agent_graph
    from agent.tools import ToolFactory
    from llm.llm import get_llm

    llm = get_llm(settings.llm_config())
    retriever = build_retriever(settings)
    if retriever is None:
        raise RuntimeError(
            "No index loaded. Run `python main.py index <path>` first or use the UI to index documents."
        )
    tools = ToolFactory(retriever).create_tools()
    return create_agent_graph(llm, tools)
