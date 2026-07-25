from __future__ import annotations

from typing import Any

from core.factory import build_graph
from core.settings import AppSettings


_CACHE: dict[str, object] = {"graph": None, "fingerprint": None}


def _compute_cache_fingerprint(settings: AppSettings) -> str:
    return (
        f"{settings.faiss_dir}|{settings.bm25_path}|{settings.llm_model}|"
        f"{settings.llm_api_base}|{settings.embedding_model}|"
        f"{settings.embedding_api_base}"
    )


def invalidate_graph_cache() -> None:
    _CACHE["graph"] = None
    _CACHE["fingerprint"] = None


def get_cached_graph(settings: AppSettings) -> Any | None:
    fingerprint = _compute_cache_fingerprint(settings)
    cached = _CACHE.get("graph")
    if cached is not None and _CACHE.get("fingerprint") == fingerprint:
        return cached
    if settings.offline_mode:
        return None

    graph = build_graph(settings)
    _CACHE["graph"] = graph
    _CACHE["fingerprint"] = fingerprint
    return graph
