"""Library search exposing paper/page evidence and scoring stages."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_settings
from api.models.search import SearchResponse, SearchResult, SearchScores
from core.factory import build_retriever
from core.persistence import load_bm25_bundle
from core.settings import AppSettings
from indexing.index_versions import (
    get_active_version_id,
    resolve_indexer_config,
)
from indexing.retrieval_pipeline import RetrievalCandidate


router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse)
async def search_library(
    q: str = Query(min_length=1, max_length=4000),
    paper_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=50),
    settings: AppSettings = Depends(get_settings),
) -> SearchResponse:
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Search query cannot be blank.")
    version_id = get_active_version_id(settings)
    if version_id is None:
        raise HTTPException(status_code=409, detail="No active index is available.")
    degraded_reason: str | None = None
    try:
        retriever = build_retriever(settings)
    except ValueError as exc:
        if "API key" not in str(exc):
            raise
        candidates = _bm25_candidates(settings, query, limit=limit * 3)
        degraded_reason = "dense_embedding_unavailable_bm25_only"
    else:
        if retriever is None:
            raise HTTPException(status_code=409, detail="No active index is available.")
        candidates, _ = retriever.search_scored(query, limit=limit * 3)

    results: list[SearchResult] = []
    for rank, candidate in enumerate(candidates, start=1):
        metadata = candidate.document.metadata
        candidate_paper_id = str(metadata.get("paper_id") or "")
        if not candidate_paper_id or (
            paper_id and candidate_paper_id != paper_id
        ):
            continue
        page_start = int(
            metadata.get("page_start") or metadata.get("page") or 1
        )
        page_end = int(metadata.get("page_end") or page_start)
        results.append(
            SearchResult(
                passage_id=str(
                    metadata.get("passage_id")
                    or metadata.get("node_id")
                    or ""
                ),
                paper_id=candidate_paper_id,
                paper_title=(
                    str(metadata["paper_title"])
                    if metadata.get("paper_title")
                    else None
                ),
                authors=[
                    str(author) for author in metadata.get("authors", []) or []
                ],
                year=(
                    int(metadata["year"]) if metadata.get("year") else None
                ),
                section_id=str(metadata.get("section_id") or ""),
                section_title=str(
                    metadata.get("section_title") or f"Page {page_start}"
                ),
                page_start=page_start,
                page_end=page_end,
                quote_text=str(
                    metadata.get("quote_text")
                    or candidate.document.page_content
                ),
                block_type=str(metadata.get("block_type") or "paragraph"),
                scores=SearchScores(
                    vector=candidate.source_scores.get("vector"),
                    bm25=candidate.source_scores.get("bm25"),
                    fusion=candidate.score,
                    boosts=dict(candidate.boosts),
                    final=candidate.final_score,
                    rerank_rank=rank,
                ),
                paper_url=(
                    f"/papers/{candidate_paper_id}?page={page_start}"
                ),
                pdf_url=(
                    f"/api/papers/{candidate_paper_id}/file#page={page_start}"
                ),
            )
        )
        if len(results) >= limit:
            break
    return SearchResponse(
        query=query,
        index_version=version_id,
        total=len(results),
        results=results,
        degraded_reason=degraded_reason,
    )


def _bm25_candidates(
    settings: AppSettings,
    query: str,
    *,
    limit: int,
) -> list[RetrievalCandidate]:
    config, _ = resolve_indexer_config(settings)
    path = Path(config["bm25_path"])
    if not path.exists():
        return []
    bundle = load_bm25_bundle(path)
    rows = bundle.topk_with_scores(query, k=limit)
    scores = [float(score) for _, score in rows]
    minimum = min(scores) if scores else 0.0
    maximum = max(scores) if scores else 0.0
    return [
        RetrievalCandidate(
            document=document,
            score=(
                0.0
                if maximum == minimum
                else (float(score) - minimum) / (maximum - minimum)
            ),
            source_scores={"bm25": float(score)},
        )
        for document, score in rows
    ]
