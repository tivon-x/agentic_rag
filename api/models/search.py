"""Search API schemas with stage-by-stage scores."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SearchScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vector: float | None
    bm25: float | None
    fusion: float
    boosts: dict[str, float]
    final: float
    rerank_rank: int


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passage_id: str
    paper_id: str
    paper_title: str | None
    authors: list[str]
    year: int | None
    section_id: str
    section_title: str
    page_start: int
    page_end: int
    quote_text: str
    block_type: str
    scores: SearchScores
    paper_url: str
    pdf_url: str


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    index_version: str
    total: int
    results: list[SearchResult]
    degraded_reason: str | None = None
