"""Paper catalog API schemas."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ParseStatus = Literal[
    "queued",
    "parsing",
    "parsed",
    "degraded",
    "needs_ocr",
    "failed",
]


class PaperSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    content_hash: str
    file_name: str
    source_type: str
    size_bytes: int
    title: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    doi: str | None
    arxiv_id: str | None
    metadata: dict[str, Any]
    metadata_status: Literal["needs_review", "verified"]
    metadata_version: int
    parse_status: ParseStatus
    parse_error: str | None
    fallback_reason: str | None
    latest_version_id: str | None
    created_at: str
    updated_at: str
    file_url: str


class PaperListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PaperSummary]
    total: int
    limit: int
    offset: int


class PaperSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    parent_id: str | None
    title: str
    level: int
    ordinal: int
    page_start: int
    page_end: int
    heading_path: list[str]


class PaperVersionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    parser_name: str
    parser_version: str
    normalization_version: str
    status: Literal["parsed", "degraded", "needs_ocr", "failed"]
    fallback_reason: str | None
    quality: dict[str, Any]
    page_count: int
    duration_ms: int
    created_at: str


class PaperDetail(PaperSummary):
    paper_version: PaperVersionResponse | None
    sections: list[PaperSection]
    reindex_job_id: str | None = None


class PaperMetadataPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=300)
    authors: list[Annotated[str, Field(max_length=200)]] | None = Field(
        default=None,
        max_length=30,
    )
    year: int | None = Field(default=None, ge=1000, le=3000)
    venue: str | None = Field(default=None, max_length=200)
    doi: str | None = Field(default=None, max_length=200)
    arxiv_id: str | None = Field(default=None, max_length=100)
