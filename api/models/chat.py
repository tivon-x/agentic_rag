from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


MAX_CHAT_CONTENT_CHARS = 20_000
MAX_CHAT_SESSION_MESSAGES = 200
MAX_CHAT_HISTORY_CHARS = 1_000_000


class ChatEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    paper_id: str | None = None
    paper_title: str | None = None
    source: str = Field(min_length=1)
    section_path: list[str] = Field(default_factory=list)
    page: int | None = Field(default=None, ge=1)
    quote: str = Field(min_length=1)
    score: float | None = None
    relevance: str | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system"] = "user"
    content: str = Field(min_length=1, max_length=MAX_CHAT_CONTENT_CHARS)
    evidence: list[ChatEvidence] | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=MAX_CHAT_CONTENT_CHARS)
    session_id: str | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    created_at: datetime
    message: ChatMessage


class ChatSessionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessage] = Field(default_factory=list)


class ChatSessionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    title: str
    message_count: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


class ChatSessionListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ChatSessionSummary] = Field(default_factory=list)
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)


class StreamToken(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["progress", "evidence", "answer.final", "error"]
    content: str | None = None
    session_id: str
    citations_markdown: str | None = None
    evidence: list[ChatEvidence] | None = None
    error: str | None = None
