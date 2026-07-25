from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class ChatSessionRecord:
    """Stored chat session metadata."""

    id: str
    created_at: datetime
    updated_at: datetime
    messages: str


@dataclass(slots=True)
class IndexingJobRecord:
    """Stored indexing job metadata."""

    id: str
    status: str
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
