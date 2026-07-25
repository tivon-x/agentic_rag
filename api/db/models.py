from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


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
    request: dict[str, Any] | None = None
    attempt_count: int = 0
    max_attempts: int = 3
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    heartbeat_at: datetime | None = None
    progress: dict[str, Any] | None = None
    active_version_before: str | None = None
    target_version: str | None = None
