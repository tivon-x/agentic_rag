from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


IndexingJobStatus = Literal[
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
]


class IndexingJobResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: str
    status: IndexingJobStatus
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
    attempt_count: int = 0
    max_attempts: int = 3
    progress: dict[str, Any] | None = None
    active_version_before: str | None = None
    target_version: str | None = None


class FileUploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    filename: str
    status: IndexingJobStatus
