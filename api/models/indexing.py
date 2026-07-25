from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


IndexingJobStatus = Literal["pending", "running", "completed", "failed"]


class IndexingJobResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: str
    status: IndexingJobStatus
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None


class FileUploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    filename: str
    status: IndexingJobStatus
