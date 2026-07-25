from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.db.database import create_indexing_job, get_indexing_job, update_indexing_job
from api.dependencies import get_settings
from api.models.indexing import FileUploadResponse, IndexingJobResponse
from api.services.graph_cache import invalidate_graph_cache
from core.settings import AppSettings
from indexing.indexer import Indexer


SUPPORTED_SOURCE_TYPES = {".pdf", ".md", ".txt"}
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()

router = APIRouter(tags=["indexing"])


@router.get("/indexing-jobs/{job_id}", response_model=IndexingJobResponse)
async def read_indexing_job(
    job_id: str,
    settings: AppSettings = Depends(get_settings),
) -> IndexingJobResponse:
    record = await get_indexing_job(settings, job_id=job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Indexing job not found.")
    return IndexingJobResponse.model_validate(record)


@router.post("/index/files", response_model=list[FileUploadResponse])
async def upload_and_index_files(
    files: list[UploadFile] = File(...),
    index_mode: str = Form(default="flat"),
    settings: AppSettings = Depends(get_settings),
) -> list[FileUploadResponse]:
    if index_mode not in {"flat", "hierarchical"}:
        raise HTTPException(status_code=400, detail="Unsupported index mode.")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    uploads_root = settings.data_dir / "api" / "uploads"
    job_responses: list[FileUploadResponse] = []

    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in SUPPORTED_SOURCE_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

        job_id = str(uuid4())
        job_dir = uploads_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        target_path = job_dir / (upload.filename or f"{job_id}{suffix}")
        contents = await upload.read()
        target_path.write_bytes(contents)

        now = datetime.now(UTC).isoformat()
        await create_indexing_job(
            settings,
            job_id=job_id,
            status="pending",
            created_at=now,
        )
        task = asyncio.create_task(
            _run_indexing_job(
                settings=settings,
                job_id=job_id,
                file_path=target_path,
                index_mode=index_mode,
            )
        )
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)
        job_responses.append(
            FileUploadResponse(job_id=job_id, filename=target_path.name, status="pending")
        )

    return job_responses


async def _run_indexing_job(
    *,
    settings: AppSettings,
    job_id: str,
    file_path: Path,
    index_mode: str,
) -> None:
    await update_indexing_job(
        settings,
        job_id=job_id,
        status="running",
        updated_at=datetime.now(UTC).isoformat(),
    )
    try:
        config = settings.indexer_config()
        config["index_mode"] = index_mode
        indexer = Indexer(config)
        await asyncio.to_thread(indexer.index, str(file_path))
        invalidate_graph_cache()
        await update_indexing_job(
            settings,
            job_id=job_id,
            status="completed",
            updated_at=datetime.now(UTC).isoformat(),
        )
    except Exception as exc:
        await update_indexing_job(
            settings,
            job_id=job_id,
            status="failed",
            updated_at=datetime.now(UTC).isoformat(),
            error_message=str(exc),
        )
