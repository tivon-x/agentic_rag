from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)

from api.db.database import (
    IdempotencyConflictError,
    create_indexing_job_idempotent,
    get_indexing_job,
    retry_failed_indexing_job,
)
from api.dependencies import get_settings
from api.models.indexing import FileUploadResponse, IndexingJobResponse
from api.services.index_worker import IndexWorker
from core.settings import AppSettings
from indexing.index_versions import get_active_version_id


SUPPORTED_SOURCE_TYPES = {".pdf", ".md", ".txt"}
UPLOAD_CHUNK_BYTES = 1024 * 1024
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}

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


@router.post(
    "/indexing-jobs/{job_id}/retry",
    response_model=IndexingJobResponse,
)
async def retry_indexing_job(
    job_id: str,
    request: Request,
    settings: AppSettings = Depends(get_settings),
) -> IndexingJobResponse:
    record = await retry_failed_indexing_job(settings, job_id=job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Indexing job not found.")
    _notify_worker(request)
    return IndexingJobResponse.model_validate(record)


@router.post("/index/files", response_model=list[FileUploadResponse])
async def upload_and_index_files(
    request: Request,
    files: list[UploadFile] = File(...),
    index_mode: str = Form(default="flat"),
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=1, max_length=200),
    settings: AppSettings = Depends(get_settings),
) -> list[FileUploadResponse]:
    idempotency_key = idempotency_key.strip()
    if not idempotency_key:
        raise HTTPException(status_code=400, detail="Idempotency-Key cannot be blank.")
    if index_mode not in {"flat", "hierarchical"}:
        raise HTTPException(status_code=400, detail="Unsupported index mode.")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if settings.index_write_mode == "legacy":
        raise HTTPException(
            status_code=409,
            detail=(
                "API indexing is disabled while INDEX_WRITE_MODE=legacy; "
                "legacy mode is read-only."
            ),
        )

    upload_root = (
        settings.upload_root or settings.data_dir / "uploads"
    ).resolve()
    staging_root = (upload_root / ".staging").resolve()
    _ensure_child_path(upload_root, staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = staging_root / uuid4().hex
    staging_dir.mkdir()
    final_job_dir: Path | None = None

    try:
        uploads = await _save_validated_uploads(
            files,
            staging_dir=staging_dir,
            max_bytes=settings.upload_max_bytes,
        )
        request_hash = _upload_request_hash(index_mode, uploads)
        job_id = uuid4().hex
        final_job_dir = (upload_root / "jobs" / job_id).resolve()
        _ensure_child_path(upload_root, final_job_dir)
        final_job_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_dir, final_job_dir)

        items = [
            {
                "filename": upload["filename"],
                "source_path": str(
                    _validated_final_path(
                        upload_root,
                        final_job_dir / upload["filename"],
                    )
                ),
            }
            for upload in uploads
        ]
        response = [
            {
                "job_id": job_id,
                "filename": upload["filename"],
                "status": "queued",
            }
            for upload in uploads
        ]
        created, stored_response = await create_indexing_job_idempotent(
            settings,
            job_id=job_id,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            request={
                "index_mode": index_mode,
                "files": [
                    {
                        "filename": upload["filename"],
                        "sha256": upload["sha256"],
                        "size": upload["size"],
                    }
                    for upload in uploads
                ],
            },
            items=items,
            response=response,
            created_at=datetime.now(UTC).isoformat(),
            active_version_before=get_active_version_id(settings),
        )
        if not created:
            _remove_upload_dir(upload_root, final_job_dir)
        else:
            _notify_worker(request)
        return [
            FileUploadResponse.model_validate(item)
            for item in stored_response
        ]
    except IdempotencyConflictError as exc:
        if final_job_dir is not None:
            _remove_upload_dir(upload_root, final_job_dir)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except HTTPException:
        if final_job_dir is not None:
            _remove_upload_dir(upload_root, final_job_dir)
        raise
    except Exception:
        if final_job_dir is not None:
            _remove_upload_dir(upload_root, final_job_dir)
        raise
    finally:
        if staging_dir.exists():
            _remove_upload_dir(upload_root, staging_dir)
        for upload in files:
            await upload.close()


async def _save_validated_uploads(
    files: list[UploadFile],
    *,
    staging_dir: Path,
    max_bytes: int,
) -> list[dict[str, str | int]]:
    uploads: list[dict[str, str | int]] = []
    seen_names: set[str] = set()
    for upload in files:
        filename = _validated_filename(upload.filename)
        if filename in seen_names:
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate filename in upload: {filename}",
            )
        seen_names.add(filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SOURCE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {suffix}",
            )

        target_path = (staging_dir / filename).resolve()
        _ensure_child_path(staging_dir.resolve(), target_path)
        digest = hashlib.sha256()
        size = 0
        with target_path.open("xb") as target:
            while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds UPLOAD_MAX_BYTES: {filename}",
                    )
                digest.update(chunk)
                target.write(chunk)
        uploads.append(
            {
                "filename": filename,
                "sha256": digest.hexdigest(),
                "size": size,
            }
        )
    return uploads


def _validated_filename(filename: str | None) -> str:
    raw = (filename or "").strip()
    if (
        not raw
        or raw in {".", ".."}
        or "\x00" in raw
        or "/" in raw
        or "\\" in raw
        or Path(raw).is_absolute()
        or Path(raw).name != raw
        or raw.endswith((" ", "."))
        or any(character in '<>:"|?*' or ord(character) < 32 for character in raw)
        or raw.split(".", 1)[0].rstrip(" .").upper() in WINDOWS_RESERVED_NAMES
    ):
        raise HTTPException(status_code=400, detail="Invalid upload filename.")
    return raw


def _upload_request_hash(
    index_mode: str,
    uploads: list[dict[str, str | int]],
) -> str:
    payload = {
        "index_mode": index_mode,
        "files": sorted(
            uploads,
            key=lambda item: (
                str(item["filename"]),
                str(item["sha256"]),
                int(item["size"]),
            ),
        ),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validated_final_path(
    upload_root: Path,
    target_path: Path,
) -> Path:
    resolved = target_path.resolve()
    _ensure_child_path(upload_root, resolved)
    if not resolved.is_file():
        raise HTTPException(status_code=500, detail="Uploaded file is missing.")
    return resolved


def _ensure_child_path(root: Path, target: Path) -> None:
    if target == root or not target.is_relative_to(root):
        raise HTTPException(status_code=400, detail="Upload path escaped UPLOAD_ROOT.")


def _remove_upload_dir(upload_root: Path, target: Path) -> None:
    resolved_root = upload_root.resolve()
    resolved_target = target.resolve()
    _ensure_child_path(resolved_root, resolved_target)
    if resolved_target.exists():
        shutil.rmtree(resolved_target)


def _notify_worker(request: Request) -> None:
    worker = getattr(request.app.state, "index_worker", None)
    if isinstance(worker, IndexWorker):
        worker.notify()
