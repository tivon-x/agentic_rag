"""Paper catalog, metadata correction, and byte-range file delivery."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
)
from fastapi.responses import StreamingResponse

from api.db.database import IndexingQueueFullError
from api.db.papers import (
    PaperVersionConflictError,
    get_paper,
    list_papers,
    update_paper_metadata,
)
from api.dependencies import get_settings
from api.models.papers import (
    PaperDetail,
    PaperListResponse,
    PaperMetadataPatch,
)
from api.services.index_worker import IndexWorker
from core.settings import AppSettings


_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
router = APIRouter(tags=["papers"])


@router.get("/papers", response_model=PaperListResponse)
async def read_papers(
    q: str | None = Query(default=None, max_length=300),
    parse_status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    settings: AppSettings = Depends(get_settings),
) -> PaperListResponse:
    payload = await list_papers(
        settings,
        query=q,
        parse_status=parse_status,
        limit=limit,
        offset=offset,
    )
    return PaperListResponse.model_validate(payload)


@router.get("/papers/{paper_id}", response_model=PaperDetail)
async def read_paper(
    paper_id: str,
    settings: AppSettings = Depends(get_settings),
) -> PaperDetail:
    payload = await get_paper(settings, paper_id=paper_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return PaperDetail.model_validate(payload)


@router.patch("/papers/{paper_id}", response_model=PaperDetail)
async def patch_paper(
    paper_id: str,
    payload: PaperMetadataPatch,
    request: Request,
    if_match: str = Header(alias="If-Match"),
    settings: AppSettings = Depends(get_settings),
) -> PaperDetail:
    expected_version = _parse_if_match(if_match)
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No metadata fields supplied.")
    if updates.get("authors") is not None:
        updates["authors"] = [
            author.strip()
            for author in updates["authors"]
            if author.strip()
        ]
    try:
        paper, reindex_job_id = await update_paper_metadata(
            settings,
            paper_id=paper_id,
            expected_version=expected_version,
            updates=updates,
        )
    except PaperVersionConflictError as exc:
        raise HTTPException(status_code=412, detail=str(exc)) from exc
    except IndexingQueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    worker = getattr(request.app.state, "index_worker", None)
    if isinstance(worker, IndexWorker):
        worker.notify()
    paper["reindex_job_id"] = reindex_job_id
    return PaperDetail.model_validate(paper)


@router.get("/papers/{paper_id}/file")
async def read_paper_file(
    paper_id: str,
    range_header: str | None = Header(default=None, alias="Range"),
    settings: AppSettings = Depends(get_settings),
) -> StreamingResponse:
    payload = await get_paper(settings, paper_id=paper_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    path = _validated_source_path(settings, paper_id)
    size = path.stat().st_size
    start, end, status_code = _resolve_range(range_header, size)
    length = end - start + 1
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Disposition": (
            "inline; filename*=UTF-8''" + quote(str(payload["file_name"]))
        ),
    }
    if status_code == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    media_type = (
        "application/pdf"
        if path.suffix.lower() == ".pdf"
        else "text/plain; charset=utf-8"
    )
    return StreamingResponse(
        _file_range(path, start=start, length=length),
        status_code=status_code,
        media_type=media_type,
        headers=headers,
    )


def _parse_if_match(value: str) -> int:
    normalized = value.strip().removeprefix("W/").strip('"')
    try:
        version = int(normalized)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="If-Match must contain the paper metadata version.",
        ) from exc
    if version <= 0:
        raise HTTPException(status_code=400, detail="Invalid If-Match version.")
    return version


def _validated_source_path(settings: AppSettings, paper_id: str) -> Path:
    import sqlite3

    if settings.app_db_path is None:
        raise HTTPException(status_code=404, detail="Paper file not found.")
    with sqlite3.connect(settings.app_db_path) as db:
        row = db.execute(
            "SELECT source_path FROM papers WHERE id = ?",
            (paper_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Paper file not found.")
    upload_root = (
        settings.upload_root or settings.data_dir / "uploads"
    ).resolve()
    path = Path(str(row[0])).resolve()
    if not path.is_relative_to(upload_root) or not path.is_file():
        raise HTTPException(status_code=404, detail="Paper file is unavailable.")
    return path


def _resolve_range(
    header: str | None,
    size: int,
) -> tuple[int, int, int]:
    if size <= 0:
        raise HTTPException(status_code=416, detail="Paper file is empty.")
    if not header:
        return 0, size - 1, 200
    if "," in header:
        raise HTTPException(
            status_code=416,
            detail="Multiple byte ranges are not supported.",
            headers={"Content-Range": f"bytes */{size}"},
        )
    match = _RANGE_RE.fullmatch(header.strip())
    if not match or not (match.group(1) or match.group(2)):
        raise HTTPException(
            status_code=416,
            detail="Invalid Range header.",
            headers={"Content-Range": f"bytes */{size}"},
        )
    if match.group(1):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else size - 1
    else:
        suffix_length = int(match.group(2))
        if suffix_length <= 0:
            raise HTTPException(status_code=416, detail="Invalid suffix range.")
        start = max(0, size - suffix_length)
        end = size - 1
    if start >= size or end < start:
        raise HTTPException(
            status_code=416,
            detail="Requested range is outside the paper file.",
            headers={"Content-Range": f"bytes */{size}"},
        )
    return start, min(end, size - 1), 206


def _file_range(path: Path, *, start: int, length: int):
    remaining = length
    with path.open("rb") as source:
        source.seek(start)
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
