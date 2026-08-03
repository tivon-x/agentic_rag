from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from api.db.migrations import migrate_database
from api.db.models import ChatSessionRecord, IndexingJobRecord
from core.settings import AppSettings


_INITIALIZED_DB_PATHS: set[Path] = set()


class IdempotencyConflictError(ValueError):
    """Raised when an idempotency key is reused for a different request."""


def _db_path(settings: AppSettings) -> Path:
    return settings.app_db_path or settings.data_dir / "api" / "sessions.db"


async def init_db(settings: AppSettings) -> Path:
    path = _db_path(settings)
    resolved = path.resolve()
    if resolved not in _INITIALIZED_DB_PATHS:
        await migrate_database(path)
        _INITIALIZED_DB_PATHS.add(resolved)
    return path


@asynccontextmanager
async def get_db(settings: AppSettings):
    path = await init_db(settings)
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        yield db


async def create_chat_session(
    settings: AppSettings,
    *,
    session_id: str,
    messages: list[dict[str, Any]] | None = None,
    created_at: str,
) -> ChatSessionRecord:
    payload = json.dumps(messages or [], ensure_ascii=False)
    async with get_db(settings) as db:
        await db.execute(
            """
            INSERT INTO chat_sessions (id, created_at, updated_at, messages)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, created_at, created_at, payload),
        )
        await db.commit()
    return ChatSessionRecord(
        id=session_id,
        created_at=_parse_timestamp(created_at),
        updated_at=_parse_timestamp(created_at),
        messages=payload,
    )


async def get_chat_session_messages(
    settings: AppSettings,
    *,
    session_id: str,
) -> list[dict[str, Any]] | None:
    async with get_db(settings) as db:
        cursor = await db.execute(
            "SELECT messages FROM chat_sessions WHERE id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
    if row is None:
        return None
    data = _json_dict_or_list(row["messages"], default=[])
    return data if isinstance(data, list) else []


async def get_chat_session(
    settings: AppSettings,
    *,
    session_id: str,
) -> dict[str, Any] | None:
    async with get_db(settings) as db:
        cursor = await db.execute(
            """
            SELECT id, created_at, updated_at, messages
            FROM chat_sessions
            WHERE id = ?
            """,
            (session_id,),
        )
        row = await cursor.fetchone()
    if row is None:
        return None

    messages = _json_dict_or_list(row["messages"], default=[])
    return {
        "session_id": str(row["id"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "messages": messages if isinstance(messages, list) else [],
    }


async def list_chat_sessions(
    settings: AppSettings,
    *,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], int]:
    """Return chat sessions ordered by most recently updated first."""
    async with get_db(settings) as db:
        count_cursor = await db.execute("SELECT COUNT(*) AS total FROM chat_sessions")
        count_row = await count_cursor.fetchone()
        cursor = await db.execute(
            """
            SELECT id, created_at, updated_at, messages
            FROM chat_sessions
            ORDER BY updated_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        rows = await cursor.fetchall()

    sessions: list[dict[str, Any]] = []
    for row in rows:
        messages = _json_dict_or_list(row["messages"], default=[])
        sessions.append(
            {
                "session_id": str(row["id"]),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
                "messages": messages if isinstance(messages, list) else [],
                "message_count": _message_count(messages),
            }
        )
    return sessions, int(count_row["total"] if count_row is not None else 0)


async def upsert_chat_session_messages(
    settings: AppSettings,
    *,
    session_id: str,
    messages: list[dict[str, Any]],
    updated_at: str,
) -> None:
    payload = json.dumps(messages, ensure_ascii=False)
    async with get_db(settings) as db:
        await db.execute(
            """
            INSERT INTO chat_sessions (id, created_at, updated_at, messages)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                updated_at = excluded.updated_at,
                messages = excluded.messages
            """,
            (session_id, updated_at, updated_at, payload),
        )
        await db.commit()


async def create_indexing_job(
    settings: AppSettings,
    *,
    job_id: str,
    status: str,
    created_at: str,
    error_message: str | None = None,
    request: dict[str, Any] | None = None,
    items: list[dict[str, Any]] | None = None,
) -> IndexingJobRecord:
    request_json = json.dumps(request or {}, ensure_ascii=False)
    async with get_db(settings) as db:
        await db.execute(
            """
            INSERT INTO indexing_jobs (
                id, status, created_at, updated_at, error_message,
                request_json, max_attempts
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                status,
                created_at,
                created_at,
                error_message,
                request_json,
                settings.index_worker_max_attempts,
            ),
        )
        for item in items or []:
            await _upsert_uploaded_paper(db, item=item, created_at=created_at)
            await db.execute(
                """
                INSERT INTO index_job_items (
                    job_id, filename, source_path, status, paper_id
                )
                VALUES (?, ?, ?, 'queued', ?)
                """,
                (
                    job_id,
                    item["filename"],
                    item["source_path"],
                    item.get("paper_id"),
                ),
            )
        await db.commit()
    record = await get_indexing_job(settings, job_id=job_id)
    if record is None:
        raise RuntimeError(f"Failed to create indexing job {job_id}.")
    return record


async def create_indexing_job_idempotent(
    settings: AppSettings,
    *,
    job_id: str,
    idempotency_key: str,
    request_hash: str,
    request: dict[str, Any],
    items: list[dict[str, Any]],
    response: list[dict[str, Any]],
    created_at: str,
    active_version_before: str | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Create one job or return the response stored for an identical request."""
    async with get_db(settings) as db:
        await db.execute("BEGIN IMMEDIATE")
        cursor = await db.execute(
            """
            SELECT request_hash, response_json
            FROM idempotency_records
            WHERE scope = 'index-files' AND key = ?
            """,
            (idempotency_key,),
        )
        existing = await cursor.fetchone()
        if existing is not None:
            if str(existing["request_hash"]) != request_hash:
                await db.rollback()
                raise IdempotencyConflictError(
                    "Idempotency-Key was already used for a different upload."
                )
            stored = _json_dict_or_list(existing["response_json"], default=[])
            await db.rollback()
            return False, stored if isinstance(stored, list) else []

        await db.execute(
            """
            INSERT INTO indexing_jobs (
                id, status, created_at, updated_at, request_json, max_attempts,
                active_version_before
            )
            VALUES (?, 'queued', ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                created_at,
                created_at,
                json.dumps(request, ensure_ascii=False),
                settings.index_worker_max_attempts,
                active_version_before,
            ),
        )
        for item in items:
            await _upsert_uploaded_paper(db, item=item, created_at=created_at)
            await db.execute(
                """
                INSERT INTO index_job_items (
                    job_id, filename, source_path, status, paper_id
                )
                VALUES (?, ?, ?, 'queued', ?)
                """,
                (
                    job_id,
                    item["filename"],
                    item["source_path"],
                    item.get("paper_id"),
                ),
            )
        await db.execute(
            """
            INSERT INTO idempotency_records (
                scope, key, request_hash, response_json, created_at
            )
            VALUES ('index-files', ?, ?, ?, ?)
            """,
            (
                idempotency_key,
                request_hash,
                json.dumps(response, ensure_ascii=False),
                created_at,
            ),
        )
        await db.commit()
    return True, response


async def get_indexing_job(
    settings: AppSettings,
    *,
    job_id: str,
) -> IndexingJobRecord | None:
    async with get_db(settings) as db:
        cursor = await db.execute(
            "SELECT * FROM indexing_jobs WHERE id = ?",
            (job_id,),
        )
        row = await cursor.fetchone()
    return _indexing_job_from_row(row) if row is not None else None


async def list_index_job_items(
    settings: AppSettings,
    *,
    job_id: str,
) -> list[dict[str, Any]]:
    async with get_db(settings) as db:
        cursor = await db.execute(
            """
            SELECT id, filename, source_path, status, error_code, error_detail,
                   paper_id
            FROM index_job_items
            WHERE job_id = ?
            ORDER BY id
            """,
            (job_id,),
        )
        rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def recover_expired_indexing_jobs(
    settings: AppSettings,
    *,
    now: datetime | None = None,
) -> tuple[int, int]:
    """Requeue expired leases up to the retry limit and fail exhausted jobs."""
    current = (now or datetime.now(UTC)).isoformat()
    async with get_db(settings) as db:
        await db.execute("BEGIN IMMEDIATE")
        requeued = await db.execute(
            """
            UPDATE indexing_jobs
            SET
                status = 'queued',
                updated_at = ?,
                lease_owner = NULL,
                lease_expires_at = NULL,
                heartbeat_at = NULL,
                error_message = 'Index worker lease expired; retry queued.'
            WHERE
                status = 'running'
                AND lease_expires_at IS NOT NULL
                AND lease_expires_at <= ?
                AND attempt_count < max_attempts
            """,
            (current, current),
        )
        failed = await db.execute(
            """
            UPDATE indexing_jobs
            SET
                status = 'failed',
                updated_at = ?,
                lease_owner = NULL,
                lease_expires_at = NULL,
                heartbeat_at = NULL,
                error_message = 'Index worker lease expired; retry limit reached.'
            WHERE
                status = 'running'
                AND lease_expires_at IS NOT NULL
                AND lease_expires_at <= ?
                AND attempt_count >= max_attempts
            """,
            (current, current),
        )
        await db.execute(
            """
            UPDATE index_job_items
            SET status = 'queued'
            WHERE job_id IN (
                SELECT id FROM indexing_jobs WHERE status = 'queued'
            )
            AND status = 'running'
            """
        )
        await db.execute(
            """
            UPDATE index_job_items
            SET
                status = 'failed',
                error_code = 'lease_expired',
                error_detail = 'Index worker lease expired; retry limit reached.'
            WHERE job_id IN (
                SELECT id
                FROM indexing_jobs
                WHERE
                    status = 'failed'
                    AND updated_at = ?
                    AND error_message =
                        'Index worker lease expired; retry limit reached.'
            )
            AND status = 'running'
            """,
            (current,),
        )
        await db.commit()
    return requeued.rowcount, failed.rowcount


async def acquire_index_worker_lease(
    settings: AppSettings,
    *,
    worker_id: str,
    now: datetime | None = None,
) -> bool:
    current = now or datetime.now(UTC)
    expires = current + timedelta(seconds=settings.index_worker_lease_seconds)
    async with get_db(settings) as db:
        await db.execute("BEGIN IMMEDIATE")
        cursor = await db.execute(
            """
            SELECT owner, expires_at
            FROM worker_leases
            WHERE name = 'index'
            """
        )
        row = await cursor.fetchone()
        if (
            row is not None
            and str(row["owner"]) != worker_id
            and str(row["expires_at"]) > current.isoformat()
        ):
            await db.rollback()
            return False
        await db.execute(
            """
            INSERT INTO worker_leases(name, owner, expires_at, heartbeat_at)
            VALUES ('index', ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                owner = excluded.owner,
                expires_at = excluded.expires_at,
                heartbeat_at = excluded.heartbeat_at
            """,
            (
                worker_id,
                expires.isoformat(),
                current.isoformat(),
            ),
        )
        await db.commit()
    return True


async def release_index_worker_lease(
    settings: AppSettings,
    *,
    worker_id: str,
) -> None:
    async with get_db(settings) as db:
        await db.execute(
            """
            DELETE FROM worker_leases
            WHERE name = 'index' AND owner = ?
            """,
            (worker_id,),
        )
        await db.commit()


async def claim_next_indexing_job(
    settings: AppSettings,
    *,
    worker_id: str,
    now: datetime | None = None,
) -> IndexingJobRecord | None:
    current = now or datetime.now(UTC)
    lease_expires = current + timedelta(seconds=settings.index_worker_lease_seconds)
    async with get_db(settings) as db:
        await db.execute("BEGIN IMMEDIATE")
        lease_cursor = await db.execute(
            """
            SELECT 1
            FROM worker_leases
            WHERE name = 'index' AND owner = ? AND expires_at > ?
            """,
            (worker_id, current.isoformat()),
        )
        if await lease_cursor.fetchone() is None:
            await db.rollback()
            return None
        cursor = await db.execute(
            """
            SELECT id
            FROM indexing_jobs
            WHERE status = 'queued' AND attempt_count < max_attempts
            ORDER BY created_at, id
            LIMIT 1
            """
        )
        row = await cursor.fetchone()
        if row is None:
            await db.rollback()
            return None
        job_id = str(row["id"])
        result = await db.execute(
            """
            UPDATE indexing_jobs
            SET
                status = 'running',
                attempt_count = attempt_count + 1,
                lease_owner = ?,
                lease_expires_at = ?,
                heartbeat_at = ?,
                updated_at = ?,
                error_message = NULL,
                progress_json = '{"stage":"indexing"}'
            WHERE id = ? AND status = 'queued'
            """,
            (
                worker_id,
                lease_expires.isoformat(),
                current.isoformat(),
                current.isoformat(),
                job_id,
            ),
        )
        if result.rowcount != 1:
            await db.rollback()
            return None
        await db.execute(
            """
            UPDATE index_job_items
            SET status = 'running', error_code = NULL, error_detail = NULL
            WHERE job_id = ? AND status = 'queued'
            """,
            (job_id,),
        )
        await db.commit()
    return await get_indexing_job(settings, job_id=job_id)


async def heartbeat_indexing_job(
    settings: AppSettings,
    *,
    job_id: str,
    worker_id: str,
    progress: dict[str, Any] | None = None,
) -> bool:
    current = datetime.now(UTC)
    lease_expires = current + timedelta(seconds=settings.index_worker_lease_seconds)
    async with get_db(settings) as db:
        result = await db.execute(
            """
            UPDATE indexing_jobs
            SET heartbeat_at = ?, lease_expires_at = ?, updated_at = ?,
                progress_json = COALESCE(?, progress_json)
            WHERE id = ? AND status = 'running' AND lease_owner = ?
            """,
            (
                current.isoformat(),
                lease_expires.isoformat(),
                current.isoformat(),
                json.dumps(progress, ensure_ascii=False) if progress else None,
                job_id,
                worker_id,
            ),
        )
        await db.commit()
    return result.rowcount == 1


async def fail_or_retry_indexing_job(
    settings: AppSettings,
    *,
    job_id: str,
    worker_id: str,
    error_message: str,
    retryable: bool = True,
) -> str:
    now = datetime.now(UTC).isoformat()
    async with get_db(settings) as db:
        await db.execute("BEGIN IMMEDIATE")
        cursor = await db.execute(
            """
            SELECT attempt_count, max_attempts
            FROM indexing_jobs
            WHERE id = ? AND status = 'running' AND lease_owner = ?
            """,
            (job_id, worker_id),
        )
        row = await cursor.fetchone()
        if row is None:
            await db.rollback()
            return "cancelled"
        next_status = (
            "queued"
            if retryable
            and int(row["attempt_count"]) < int(row["max_attempts"])
            else "failed"
        )
        await db.execute(
            """
            UPDATE indexing_jobs
            SET
                status = ?,
                updated_at = ?,
                error_message = ?,
                lease_owner = NULL,
                lease_expires_at = NULL,
                heartbeat_at = NULL
            WHERE id = ? AND lease_owner = ?
            """,
            (next_status, now, error_message, job_id, worker_id),
        )
        await db.execute(
            """
            UPDATE index_job_items
            SET status = ?, error_code = 'indexing_failed', error_detail = ?
            WHERE job_id = ? AND status = 'running'
            """,
            (next_status, error_message, job_id),
        )
        await db.commit()
    return next_status


async def retry_failed_indexing_job(
    settings: AppSettings,
    *,
    job_id: str,
) -> IndexingJobRecord | None:
    now = datetime.now(UTC).isoformat()
    async with get_db(settings) as db:
        await db.execute("BEGIN IMMEDIATE")
        await db.execute(
            """
            UPDATE indexing_jobs
            SET
                status = 'queued',
                updated_at = ?,
                error_message = NULL,
                attempt_count = 0,
                lease_owner = NULL,
                lease_expires_at = NULL,
                heartbeat_at = NULL
            WHERE id = ? AND status = 'failed'
            """,
            (now, job_id),
        )
        await db.execute(
            """
            UPDATE index_job_items
            SET status = 'queued', error_code = NULL, error_detail = NULL
            WHERE job_id = ? AND status = 'failed'
            """,
            (job_id,),
        )
        await db.commit()
    return await get_indexing_job(settings, job_id=job_id)


async def create_index_version_record(
    settings: AppSettings,
    *,
    version_id: str,
) -> None:
    now = datetime.now(UTC).isoformat()
    async with get_db(settings) as db:
        await db.execute(
            """
            INSERT INTO index_versions(id, status, created_at)
            VALUES (?, 'building', ?)
            """,
            (version_id, now),
        )
        await db.commit()


async def mark_index_version_ready(
    settings: AppSettings,
    *,
    version_id: str,
    manifest_path: str,
) -> None:
    async with get_db(settings) as db:
        await db.execute(
            """
            UPDATE index_versions
            SET status = 'ready', manifest_path = ?, error_message = NULL
            WHERE id = ?
            """,
            (manifest_path, version_id),
        )
        await db.commit()


async def mark_index_version_failed(
    settings: AppSettings,
    *,
    version_id: str,
    error_message: str,
) -> None:
    async with get_db(settings) as db:
        await db.execute(
            """
            UPDATE index_versions
            SET status = 'failed', error_message = ?
            WHERE id = ? AND status != 'active'
            """,
            (error_message, version_id),
        )
        await db.commit()


def _indexing_job_from_row(row: aiosqlite.Row) -> IndexingJobRecord:
    request = _json_dict_or_list(row["request_json"], default={})
    progress = _json_dict_or_list(row["progress_json"], default={})
    return IndexingJobRecord(
        id=str(row["id"]),
        status=str(row["status"]),
        created_at=_parse_timestamp(str(row["created_at"])),
        updated_at=_parse_timestamp(str(row["updated_at"])),
        error_message=str(row["error_message"]) if row["error_message"] else None,
        request=request if isinstance(request, dict) else {},
        attempt_count=int(row["attempt_count"]),
        max_attempts=int(row["max_attempts"]),
        lease_owner=str(row["lease_owner"]) if row["lease_owner"] else None,
        lease_expires_at=(
            _parse_timestamp(str(row["lease_expires_at"]))
            if row["lease_expires_at"]
            else None
        ),
        heartbeat_at=(
            _parse_timestamp(str(row["heartbeat_at"]))
            if row["heartbeat_at"]
            else None
        ),
        progress=progress if isinstance(progress, dict) else {},
        active_version_before=(
            str(row["active_version_before"])
            if row["active_version_before"]
            else None
        ),
        target_version=str(row["target_version"]) if row["target_version"] else None,
    )


def _json_dict_or_list(value: Any, *, default: Any) -> Any:
    try:
        return json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return default


def _message_count(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    roles = {"user", "assistant", "system"}
    return sum(
        1
        for item in messages
        if isinstance(item, dict)
        and item.get("role") in roles
        and isinstance(item.get("content"), str)
        and bool(item["content"].strip())
    )


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


async def _upsert_uploaded_paper(
    db: aiosqlite.Connection,
    *,
    item: dict[str, Any],
    created_at: str,
) -> None:
    paper_id = str(item.get("paper_id") or "")
    if not paper_id:
        return
    await db.execute(
        """
        INSERT INTO papers (
            id, content_hash, file_name, source_path, source_type, size_bytes,
            created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO NOTHING
        """,
        (
            paper_id,
            str(item.get("content_hash") or paper_id),
            str(item["filename"]),
            str(item["source_path"]),
            str(item.get("source_type") or "application/octet-stream"),
            int(item.get("size_bytes") or 0),
            created_at,
            created_at,
        ),
    )
