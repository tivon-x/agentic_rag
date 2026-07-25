from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

from api.db.models import ChatSessionRecord, IndexingJobRecord
from core.settings import AppSettings


def _db_path(settings: AppSettings) -> Path:
    return settings.data_dir / "api" / "sessions.db"


async def init_db(settings: AppSettings) -> Path:
    path = _db_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(path) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                messages TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS indexing_jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error_message TEXT
            )
            """
        )
        await db.commit()
    return path


@asynccontextmanager
async def get_db(settings: AppSettings):
    path = await init_db(settings)
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
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
    value = row["messages"]
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
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

    try:
        messages = json.loads(row["messages"])
    except json.JSONDecodeError:
        messages = []

    return {
        "session_id": str(row["id"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "messages": messages if isinstance(messages, list) else [],
    }


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
) -> IndexingJobRecord:
    async with get_db(settings) as db:
        await db.execute(
            """
            INSERT INTO indexing_jobs (id, status, created_at, updated_at, error_message)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job_id, status, created_at, created_at, error_message),
        )
        await db.commit()
    return IndexingJobRecord(
        id=job_id,
        status=status,
        created_at=_parse_timestamp(created_at),
        updated_at=_parse_timestamp(created_at),
        error_message=error_message,
    )


async def update_indexing_job(
    settings: AppSettings,
    *,
    job_id: str,
    status: str,
    updated_at: str,
    error_message: str | None = None,
) -> None:
    async with get_db(settings) as db:
        await db.execute(
            """
            UPDATE indexing_jobs
            SET status = ?, updated_at = ?, error_message = ?
            WHERE id = ?
            """,
            (status, updated_at, error_message, job_id),
        )
        await db.commit()


async def get_indexing_job(
    settings: AppSettings,
    *,
    job_id: str,
) -> IndexingJobRecord | None:
    async with get_db(settings) as db:
        cursor = await db.execute(
            """
            SELECT id, status, created_at, updated_at, error_message
            FROM indexing_jobs
            WHERE id = ?
            """,
            (job_id,),
        )
        row = await cursor.fetchone()
    if row is None:
        return None
    return IndexingJobRecord(
        id=str(row["id"]),
        status=str(row["status"]),
        created_at=_parse_timestamp(str(row["created_at"])),
        updated_at=_parse_timestamp(str(row["updated_at"])),
        error_message=str(row["error_message"]) if row["error_message"] else None,
    )


def _parse_timestamp(value: str) -> Any:
    from datetime import datetime

    return datetime.fromisoformat(value)
