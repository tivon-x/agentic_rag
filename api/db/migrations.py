"""Forward-only SQLite schema migrations with pre-migration recovery backups."""

from __future__ import annotations

import sqlite3
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite


CURRENT_SCHEMA_VERSION = 2
Migration = Callable[[aiosqlite.Connection], Awaitable[None]]


def _has_table_sync(path: Path, table_name: str) -> bool:
    if not path.exists():
        return False
    with sqlite3.connect(path) as db:
        row = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
    return row is not None


def _current_version_sync(path: Path) -> int:
    if not _has_table_sync(path, "schema_migrations"):
        return 0
    with sqlite3.connect(path) as db:
        row = db.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
    return int(row[0]) if row else 0


def create_recovery_backup(path: Path, from_version: int) -> Path:
    """Create a consistent SQLite backup before a sessions.db migration."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = path.with_name(
        f"{path.name}.backup-v{from_version}-{timestamp}"
    )
    with sqlite3.connect(path) as source, sqlite3.connect(backup_path) as target:
        source.backup(target)
    return backup_path


async def _migration_1(db: aiosqlite.Connection) -> None:
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


async def _migration_2(db: aiosqlite.Connection) -> None:
    await db.execute("ALTER TABLE indexing_jobs RENAME TO indexing_jobs_legacy")
    await db.execute(
        """
        CREATE TABLE indexing_jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL CHECK (
                status IN ('queued', 'running', 'completed', 'failed', 'cancelled')
            ),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            error_message TEXT,
            request_json TEXT NOT NULL DEFAULT '{}',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 3,
            lease_owner TEXT,
            lease_expires_at TEXT,
            heartbeat_at TEXT,
            progress_json TEXT NOT NULL DEFAULT '{}',
            active_version_before TEXT,
            target_version TEXT
        )
        """
    )
    await db.execute(
        """
        INSERT INTO indexing_jobs (
            id, status, created_at, updated_at, error_message
        )
        SELECT
            id,
            CASE
                WHEN status = 'pending' THEN 'queued'
                WHEN status IN (
                    'queued', 'running', 'completed', 'failed', 'cancelled'
                ) THEN status
                ELSE 'failed'
            END,
            created_at,
            updated_at,
            error_message
        FROM indexing_jobs_legacy
        """
    )
    await db.execute("DROP TABLE indexing_jobs_legacy")
    await db.execute(
        """
        CREATE TABLE index_job_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL REFERENCES indexing_jobs(id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            source_path TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('queued', 'running', 'completed', 'failed', 'cancelled')
            ),
            error_code TEXT,
            error_detail TEXT,
            UNIQUE(job_id, source_path)
        )
        """
    )
    await db.execute(
        """
        CREATE TABLE idempotency_records (
            scope TEXT NOT NULL,
            key TEXT NOT NULL,
            request_hash TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            PRIMARY KEY(scope, key)
        )
        """
    )
    await db.execute(
        """
        CREATE TABLE index_versions (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL CHECK (
                status IN ('building', 'ready', 'active', 'failed')
            ),
            manifest_path TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            activated_at TEXT
        )
        """
    )
    await db.execute(
        """
        CREATE TABLE app_state (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    await db.execute(
        """
        CREATE TABLE worker_leases (
            name TEXT PRIMARY KEY,
            owner TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            heartbeat_at TEXT NOT NULL
        )
        """
    )
    await db.execute(
        """
        CREATE INDEX idx_indexing_jobs_claim
        ON indexing_jobs(status, lease_expires_at, created_at)
        """
    )


MIGRATIONS: dict[int, Migration] = {
    1: _migration_1,
    2: _migration_2,
}


async def migrate_database(path: Path) -> list[Path]:
    """Migrate a database to the current schema and return backups created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists() and path.stat().st_size > 0
    initial_version = _current_version_sync(path) if existed else 0
    if initial_version > CURRENT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema version {initial_version} is newer than supported "
            f"version {CURRENT_SCHEMA_VERSION}."
        )
    backups: list[Path] = []
    if existed and initial_version < CURRENT_SCHEMA_VERSION:
        backups.append(create_recovery_backup(path, initial_version))

    async with aiosqlite.connect(path) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute("PRAGMA journal_mode = WAL")
        had_unversioned_schema = (
            initial_version == 0
            and existed
            and (
                await _table_exists(db, "chat_sessions")
                or await _table_exists(db, "indexing_jobs")
            )
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """
        )
        await db.commit()
        if had_unversioned_schema:
            await db.execute("BEGIN IMMEDIATE")
            await _migration_1(db)
            await db.execute(
                """
                INSERT OR IGNORE INTO schema_migrations(version, applied_at)
                VALUES (1, ?)
                """,
                (datetime.now(UTC).isoformat(),),
            )
            await db.commit()
            initial_version = 1

        for version in range(initial_version + 1, CURRENT_SCHEMA_VERSION + 1):
            await db.execute("BEGIN IMMEDIATE")
            try:
                current_version = await _current_version(db)
                if current_version > CURRENT_SCHEMA_VERSION:
                    raise RuntimeError(
                        f"Database schema version {current_version} is newer than "
                        f"supported version {CURRENT_SCHEMA_VERSION}."
                    )
                if current_version >= version:
                    await db.rollback()
                    continue
                migration = MIGRATIONS[version]
                await migration(db)
                await db.execute(
                    """
                    INSERT INTO schema_migrations(version, applied_at)
                    VALUES (?, ?)
                    """,
                    (version, datetime.now(UTC).isoformat()),
                )
            except Exception:
                await db.rollback()
                raise
            else:
                await db.commit()
    return backups


async def _table_exists(db: aiosqlite.Connection, table_name: str) -> bool:
    cursor = await db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    )
    return await cursor.fetchone() is not None


async def _current_version(db: aiosqlite.Connection) -> int:
    cursor = await db.execute(
        "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
    )
    row = await cursor.fetchone()
    return int(row[0]) if row else 0
