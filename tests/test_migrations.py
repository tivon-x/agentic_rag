from __future__ import annotations

import asyncio
import sqlite3

import pytest

from api.db.database import init_db
from api.db.migrations import CURRENT_SCHEMA_VERSION, migrate_database
from core.settings import load_settings


def test_legacy_sessions_db_is_backed_up_and_migrated(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "data" / "api" / "sessions.db"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as db:
        db.execute(
            """
            CREATE TABLE chat_sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                messages TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        db.execute(
            """
            CREATE TABLE indexing_jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error_message TEXT
            )
            """
        )
        db.execute(
            """
            INSERT INTO indexing_jobs(id, status, created_at, updated_at)
            VALUES ('legacy-job', 'pending', '2026-01-01', '2026-01-01')
            """
        )
        db.commit()

    monkeypatch.setenv("APP_DB_PATH", str(db_path))
    settings = load_settings(
        base_dir=tmp_path,
        env_file=tmp_path / "missing.env",
    )
    asyncio.run(init_db(settings))

    backups = list(db_path.parent.glob("sessions.db.backup-v0-*"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert (
            backup.execute(
                "SELECT status FROM indexing_jobs WHERE id = 'legacy-job'"
            ).fetchone()[0]
            == "pending"
        )
    with sqlite3.connect(db_path) as db:
        assert (
            db.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
            == CURRENT_SCHEMA_VERSION
        )
        assert (
            db.execute(
                "SELECT status FROM indexing_jobs WHERE id = 'legacy-job'"
            ).fetchone()[0]
            == "queued"
        )


def test_future_database_schema_is_rejected(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """
        )
        db.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            (CURRENT_SCHEMA_VERSION + 1, "2026-01-01"),
        )
        db.commit()

    with pytest.raises(RuntimeError, match="newer than supported"):
        asyncio.run(migrate_database(db_path))


def test_concurrent_new_database_migration_is_serialized(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"

    async def migrate_twice() -> None:
        await asyncio.gather(
            migrate_database(db_path),
            migrate_database(db_path),
        )

    asyncio.run(migrate_twice())

    with sqlite3.connect(db_path) as db:
        versions = db.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        jobs_table = db.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'indexing_jobs'
            """
        ).fetchone()
    assert versions == [(1,), (2,)]
    assert jobs_table == (1,)
