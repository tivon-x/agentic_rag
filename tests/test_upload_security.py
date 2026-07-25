from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace

from fastapi.testclient import TestClient

from api.db.database import create_indexing_job, get_indexing_job
from api.main import create_app
from api.services.index_worker import IndexWorker
from core.settings import load_settings


def _settings(tmp_path, monkeypatch, *, max_bytes: int = 1024):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("APP_DB_PATH", str(data_dir / "api" / "sessions.db"))
    monkeypatch.setenv("UPLOAD_ROOT", str(data_dir / "uploads"))
    monkeypatch.setenv("INDEX_ROOT", str(data_dir / "indexes"))
    monkeypatch.setenv("UPLOAD_MAX_BYTES", str(max_bytes))
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "8")
    return load_settings(base_dir=tmp_path, env_file=tmp_path / "missing.env")


def _disable_worker(monkeypatch) -> None:
    async def no_start(self: IndexWorker) -> None:
        return None

    monkeypatch.setattr(IndexWorker, "start", no_start)


def test_upload_rejects_path_traversal_and_stays_inside_upload_root(
    tmp_path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _disable_worker(monkeypatch)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/index/files",
            data={"index_mode": "flat"},
            files={"files": ("../escape.txt", b"malicious", "text/plain")},
            headers={"Idempotency-Key": "traversal"},
        )

    assert response.status_code == 400
    assert not (settings.upload_root.parent / "escape.txt").exists()


def test_upload_enforces_size_limit(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch, max_bytes=4)
    _disable_worker(monkeypatch)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/index/files",
            files={"files": ("paper.txt", b"12345", "text/plain")},
            headers={"Idempotency-Key": "too-large"},
        )

    assert response.status_code == 413
    assert list(settings.upload_root.rglob("paper.txt")) == []


def test_upload_requires_idempotency_key(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _disable_worker(monkeypatch)

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/index/files",
            files={"files": ("paper.txt", b"paper", "text/plain")},
        )

    assert response.status_code == 422


def test_duplicate_idempotency_key_reuses_one_job(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    _disable_worker(monkeypatch)
    app = create_app(settings)

    with TestClient(app) as client:
        first = client.post(
            "/api/index/files",
            files={"files": ("paper.txt", b"same paper", "text/plain")},
            headers={"Idempotency-Key": "same-request"},
        )
        repeated = client.post(
            "/api/index/files",
            files={"files": ("paper.txt", b"same paper", "text/plain")},
            headers={"Idempotency-Key": "same-request"},
        )
        conflict = client.post(
            "/api/index/files",
            files={"files": ("paper.txt", b"different paper", "text/plain")},
            headers={"Idempotency-Key": "same-request"},
        )

    assert first.status_code == 200
    assert repeated.status_code == 200
    assert first.json() == repeated.json()
    assert first.json()[0]["status"] == "queued"
    assert conflict.status_code == 409
    with sqlite3.connect(settings.app_db_path) as db:
        count = db.execute("SELECT COUNT(*) FROM indexing_jobs").fetchone()[0]
    assert count == 1
    uploaded = list((settings.upload_root / "jobs").rglob("paper.txt"))
    assert len(uploaded) == 1
    assert uploaded[0].resolve().is_relative_to(settings.upload_root.resolve())


def test_api_upload_is_read_only_in_legacy_mode(tmp_path, monkeypatch) -> None:
    settings = replace(
        _settings(tmp_path, monkeypatch),
        index_write_mode="legacy",
    )
    source = settings.upload_root / "jobs" / "queued-job" / "paper.txt"
    source.parent.mkdir(parents=True)
    source.write_text("queued before rollback", encoding="utf-8")
    asyncio.run(
        create_indexing_job(
            settings,
            job_id="queued-job",
            status="queued",
            created_at="2026-01-01T00:00:00+00:00",
            items=[
                {
                    "filename": source.name,
                    "source_path": str(source),
                }
            ],
        )
    )
    app = create_app(settings)

    with TestClient(app) as client:
        assert not app.state.index_worker.is_running
        response = client.post(
            "/api/index/files",
            files={"files": ("paper.txt", b"paper", "text/plain")},
            headers={"Idempotency-Key": "legacy-read-only"},
        )

    assert response.status_code == 409
    queued = asyncio.run(get_indexing_job(settings, job_id="queued-job"))
    assert queued is not None
    assert queued.status == "queued"
    with sqlite3.connect(settings.app_db_path) as db:
        count = db.execute("SELECT COUNT(*) FROM indexing_jobs").fetchone()[0]
    assert count == 1


def test_legacy_api_ignores_broken_versioned_pointer(
    tmp_path,
    monkeypatch,
) -> None:
    settings = replace(
        _settings(tmp_path, monkeypatch),
        index_write_mode="legacy",
    )
    assert settings.index_root is not None
    settings.index_root.mkdir(parents=True, exist_ok=True)
    (settings.index_root / "active.json").write_text(
        '{"version_id":"' + ("a" * 32) + '"}',
        encoding="utf-8",
    )
    app = create_app(settings)

    with TestClient(app) as client:
        assert not app.state.index_worker.is_running
        response = client.post(
            "/api/index/files",
            files={"files": ("paper.txt", b"paper", "text/plain")},
            headers={"Idempotency-Key": "legacy-broken-pointer"},
        )

    assert response.status_code == 409
