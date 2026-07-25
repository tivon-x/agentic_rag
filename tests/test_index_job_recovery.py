from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from api.db.database import (
    acquire_index_worker_lease,
    claim_next_indexing_job,
    create_indexing_job,
    fail_or_retry_indexing_job,
    get_indexing_job,
    list_index_job_items,
    recover_expired_indexing_jobs,
    retry_failed_indexing_job,
)
from api.services.index_worker import IndexWorker
from core.settings import load_settings


def _settings(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "APP_DB_PATH",
        str(tmp_path / "data" / "api" / "sessions.db"),
    )
    monkeypatch.setenv("UPLOAD_ROOT", str(tmp_path / "data" / "uploads"))
    monkeypatch.setenv("INDEX_WORKER_LEASE_SECONDS", "2")
    monkeypatch.setenv("INDEX_WORKER_HEARTBEAT_SECONDS", "1")
    monkeypatch.setenv("INDEX_WORKER_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("INDEX_WORKER_POLL_SECONDS", "0.01")
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "8")
    return load_settings(base_dir=tmp_path, env_file=tmp_path / "missing.env")


def test_sqlite_lease_allows_one_claim_and_bounded_recovery(
    tmp_path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path, monkeypatch)

    async def scenario() -> None:
        await create_indexing_job(
            settings,
            job_id="recovery-job",
            status="queued",
            created_at="2026-01-01T00:00:00+00:00",
        )
        claims = await asyncio.gather(
            acquire_index_worker_lease(settings, worker_id="worker-a"),
            acquire_index_worker_lease(settings, worker_id="worker-b"),
        )
        lease_owner = "worker-a" if claims[0] else "worker-b"
        assert claims.count(True) == 1
        job_claims = await asyncio.gather(
            claim_next_indexing_job(settings, worker_id="worker-a"),
            claim_next_indexing_job(settings, worker_id="worker-b"),
        )
        claimed = [record for record in job_claims if record is not None]
        assert len(claimed) == 1
        first = claimed[0]
        assert first.lease_owner == lease_owner

        recovered, failed = await recover_expired_indexing_jobs(
            settings,
            now=first.lease_expires_at + timedelta(seconds=1),
        )
        assert (recovered, failed) == (1, 0)

        assert await acquire_index_worker_lease(
            settings,
            worker_id="worker-c",
            now=first.lease_expires_at + timedelta(seconds=1),
        )
        second = await claim_next_indexing_job(
            settings,
            worker_id="worker-c",
            now=first.lease_expires_at + timedelta(seconds=1),
        )
        assert second is not None
        assert second.attempt_count == 2
        assert (
            await fail_or_retry_indexing_job(
                settings,
                job_id=second.id,
                worker_id="worker-c",
                error_message="injected crash",
            )
            == "queued"
        )

        future = second.lease_expires_at + timedelta(seconds=1)
        assert await acquire_index_worker_lease(
            settings,
            worker_id="worker-d",
            now=future,
        )
        third = await claim_next_indexing_job(
            settings,
            worker_id="worker-d",
            now=future,
        )
        assert third is not None
        assert third.attempt_count == 3
        assert (
            await fail_or_retry_indexing_job(
                settings,
                job_id=third.id,
                worker_id="worker-d",
                error_message="final injected failure",
            )
            == "failed"
        )
        failed_record = await get_indexing_job(settings, job_id=third.id)
        assert failed_record is not None
        assert failed_record.status == "failed"

        retried = await retry_failed_indexing_job(settings, job_id=third.id)
        assert retried is not None
        assert retried.status == "queued"
        assert retried.attempt_count == 0

    asyncio.run(scenario())


def test_index_worker_start_is_singleton_per_app(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)

    async def scenario() -> None:
        worker = IndexWorker(settings)
        await worker.start()
        first_task = worker._task
        await worker.start()
        assert worker._task is first_task
        assert worker.is_running
        await worker.stop()

    asyncio.run(scenario())


def test_worker_loop_recovers_from_transient_sqlite_error(
    tmp_path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    worker = IndexWorker(settings)
    calls = 0

    async def flaky_acquire(*args, **kwargs) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise sqlite3.OperationalError("database is locked")
        worker._stop.set()
        return False

    monkeypatch.setattr(
        "api.services.index_worker.acquire_index_worker_lease",
        flaky_acquire,
    )
    asyncio.run(worker._run_loop())

    assert calls == 2


def test_exhausted_recovery_fails_running_items(
    tmp_path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path, monkeypatch)

    async def scenario() -> None:
        await create_indexing_job(
            settings,
            job_id="exhausted-job",
            status="queued",
            created_at="2026-01-01T00:00:00+00:00",
            items=[
                {
                    "filename": "paper.txt",
                    "source_path": str(settings.upload_root / "paper.txt"),
                }
            ],
        )
        current = datetime(2026, 1, 1, tzinfo=UTC)
        outcome = (0, 0)
        for attempt in range(3):
            worker_id = f"worker-{attempt}"
            assert await acquire_index_worker_lease(
                settings,
                worker_id=worker_id,
                now=current,
            )
            claimed = await claim_next_indexing_job(
                settings,
                worker_id=worker_id,
                now=current,
            )
            assert claimed is not None
            assert claimed.lease_expires_at is not None
            current = claimed.lease_expires_at + timedelta(seconds=1)
            outcome = await recover_expired_indexing_jobs(
                settings,
                now=current,
            )
        assert outcome == (0, 1)
        failed = await get_indexing_job(settings, job_id="exhausted-job")
        assert failed is not None
        assert failed.status == "failed"
        items = await list_index_job_items(settings, job_id="exhausted-job")
        assert items[0]["status"] == "failed"
        assert items[0]["error_code"] == "lease_expired"

    asyncio.run(scenario())


def test_legacy_worker_rejects_mutable_index_writes(
    tmp_path,
    monkeypatch,
) -> None:
    settings = replace(_settings(tmp_path, monkeypatch), index_write_mode="legacy")
    source = settings.upload_root / "jobs" / "legacy-job" / "paper.txt"
    source.parent.mkdir(parents=True)
    source.write_text("must not reach mutable legacy index", encoding="utf-8")

    async def scenario() -> None:
        await create_indexing_job(
            settings,
            job_id="legacy-job",
            status="queued",
            created_at="2026-01-01T00:00:00+00:00",
            items=[
                {
                    "filename": source.name,
                    "source_path": str(source),
                }
            ],
        )
        worker = IndexWorker(settings)
        assert await acquire_index_worker_lease(
            settings,
            worker_id=worker.worker_id,
        )
        claimed = await claim_next_indexing_job(
            settings,
            worker_id=worker.worker_id,
        )
        assert claimed is not None
        await worker._run_job(claimed.id, claimed.request or {})
        record = await get_indexing_job(settings, job_id=claimed.id)
        assert record is not None
        assert record.status == "failed"
        assert record.attempt_count == 1

    asyncio.run(scenario())
    assert not (settings.faiss_dir / "index.faiss").exists()
