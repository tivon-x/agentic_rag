from __future__ import annotations

import asyncio
from datetime import timedelta

from api.db.database import (
    acquire_index_worker_lease,
    claim_next_indexing_job,
    create_indexing_job,
    fail_or_retry_indexing_job,
    get_indexing_job,
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
