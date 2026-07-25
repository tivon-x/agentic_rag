"""Single-process SQLite-leased index worker."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from uuid import uuid4

from api.db.database import (
    acquire_index_worker_lease,
    claim_next_indexing_job,
    create_index_version_record,
    fail_or_retry_indexing_job,
    heartbeat_indexing_job,
    list_index_job_items,
    mark_index_version_failed,
    mark_index_version_ready,
    recover_expired_indexing_jobs,
    release_index_worker_lease,
)
from api.services.graph_cache import invalidate_graph_cache
from core.settings import AppSettings
from indexing.index_versions import (
    MANIFEST_NAME,
    activate_index_version,
    create_index_version,
)


logger = logging.getLogger(__name__)


class _NonRetryableIndexingError(RuntimeError):
    """An indexing failure that must transition directly to failed."""


class IndexWorker:
    """Own exactly one serial index execution loop for a FastAPI process."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.worker_id = f"index-worker-{uuid4().hex}"
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._task = asyncio.create_task(
            self._run_loop(),
            name=self.worker_id,
        )

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._task is None:
            return
        stopped_gracefully = True
        try:
            await asyncio.wait_for(
                asyncio.shield(self._task),
                timeout=2,
            )
        except TimeoutError:
            stopped_gracefully = False
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if stopped_gracefully:
            await release_index_worker_lease(
                self.settings,
                worker_id=self.worker_id,
            )
        self._task = None

    def notify(self) -> None:
        self._wake.set()

    async def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                owns_lease = await acquire_index_worker_lease(
                    self.settings,
                    worker_id=self.worker_id,
                )
                if not owns_lease:
                    await self._wait_for_work()
                    continue
                await recover_expired_indexing_jobs(self.settings)
                job = await claim_next_indexing_job(
                    self.settings,
                    worker_id=self.worker_id,
                )
                if job is None:
                    await self._wait_for_work()
                    continue
                await self._run_job(job.id, job.request or {})
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Index worker loop failed; retrying after backoff.")
                await self._wait_for_work()

    async def _wait_for_work(self) -> None:
        self._wake.clear()
        try:
            await asyncio.wait_for(
                self._wake.wait(),
                timeout=self.settings.index_worker_poll_seconds,
            )
        except TimeoutError:
            pass

    async def _run_job(self, job_id: str, request: dict[str, object]) -> None:
        heartbeat_task = asyncio.create_task(self._heartbeat(job_id))
        version_id: str | None = None
        version_activated = False
        try:
            if self.settings.index_write_mode == "legacy":
                raise _NonRetryableIndexingError(
                    "API indexing is disabled while INDEX_WRITE_MODE=legacy; "
                    "legacy mode is read-only."
                )
            items = await list_index_job_items(self.settings, job_id=job_id)
            source_paths = self._validated_source_paths(items)
            index_mode = str(request.get("index_mode") or self.settings.index_mode)
            version_id = uuid4().hex
            await create_index_version_record(
                self.settings,
                version_id=version_id,
            )
            _, version_dir = await asyncio.to_thread(
                create_index_version,
                self.settings,
                source_paths=source_paths,
                index_mode=index_mode,
                version_id=version_id,
            )
            await self._assert_lease_owned(job_id)
            await mark_index_version_ready(
                self.settings,
                version_id=version_id,
                manifest_path=str(version_dir / MANIFEST_NAME),
            )
            await asyncio.to_thread(
                activate_index_version,
                self.settings,
                version_id,
                job_id=job_id,
                worker_id=self.worker_id,
            )
            version_activated = True
            invalidate_graph_cache()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Indexing job %s failed", job_id)
            if version_id is not None and not version_activated:
                await mark_index_version_failed(
                    self.settings,
                    version_id=version_id,
                    error_message=str(exc),
                )
            await fail_or_retry_indexing_job(
                self.settings,
                job_id=job_id,
                worker_id=self.worker_id,
                error_message=str(exc),
                retryable=not isinstance(exc, _NonRetryableIndexingError),
            )
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception(
                    "Indexing job %s heartbeat failed; lease checks guard activation.",
                    job_id,
                )

    async def _assert_lease_owned(self, job_id: str) -> None:
        owns_worker_lease = await acquire_index_worker_lease(
            self.settings,
            worker_id=self.worker_id,
        )
        if not owns_worker_lease:
            raise RuntimeError(
                "Index worker lease was lost before active index activation."
            )
        owns_job_lease = await heartbeat_indexing_job(
            self.settings,
            job_id=job_id,
            worker_id=self.worker_id,
        )
        if not owns_job_lease:
            raise RuntimeError(
                "Index worker lease was lost before active index activation."
            )

    async def _heartbeat(self, job_id: str) -> None:
        while True:
            await asyncio.sleep(self.settings.index_worker_heartbeat_seconds)
            lease_owned = await acquire_index_worker_lease(
                self.settings,
                worker_id=self.worker_id,
            )
            if not lease_owned:
                return
            owned = await heartbeat_indexing_job(
                self.settings,
                job_id=job_id,
                worker_id=self.worker_id,
            )
            if not owned:
                return

    def _validated_source_paths(
        self,
        items: list[dict[str, object]],
    ) -> list[Path]:
        upload_root = (
            self.settings.upload_root or self.settings.data_dir / "uploads"
        ).resolve()
        paths: list[Path] = []
        for item in items:
            path = Path(str(item["source_path"])).resolve()
            if not path.is_relative_to(upload_root) or not path.is_file():
                raise ValueError("Index job source escaped UPLOAD_ROOT or is missing.")
            paths.append(path)
        if not paths:
            raise ValueError("Indexing job has no uploaded files.")
        return paths
