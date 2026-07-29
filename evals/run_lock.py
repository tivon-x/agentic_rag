"""Cross-platform exclusive locks for reproducible evaluation run directories."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Iterator, TextIO


@contextmanager
def exclusive_run_lock(run_dir: Path) -> Iterator[Path]:
    """Fail immediately when another process is writing the same run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        _lock_handle(handle)
    except OSError as exc:
        handle.close()
        raise RuntimeError(
            f"Evaluation run directory is already active: {run_dir}"
        ) from exc
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield lock_path
    finally:
        _unlock_handle(handle)
        handle.close()


def _lock_handle(handle: TextIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        if not handle.read(1):
            handle.seek(0)
            handle.write("\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_handle(handle: TextIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
