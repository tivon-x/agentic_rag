"""Immutable index versions, embedding contracts, and active pointer handling."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.persistence import load_bm25_bundle
from core.settings import AppSettings
from indexing.indexer import Indexer
from indexing.vectorstore import validate_faiss_persistence


MANIFEST_NAME = "manifest.json"
ACTIVE_POINTER_NAME = "active.json"
_VERSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
logger = logging.getLogger(__name__)


class IndexCompatibilityError(RuntimeError):
    """Raised when query embeddings cannot safely load an active index."""


def embedding_contract(settings: AppSettings) -> dict[str, Any]:
    """Return the non-secret embedding contract persisted in a manifest."""
    offline_provider = "fake" if settings.offline_mode else settings.embedding_provider
    offline_model = "fake-deterministic" if settings.offline_mode else settings.embedding_model
    return {
        "provider": offline_provider,
        "model": offline_model,
        "dimension": settings.embedding_dimensions,
        "input_mode": settings.embedding_input_mode,
        "check_embedding_ctx_length": settings.embedding_check_context_length,
        "max_input_chars": settings.embedding_max_input_chars,
    }


def active_pointer_path(settings: AppSettings) -> Path:
    return _index_root(settings) / ACTIVE_POINTER_NAME


def get_active_version_id(settings: AppSettings) -> str | None:
    database_version = _active_version_from_database(settings)
    if database_version is not None:
        return database_version
    return _active_version_from_pointer(settings)


def _active_version_from_pointer(settings: AppSettings) -> str | None:
    path = active_pointer_path(settings)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexCompatibilityError(f"Active index pointer is unreadable: {exc}") from exc
    version_id = str(payload.get("version_id") or "")
    if not _VERSION_ID_RE.fullmatch(version_id):
        raise IndexCompatibilityError("Active index pointer contains an invalid version.")
    return version_id


def get_active_version_dir(settings: AppSettings) -> Path | None:
    version_id = get_active_version_id(settings)
    if version_id is None:
        return None
    version_dir = (_index_root(settings) / version_id).resolve()
    root = _index_root(settings).resolve()
    if not version_dir.is_relative_to(root) or not version_dir.is_dir():
        raise IndexCompatibilityError(
            f"Active index version {version_id} does not exist."
        )
    return version_dir


def load_manifest(version_dir: Path) -> dict[str, Any]:
    manifest_path = version_dir / MANIFEST_NAME
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexCompatibilityError(
            f"Index manifest is missing or unreadable: {manifest_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise IndexCompatibilityError("Index manifest must be a JSON object.")
    return payload


def validate_embedding_compatibility(
    settings: AppSettings,
    manifest: dict[str, Any],
) -> None:
    expected = embedding_contract(settings)
    actual = manifest.get("embedding")
    if not isinstance(actual, dict):
        raise IndexCompatibilityError(
            "Index manifest has no embedding contract; rebuild the index."
        )
    incompatible = [
        key for key, expected_value in expected.items() if actual.get(key) != expected_value
    ]
    if incompatible:
        details = ", ".join(
            f"{key}: index={actual.get(key)!r}, current={expected[key]!r}"
            for key in incompatible
        )
        raise IndexCompatibilityError(
            f"Embedding configuration is incompatible ({details}); rebuild the index."
        )


def resolve_indexer_config(settings: AppSettings) -> tuple[dict[str, Any], str]:
    """Resolve the query index, with an explicit legacy read adapter."""
    if settings.index_write_mode == "legacy":
        return settings.indexer_config(), "legacy"
    version_dir = get_active_version_dir(settings)
    if version_dir is None:
        if _legacy_index_exists(settings):
            raise IndexCompatibilityError(
                "A legacy index has no embedding contract. Set "
                "INDEX_WRITE_MODE=legacy for read-only rollback, or rebuild "
                "a versioned index from source files."
            )
        return settings.indexer_config(), "uninitialized"
    manifest = load_manifest(version_dir)
    validate_embedding_compatibility(settings, manifest)
    validate_index_version(version_dir)
    return settings.indexer_config(version_dir=version_dir), str(
        manifest.get("version_id") or version_dir.name
    )


def create_index_version(
    settings: AppSettings,
    *,
    source_paths: list[Path] | None = None,
    documents: list[Any] | None = None,
    index_mode: str,
    version_id: str | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> tuple[str, Path]:
    """Build and validate an immutable version without changing the active pointer."""
    chosen_id = version_id or uuid4().hex
    if not _VERSION_ID_RE.fullmatch(chosen_id):
        raise ValueError("Index version id must be a lowercase UUID hex value.")
    root = _index_root(settings)
    root.mkdir(parents=True, exist_ok=True)
    staging_dir = root / f".building-{chosen_id}"
    final_dir = root / chosen_id
    if staging_dir.exists() or final_dir.exists():
        raise FileExistsError(f"Index version already exists: {chosen_id}")
    staging_dir.mkdir(parents=True)

    try:
        if documents is None:
            _seed_staging_version(settings, staging_dir)
        config = settings.indexer_config(version_dir=staging_dir)
        config["index_mode"] = index_mode
        config.update(config_overrides or {})
        indexer = Indexer(config)
        if documents is not None:
            result = indexer.index_documents(documents)
            if result is None:
                raise ValueError("No catalog passages are available for indexing.")
        else:
            for source_path in source_paths or []:
                result = indexer.index(str(source_path))
                if result is None:
                    raise ValueError(
                        f"No indexable content found in {source_path.name}."
                    )

        manifest = {
            "schema_version": 1,
            "version_id": chosen_id,
            "status": "ready",
            "created_at": datetime.now(UTC).isoformat(),
            "embedding": embedding_contract(settings),
            "index_mode": index_mode,
            "leaf_node_type": str(config.get("leaf_node_type", "")),
            "parent_embed_pooling": str(config.get("parent_embed_pooling", "")),
            "document_schema": (
                "paper-passages-v1" if documents is not None else "legacy-documents"
            ),
            "code_version": _code_version(settings.base_dir),
        }
        (staging_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        validate_index_version(staging_dir, expected_version_id=chosen_id)
        os.replace(staging_dir, final_dir)
        return chosen_id, final_dir
    except Exception as exc:
        _preserve_failed_version(root, staging_dir, chosen_id, exc)
        raise


def activate_index_version(
    settings: AppSettings,
    version_id: str,
    *,
    job_id: str | None = None,
    worker_id: str | None = None,
) -> Path:
    """Atomically switch the active pointer to a validated compatible version."""
    if not _VERSION_ID_RE.fullmatch(version_id):
        raise ValueError("Invalid index version id.")
    version_dir = _index_root(settings) / version_id
    manifest = load_manifest(version_dir)
    validate_embedding_compatibility(settings, manifest)
    validate_index_version(version_dir)

    pointer = active_pointer_path(settings)
    database_updated = _mirror_active_version_to_database(
        settings,
        version_id=version_id,
        manifest=manifest,
        job_id=job_id,
        worker_id=worker_id,
    )
    try:
        _write_active_pointer(settings, version_id)
    except OSError:
        if not database_updated:
            raise
        logger.warning(
            "Active index database state committed, but active.json mirror "
            "could not be replaced; startup reconciliation will retry.",
            exc_info=True,
        )
    return pointer


def reconcile_active_pointer(settings: AppSettings) -> Path | None:
    """Reconcile a validated file-only deployment into authoritative SQLite."""
    version_id = _active_version_from_database(settings)
    if version_id is None:
        version_id = _active_version_from_pointer(settings)
        if version_id is None:
            return None
        version_dir = _index_root(settings) / version_id
        manifest = load_manifest(version_dir)
        validate_embedding_compatibility(settings, manifest)
        validate_index_version(version_dir, expected_version_id=version_id)
        if not _mirror_active_version_to_database(
            settings,
            version_id=version_id,
            manifest=manifest,
        ):
            raise IndexCompatibilityError(
                "Cannot import active.json without an initialized database."
            )
    try:
        return _write_active_pointer(settings, version_id)
    except OSError:
        logger.warning(
            "Could not reconcile active.json from SQLite active state.",
            exc_info=True,
        )
        return None


def validate_index_version(
    version_dir: Path,
    *,
    expected_version_id: str | None = None,
) -> None:
    """Validate the minimum persisted artifacts before activation or loading."""
    manifest = load_manifest(version_dir)
    expected_id = expected_version_id or version_dir.name
    if (
        manifest.get("status") != "ready"
        or manifest.get("version_id") != expected_id
    ):
        raise ValueError(
            "Index version validation failed; manifest identity or status is invalid."
        )
    required = (
        version_dir / "faiss" / "index.faiss",
        version_dir / "faiss" / "index.pkl",
        version_dir / "bm25.pkl",
        version_dir / MANIFEST_NAME,
    )
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"Index version validation failed; missing artifacts: {missing}")
    embedding = manifest.get("embedding")
    dimension = embedding.get("dimension") if isinstance(embedding, dict) else None
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError(
            "Index version validation failed; embedding dimension is invalid."
        )
    try:
        validate_faiss_persistence(
            version_dir / "faiss",
            expected_dimension=dimension,
        )
        bundle = load_bm25_bundle(version_dir / "bm25.pkl")
    except Exception as exc:
        raise ValueError(
            "Index version validation failed; persisted artifacts are unreadable."
        ) from exc
    if len(bundle.documents) != len(bundle.tokenized_corpus):
        raise ValueError(
            "Index version validation failed; BM25 documents and tokens differ."
        )


def _seed_staging_version(settings: AppSettings, staging_dir: Path) -> None:
    active_dir = get_active_version_dir(settings)
    if active_dir is not None:
        validate_embedding_compatibility(settings, load_manifest(active_dir))
        for child in active_dir.iterdir():
            if child.name == MANIFEST_NAME:
                continue
            destination = staging_dir / child.name
            if child.is_dir():
                shutil.copytree(child, destination)
            else:
                shutil.copy2(child, destination)
        return


def _preserve_failed_version(
    root: Path,
    staging_dir: Path,
    version_id: str,
    error: Exception,
) -> None:
    if not staging_dir.exists():
        return
    failed_root = root / "failed"
    failed_root.mkdir(parents=True, exist_ok=True)
    failed_dir = failed_root / version_id
    (staging_dir / "failure.json").write_text(
        json.dumps(
            {
                "version_id": version_id,
                "failed_at": datetime.now(UTC).isoformat(),
                "error": str(error),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(staging_dir, failed_dir)


def _code_version(base_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=base_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _active_version_from_database(settings: AppSettings) -> str | None:
    db_path = settings.app_db_path
    if db_path is None or not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path) as db:
            db.execute("BEGIN")
            table = db.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'app_state'
                """
            ).fetchone()
            if table is None:
                return None
            row = db.execute(
                """
                SELECT value_json FROM app_state
                WHERE key = 'active_index_version'
                """
            ).fetchone()
            if row is None:
                return None
            try:
                version_id = str(json.loads(row[0]).get("version_id") or "")
            except (AttributeError, json.JSONDecodeError, TypeError) as exc:
                raise IndexCompatibilityError(
                    "Database active index state is invalid."
                ) from exc
            if not _VERSION_ID_RE.fullmatch(version_id):
                raise IndexCompatibilityError(
                    "Database active index state contains an invalid version."
                )
            status_row = db.execute(
                "SELECT status FROM index_versions WHERE id = ?",
                (version_id,),
            ).fetchone()
    except sqlite3.Error as exc:
        raise IndexCompatibilityError(
            f"Cannot read active index state from {db_path}: {exc}"
        ) from exc
    if status_row is None or status_row[0] != "active":
        raise IndexCompatibilityError(
            "Database active index state does not reference an active version."
        )
    return version_id


def _mirror_active_version_to_database(
    settings: AppSettings,
    *,
    version_id: str,
    manifest: dict[str, Any],
    job_id: str | None = None,
    worker_id: str | None = None,
) -> bool:
    db_path = settings.app_db_path
    if db_path is None or not db_path.exists():
        if job_id is not None:
            raise IndexCompatibilityError(
                "Index job database is unavailable during activation."
            )
        return False
    activated_at = datetime.now(UTC).isoformat()
    try:
        with sqlite3.connect(db_path) as db:
            table = db.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'index_versions'
                """
            ).fetchone()
            if table is None:
                if job_id is not None:
                    raise IndexCompatibilityError(
                        "Index version tables are unavailable during activation."
                    )
                return False
            db.execute("BEGIN IMMEDIATE")
            if (job_id is None) != (worker_id is None):
                raise ValueError("job_id and worker_id must be provided together.")
            if job_id is not None and worker_id is not None:
                worker_lease = db.execute(
                    """
                    SELECT 1 FROM worker_leases
                    WHERE
                        name = 'index'
                        AND owner = ?
                        AND expires_at > ?
                    """,
                    (worker_id, activated_at),
                ).fetchone()
                if worker_lease is None:
                    raise IndexCompatibilityError(
                        "Index worker lease was lost before active index activation."
                    )
                completed = db.execute(
                    """
                    UPDATE indexing_jobs
                    SET
                        status = 'completed',
                        updated_at = ?,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        heartbeat_at = NULL,
                        target_version = ?,
                        progress_json = '{"stage":"completed"}'
                    WHERE
                        id = ?
                        AND status = 'running'
                        AND lease_owner = ?
                        AND lease_expires_at > ?
                    """,
                    (
                        activated_at,
                        version_id,
                        job_id,
                        worker_id,
                        activated_at,
                    ),
                )
                if completed.rowcount != 1:
                    raise IndexCompatibilityError(
                        "Index job lease was lost before active index activation."
                    )
                db.execute(
                    """
                    UPDATE index_job_items
                    SET status = 'completed'
                    WHERE job_id = ? AND status = 'running'
                    """,
                    (job_id,),
                )
            db.execute(
                """
                INSERT OR IGNORE INTO index_versions(
                    id, status, manifest_path, created_at
                )
                VALUES (?, 'ready', ?, ?)
                """,
                (
                    version_id,
                    str(_index_root(settings) / version_id / MANIFEST_NAME),
                    str(manifest.get("created_at") or activated_at),
                ),
            )
            db.execute(
                """
                UPDATE index_versions
                SET status = 'ready'
                WHERE status = 'active' AND id != ?
                """,
                (version_id,),
            )
            result = db.execute(
                """
                UPDATE index_versions
                SET status = 'active', activated_at = ?
                WHERE id = ? AND status IN ('ready', 'active')
                """,
                (activated_at, version_id),
            )
            if result.rowcount != 1:
                raise IndexCompatibilityError(
                    f"Index version {version_id} cannot become active."
                )
            db.execute(
                """
                INSERT INTO app_state(key, value_json, updated_at)
                VALUES ('active_index_version', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (
                    json.dumps({"version_id": version_id}),
                    activated_at,
                ),
            )
            db.commit()
            return True
    except sqlite3.Error as exc:
        raise IndexCompatibilityError(
            f"Cannot update active index state in {db_path}: {exc}"
        ) from exc


def _write_active_pointer(settings: AppSettings, version_id: str) -> Path:
    pointer = active_pointer_path(settings)
    temporary = pointer.with_name(f".{pointer.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "version_id": version_id,
                "activated_at": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        os.replace(temporary, pointer)
    finally:
        temporary.unlink(missing_ok=True)
    return pointer


def _legacy_index_exists(settings: AppSettings) -> bool:
    return any(
        path.is_file()
        for path in (
            settings.faiss_dir / "index.faiss",
            settings.faiss_dir / "index.pkl",
            settings.bm25_path,
            settings.nodes_path,
            settings.doc_trees_path,
        )
    )


def _index_root(settings: AppSettings) -> Path:
    return settings.index_root or settings.data_dir / "indexes"
