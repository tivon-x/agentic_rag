from __future__ import annotations

import asyncio
import json
import sqlite3
from argparse import Namespace
from dataclasses import replace
from datetime import timedelta

import pytest
from langchain_core.documents import Document

from core.persistence import load_bm25_bundle, save_bm25_bundle
from core.settings import load_settings
from api.db.database import (
    acquire_index_worker_lease,
    claim_next_indexing_job,
    create_indexing_job,
    init_db,
)
from indexing.bm25_index import create_bm25_bundle
from indexing.index_versions import (
    MANIFEST_NAME,
    IndexCompatibilityError,
    activate_index_version,
    active_pointer_path,
    create_index_version,
    embedding_contract,
    reconcile_active_pointer,
    resolve_indexer_config,
)
from indexing.indexer import Indexer
from indexing.embeddings import FakeEmbeddings
from indexing.vectorstore import FaissVectorStore
from main import build_parser, cmd_index


def _settings(tmp_path, monkeypatch, *, offline: bool):
    data_dir = tmp_path / "data"
    index_dir = data_dir / "index"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("INDEX_DIR", str(index_dir))
    monkeypatch.setenv("FAISS_DIR", str(index_dir / "faiss"))
    monkeypatch.setenv("BM25_PATH", str(index_dir / "bm25.pkl"))
    monkeypatch.setenv("NODES_PATH", str(index_dir / "nodes.jsonl"))
    monkeypatch.setenv("DOC_TREES_PATH", str(index_dir / "doc_trees.json"))
    monkeypatch.setenv("INDEX_ROOT", str(data_dir / "indexes"))
    monkeypatch.setenv("EMBEDDING_MODEL", "model-a")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "16")
    monkeypatch.setenv("EMBEDDING_INPUT_MODE", "raw")
    monkeypatch.setenv("EMBEDDING_CHECK_CONTEXT_LENGTH", "false")
    monkeypatch.setenv("OFFLINE_MODE", "1" if offline else "0")
    return load_settings(base_dir=tmp_path, env_file=tmp_path / "missing.env")


def test_version_manifest_is_secret_free_and_activation_is_atomic(
    tmp_path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path, monkeypatch, offline=True)
    source = tmp_path / "paper.txt"
    source.write_text("A reliable immutable index version.", encoding="utf-8")

    version_id, version_dir = create_index_version(
        settings,
        source_paths=[source],
        index_mode="flat",
    )
    assert not active_pointer_path(settings).exists()
    manifest = json.loads(
        (version_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["embedding"] == embedding_contract(settings)
    assert manifest["code_version"]
    assert "api_key" not in json.dumps(manifest).lower()

    activate_index_version(settings, version_id)
    original_pointer = active_pointer_path(settings).read_text(encoding="utf-8")

    invalid_id = "f" * 32
    invalid_dir = settings.index_root / invalid_id
    invalid_dir.mkdir()
    (invalid_dir / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version_id": invalid_id,
                "status": "ready",
                "embedding": embedding_contract(settings),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="validation failed"):
        activate_index_version(settings, invalid_id)
    assert active_pointer_path(settings).read_text(encoding="utf-8") == original_pointer


@pytest.mark.parametrize(
    "replacement",
    [
        {"embedding_model": "model-b"},
        {"embedding_dimensions": 32},
        {
            "embedding_input_mode": "tokenized",
            "embedding_check_context_length": True,
        },
    ],
)
def test_incompatible_embedding_contract_refuses_active_index(
    tmp_path,
    monkeypatch,
    replacement,
) -> None:
    settings = _settings(tmp_path, monkeypatch, offline=False)
    version_id = "a" * 32
    _write_dummy_version(settings, version_id)
    activate_index_version(settings, version_id)

    changed = replace(settings, **replacement)
    with pytest.raises(IndexCompatibilityError, match="rebuild the index"):
        resolve_indexer_config(changed)


def test_legacy_read_adapter_remains_available(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch, offline=True)
    legacy = replace(settings, index_write_mode="legacy")

    config, version = resolve_indexer_config(legacy)

    assert version == "legacy"
    assert config["vectorstore"]["persist_directory"] == str(settings.faiss_dir)


def test_versioned_mode_refuses_and_does_not_seed_uncontracted_legacy_index(
    tmp_path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path, monkeypatch, offline=True)
    legacy_source = tmp_path / "legacy.txt"
    legacy_source.write_text("legacy model A vector", encoding="utf-8")
    Indexer(settings.indexer_config()).index(str(legacy_source))

    with pytest.raises(IndexCompatibilityError, match="INDEX_WRITE_MODE=legacy"):
        resolve_indexer_config(settings)

    new_source = tmp_path / "new.txt"
    new_source.write_text("new model B vector", encoding="utf-8")
    _, version_dir = create_index_version(
        settings,
        source_paths=[new_source],
        index_mode="flat",
    )
    bundle = load_bm25_bundle(version_dir / "bm25.pkl")
    contents = [document.page_content for document in bundle.documents]
    assert any("new model B" in content for content in contents)
    assert all("legacy model A" not in content for content in contents)


def test_corrupt_faiss_refuses_silent_empty_fallback(
    tmp_path,
    monkeypatch,
) -> None:
    faiss_dir = tmp_path / "faiss"
    faiss_dir.mkdir()
    (faiss_dir / "index.faiss").write_bytes(b"bad")
    (faiss_dir / "index.pkl").write_bytes(b"bad")

    def fail_load(*args, **kwargs):
        raise ValueError("corrupt")

    monkeypatch.setattr("indexing.vectorstore.FAISS.load_local", fail_load)

    with pytest.raises(RuntimeError, match="rebuild or roll back"):
        FaissVectorStore(
            embeddings=FakeEmbeddings(dimensions=8),
            persist_directory=str(faiss_dir),
        )


def test_activate_version_updates_database_and_supports_rollback(
    tmp_path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path, monkeypatch, offline=False)
    asyncio.run(init_db(settings))
    first_id = "1" * 32
    second_id = "2" * 32
    _write_dummy_version(settings, first_id)
    _write_dummy_version(settings, second_id)

    activate_index_version(settings, first_id)
    activate_index_version(settings, second_id)
    activate_index_version(settings, first_id)

    pointer = json.loads(
        active_pointer_path(settings).read_text(encoding="utf-8")
    )
    assert pointer["version_id"] == first_id
    with sqlite3.connect(settings.app_db_path) as db:
        active_rows = db.execute(
            "SELECT id FROM index_versions WHERE status = 'active'"
        ).fetchall()
        state = json.loads(
            db.execute(
                """
                SELECT value_json FROM app_state
                WHERE key = 'active_index_version'
                """
            ).fetchone()[0]
        )
    assert active_rows == [(first_id,)]
    assert state["version_id"] == first_id


def _write_dummy_version(settings, version_id: str) -> None:
    version_dir = settings.index_root / version_id
    document = Document(page_content=f"index version {version_id}")
    vector_store = FaissVectorStore(
        embeddings=FakeEmbeddings(dimensions=settings.embedding_dimensions)
    )
    vector_store.add_documents([document])
    vector_store.save(str(version_dir / "faiss"))
    save_bm25_bundle(
        version_dir / "bm25.pkl",
        create_bm25_bundle([document]),
    )
    (version_dir / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version_id": version_id,
                "status": "ready",
                "embedding": embedding_contract(settings),
            }
        ),
        encoding="utf-8",
    )


def test_corrupt_version_cannot_replace_active_pointer(
    tmp_path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path, monkeypatch, offline=False)
    good_id = "3" * 32
    corrupt_id = "4" * 32
    _write_dummy_version(settings, good_id)
    _write_dummy_version(settings, corrupt_id)
    activate_index_version(settings, good_id)
    original_pointer = active_pointer_path(settings).read_text(encoding="utf-8")
    (settings.index_root / corrupt_id / "faiss" / "index.faiss").write_bytes(
        b"not-faiss"
    )

    with pytest.raises(ValueError, match="persisted artifacts are unreadable"):
        activate_index_version(settings, corrupt_id)

    assert active_pointer_path(settings).read_text(encoding="utf-8") == original_pointer


def test_database_activation_survives_pointer_mirror_failure(
    tmp_path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path, monkeypatch, offline=False)
    asyncio.run(init_db(settings))
    version_id = "6" * 32
    _write_dummy_version(settings, version_id)

    with monkeypatch.context() as context:
        context.setattr(
            "indexing.index_versions.os.replace",
            lambda *_: (_ for _ in ()).throw(OSError("injected pointer failure")),
        )
        activate_index_version(settings, version_id)

    with sqlite3.connect(settings.app_db_path) as db:
        active = db.execute(
            "SELECT status FROM index_versions WHERE id = ?",
            (version_id,),
        ).fetchone()
        state = json.loads(
            db.execute(
                """
                SELECT value_json FROM app_state
                WHERE key = 'active_index_version'
                """
            ).fetchone()[0]
        )
    assert active == ("active",)
    assert state["version_id"] == version_id
    assert not active_pointer_path(settings).exists()

    reconcile_active_pointer(settings)
    pointer = json.loads(active_pointer_path(settings).read_text(encoding="utf-8"))
    assert pointer["version_id"] == version_id


def test_cli_first_index_initializes_authoritative_database(
    tmp_path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path, monkeypatch, offline=True)
    source = tmp_path / "paper.txt"
    source.write_text("CLI first index must initialize SQLite.", encoding="utf-8")
    monkeypatch.setattr("main.load_settings", lambda: settings)

    assert settings.app_db_path is not None
    assert not settings.app_db_path.exists()
    result = cmd_index(
        Namespace(
            paths=[str(source)],
            mode="flat",
            leaf_node_type=None,
            parent_embed_pooling=None,
        )
    )

    assert result == 0
    with sqlite3.connect(settings.app_db_path) as db:
        state = json.loads(
            db.execute(
                """
                SELECT value_json FROM app_state
                WHERE key = 'active_index_version'
                """
            ).fetchone()[0]
        )
        active = db.execute(
            "SELECT id FROM index_versions WHERE status = 'active'"
        ).fetchall()
    assert active == [(state["version_id"],)]


def test_startup_imports_valid_file_only_active_pointer(
    tmp_path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path, monkeypatch, offline=False)
    version_id = "7" * 32
    _write_dummy_version(settings, version_id)
    activate_index_version(settings, version_id)

    assert settings.app_db_path is not None
    assert not settings.app_db_path.exists()
    asyncio.run(init_db(settings))
    reconcile_active_pointer(settings)

    with sqlite3.connect(settings.app_db_path) as db:
        state = json.loads(
            db.execute(
                """
                SELECT value_json FROM app_state
                WHERE key = 'active_index_version'
                """
            ).fetchone()[0]
        )
        status = db.execute(
            "SELECT status FROM index_versions WHERE id = ?",
            (version_id,),
        ).fetchone()
    assert state["version_id"] == version_id
    assert status == ("active",)


def test_activate_index_cli_contract() -> None:
    args = build_parser().parse_args(["activate-index", "a" * 32])

    assert args.version_id == "a" * 32
    assert args.func.__name__ == "cmd_activate_index"


def test_lost_worker_lease_cannot_activate_version(
    tmp_path,
    monkeypatch,
) -> None:
    settings = replace(
        _settings(tmp_path, monkeypatch, offline=False),
        index_worker_lease_seconds=2,
        index_worker_heartbeat_seconds=1,
    )
    version_id = "5" * 32
    _write_dummy_version(settings, version_id)

    async def lose_lease() -> None:
        await create_indexing_job(
            settings,
            job_id="lost-lease-job",
            status="queued",
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert await acquire_index_worker_lease(settings, worker_id="worker-a")
        claimed = await claim_next_indexing_job(settings, worker_id="worker-a")
        assert claimed is not None
        assert claimed.lease_expires_at is not None
        assert await acquire_index_worker_lease(
            settings,
            worker_id="worker-b",
            now=claimed.lease_expires_at + timedelta(seconds=1),
        )

    asyncio.run(lose_lease())

    with pytest.raises(IndexCompatibilityError, match="worker lease was lost"):
        activate_index_version(
            settings,
            version_id,
            job_id="lost-lease-job",
            worker_id="worker-a",
        )
    assert not active_pointer_path(settings).exists()
