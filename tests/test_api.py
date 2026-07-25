from __future__ import annotations

import time

from fastapi.testclient import TestClient

from api.db.database import create_indexing_job
from api.main import app
from api.services.graph_cache import invalidate_graph_cache
from core.settings import load_settings


def _configure_tmp_paths(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    index_dir = data_dir / "index"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("INDEX_DIR", str(index_dir))
    monkeypatch.setenv("FAISS_DIR", str(index_dir / "faiss"))
    monkeypatch.setenv("BM25_PATH", str(index_dir / "bm25.pkl"))
    monkeypatch.setenv("NODES_PATH", str(index_dir / "nodes.jsonl"))
    monkeypatch.setenv("DOC_TREES_PATH", str(index_dir / "doc_trees.json"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "logs" / "agentic_rag.log"))
    monkeypatch.setenv("APP_DB_PATH", str(data_dir / "api" / "sessions.db"))
    monkeypatch.setenv("INDEX_ROOT", str(data_dir / "indexes"))
    monkeypatch.setenv("UPLOAD_ROOT", str(data_dir / "uploads"))
    monkeypatch.setenv("INDEX_WRITE_MODE", "versioned")
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "16")
    return data_dir, index_dir


def test_health_endpoint_returns_ok(monkeypatch, tmp_path):
    _configure_tmp_paths(monkeypatch, tmp_path)
    invalidate_graph_cache()

    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_corpus_profile_roundtrip(monkeypatch, tmp_path):
    _, index_dir = _configure_tmp_paths(monkeypatch, tmp_path)
    invalidate_graph_cache()

    payload = {
        "name": "测试知识库",
        "summary": "这是一个测试摘要",
        "coverage": "测试覆盖范围",
        "non_coverage": "天气",
        "usage_notes": "优先回答测试问题",
        "source_examples": ["sample.txt"],
        "recommended_questions": ["这个库讲什么"],
        "forbidden_questions": ["今天温度多少"],
        "domain_keywords": ["测试"],
        "preferred_answer_style": "先结论后证据",
        "primary_entities": ["Agentic RAG"],
    }

    with TestClient(app) as client:
        put_response = client.put("/api/corpus-profile", json=payload)
        get_response = client.get("/api/corpus-profile")

    assert put_response.status_code == 200
    assert get_response.status_code == 200
    assert get_response.json() == payload
    assert (index_dir / "corpus_profile.json").exists()


def test_chat_session_persists_messages_and_can_be_loaded(monkeypatch, tmp_path):
    _configure_tmp_paths(monkeypatch, tmp_path)
    invalidate_graph_cache()

    with TestClient(app) as client:
        create_response = client.post("/api/chat", json={"message": "你好"})
        assert create_response.status_code == 200

        session_id = create_response.json()["session_id"]
        session_response = client.get(f"/api/chat/{session_id}")
        stream_response = client.get(f"/api/chat/stream?session_id={session_id}")

    assert session_response.status_code == 200
    assert session_response.json()["messages"] == [{"role": "user", "content": "你好"}]
    assert stream_response.status_code == 200
    assert 'event: stream-error' in stream_response.text
    assert '"error": "No index loaded."' in stream_response.text


def test_upload_endpoint_creates_job_and_status_can_be_polled(
    monkeypatch,
    tmp_path,
):
    _configure_tmp_paths(monkeypatch, tmp_path)
    invalidate_graph_cache()

    with TestClient(app) as client:
        upload_response = client.post(
            "/api/index/files",
            data={"index_mode": "flat"},
            files={"files": ("sample.txt", b"hello api", "text/plain")},
            headers={"Idempotency-Key": "api-upload-test"},
        )
        assert upload_response.status_code == 200
        job_id = upload_response.json()[0]["job_id"]

        deadline = time.monotonic() + 5
        while True:
            status_response = client.get(f"/api/indexing-jobs/{job_id}")
            if status_response.json()["status"] in {"completed", "failed"}:
                break
            if time.monotonic() >= deadline:
                raise AssertionError("Index worker did not finish within five seconds.")
            time.sleep(0.05)

    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"


def test_indexing_job_detail_returns_created_job(monkeypatch, tmp_path):
    _configure_tmp_paths(monkeypatch, tmp_path)
    invalidate_graph_cache()
    settings = load_settings(base_dir=tmp_path, env_file=tmp_path / ".env")

    job_id = "job-123"
    created_at = "2026-03-19T00:00:00+00:00"

    import asyncio

    asyncio.run(
        create_indexing_job(
            settings,
            job_id=job_id,
            status="completed",
            created_at=created_at,
        )
    )

    with TestClient(app) as client:
        response = client.get(f"/api/indexing-jobs/{job_id}")

    assert response.status_code == 200
    assert response.json()["id"] == job_id
    assert response.json()["status"] == "completed"


def test_versioned_index_worker_activates_only_completed_batch(
    monkeypatch,
    tmp_path,
) -> None:
    data_dir, _ = _configure_tmp_paths(monkeypatch, tmp_path)
    monkeypatch.setenv("INDEX_WRITE_MODE", "versioned")
    invalidate_graph_cache()

    with TestClient(app) as client:
        upload_response = client.post(
            "/api/index/files",
            files=[
                ("files", ("paper-a.txt", b"paper A", "text/plain")),
                ("files", ("paper-b.txt", b"paper B", "text/plain")),
            ],
            headers={"Idempotency-Key": "versioned-batch"},
        )
        assert upload_response.status_code == 200
        assert len({item["job_id"] for item in upload_response.json()}) == 1
        job_id = upload_response.json()[0]["job_id"]

        deadline = time.monotonic() + 5
        while True:
            status_response = client.get(f"/api/indexing-jobs/{job_id}")
            payload = status_response.json()
            if payload["status"] in {"completed", "failed"}:
                break
            if time.monotonic() >= deadline:
                raise AssertionError("Versioned worker did not finish.")
            time.sleep(0.05)

    assert payload["status"] == "completed", payload["error_message"]
    assert payload["target_version"]
    assert (data_dir / "indexes" / "active.json").exists()
