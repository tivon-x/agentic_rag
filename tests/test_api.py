from __future__ import annotations

import time

from fastapi.testclient import TestClient

from api.db.database import create_indexing_job, update_indexing_job
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
    assert '"error":"No index loaded."' in stream_response.text


def test_upload_endpoint_creates_job_and_status_can_be_polled(
    monkeypatch,
    tmp_path,
):
    _configure_tmp_paths(monkeypatch, tmp_path)
    invalidate_graph_cache()

    async def fake_run_indexing_job(*, settings, job_id, file_path, index_mode):
        assert file_path.exists()
        assert index_mode == "flat"
        await update_indexing_job(
            settings,
            job_id=job_id,
            status="completed",
            updated_at="2026-03-19T00:00:01+00:00",
        )

    monkeypatch.setattr("api.routers.indexing._run_indexing_job", fake_run_indexing_job)

    with TestClient(app) as client:
        upload_response = client.post(
            "/api/index/files",
            data={"index_mode": "flat"},
            files={"files": ("sample.txt", b"hello api", "text/plain")},
        )
        assert upload_response.status_code == 200
        job_id = upload_response.json()[0]["job_id"]

        time.sleep(0.05)
        status_response = client.get(f"/api/indexing-jobs/{job_id}")

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
            status="pending",
            created_at=created_at,
        )
    )

    with TestClient(app) as client:
        response = client.get(f"/api/indexing-jobs/{job_id}")

    assert response.status_code == 200
    assert response.json()["id"] == job_id
    assert response.json()["status"] == "pending"
