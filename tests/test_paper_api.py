from __future__ import annotations

import sqlite3
import time

import pymupdf
from fastapi.testclient import TestClient

from api.main import app
from api.services.graph_cache import invalidate_graph_cache


def _configure(monkeypatch, tmp_path) -> None:
    data = tmp_path / "data"
    index = data / "index"
    for name in (
        "EMBEDDING_MODEL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_API_BASE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("INDEX_DIR", str(index))
    monkeypatch.setenv("INDEX_ROOT", str(data / "indexes"))
    monkeypatch.setenv("UPLOAD_ROOT", str(data / "uploads"))
    monkeypatch.setenv("PARSED_ARTIFACT_ROOT", str(data / "parsed"))
    monkeypatch.setenv("APP_DB_PATH", str(data / "api" / "sessions.db"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "logs" / "app.log"))
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "16")
    monkeypatch.setenv("RERANKER_BACKEND", "none")
    monkeypatch.setenv("INDEX_WORKER_POLL_SECONDS", "0.05")


def _pdf_bytes() -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Paper API Range Evidence", fontsize=22)
    page.insert_text((72, 115), "Lin Qiao", fontsize=12)
    page.insert_text((72, 150), "Abstract", fontsize=15)
    page.insert_text((72, 180), "A violet-cascade marker appears here.", fontsize=11)
    payload = document.tobytes()
    document.close()
    return payload


def _wait_for_job(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + 20
    while True:
        payload = client.get(f"/api/indexing-jobs/{job_id}").json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        if time.monotonic() >= deadline:
            raise AssertionError("Index worker did not complete.")
        time.sleep(0.05)


def test_paper_catalog_metadata_patch_and_range_response(
    monkeypatch,
    tmp_path,
) -> None:
    _configure(monkeypatch, tmp_path)
    invalidate_graph_cache()
    payload = _pdf_bytes()

    with TestClient(app) as client:
        uploaded = client.post(
            "/api/index/files",
            files={"files": ("range-paper.pdf", payload, "application/pdf")},
            headers={"Idempotency-Key": "paper-range"},
        )
        assert uploaded.status_code == 200
        job = _wait_for_job(client, uploaded.json()[0]["job_id"])
        assert job["status"] == "completed", job["error_message"]

        library = client.get("/api/papers")
        assert library.status_code == 200
        paper = library.json()["items"][0]
        paper_id = paper["id"]
        detail = client.get(f"/api/papers/{paper_id}")
        assert detail.status_code == 200
        assert detail.json()["sections"]
        assert detail.json()["file_url"] == f"/api/papers/{paper_id}/file"

        byte_range = client.get(
            f"/api/papers/{paper_id}/file",
            headers={"Range": "bytes=0-31"},
        )
        assert byte_range.status_code == 206
        assert byte_range.headers["accept-ranges"] == "bytes"
        assert byte_range.headers["content-range"].startswith("bytes 0-31/")
        assert len(byte_range.content) == 32

        patched = client.patch(
            f"/api/papers/{paper_id}",
            headers={"If-Match": str(detail.json()["metadata_version"])},
            json={
                "title": "User corrected title",
                "authors": [],
                "year": 2026,
                "venue": None,
                "doi": None,
                "arxiv_id": None,
            },
        )
        assert patched.status_code == 200
        assert patched.json()["title"] == "User corrected title"
        assert patched.json()["metadata_status"] == "verified"
        assert patched.json()["reindex_job_id"]


def test_invalid_or_multiple_range_is_rejected(monkeypatch, tmp_path) -> None:
    _configure(monkeypatch, tmp_path)
    invalidate_graph_cache()

    with TestClient(app) as client:
        uploaded = client.post(
            "/api/index/files",
            files={"files": ("paper.pdf", _pdf_bytes(), "application/pdf")},
            headers={"Idempotency-Key": "paper-bad-range"},
        )
        job = _wait_for_job(client, uploaded.json()[0]["job_id"])
        assert job["status"] == "completed"
        paper_id = client.get("/api/papers").json()["items"][0]["id"]
        response = client.get(
            f"/api/papers/{paper_id}/file",
            headers={"Range": "bytes=0-1,4-5"},
        )

    assert response.status_code == 416


def test_metadata_patch_rolls_back_when_reindex_job_cannot_be_queued(
    monkeypatch,
    tmp_path,
) -> None:
    _configure(monkeypatch, tmp_path)
    invalidate_graph_cache()

    with TestClient(app, raise_server_exceptions=False) as client:
        uploaded = client.post(
            "/api/index/files",
            files={"files": ("paper.pdf", _pdf_bytes(), "application/pdf")},
            headers={"Idempotency-Key": "paper-metadata-atomicity"},
        )
        job = _wait_for_job(client, uploaded.json()[0]["job_id"])
        assert job["status"] == "completed"
        paper = client.get("/api/papers").json()["items"][0]
        paper_id = paper["id"]
        original_title = paper["title"]
        original_version = paper["metadata_version"]

        with sqlite3.connect(
            tmp_path / "data" / "api" / "sessions.db",
        ) as db:
            db.execute(
                """
                CREATE TRIGGER reject_metadata_reindex
                BEFORE INSERT ON indexing_jobs
                WHEN NEW.request_json LIKE '%metadata_reindex%'
                BEGIN
                    SELECT RAISE(ABORT, 'synthetic reindex insert failure');
                END
                """
            )

        response = client.patch(
            f"/api/papers/{paper_id}",
            headers={"If-Match": str(original_version)},
            json={"title": "Must Roll Back"},
        )
        after = client.get(f"/api/papers/{paper_id}").json()

    assert response.status_code == 500
    assert after["title"] == original_title
    assert after["metadata_version"] == original_version
