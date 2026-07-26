from __future__ import annotations

import time

import pymupdf
from fastapi.testclient import TestClient

from api.main import app
from api.services.graph_cache import invalidate_graph_cache


def _configure(monkeypatch, tmp_path) -> None:
    data = tmp_path / "data"
    index = data / "index"
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
    page.insert_text((72, 72), "Searchable Evidence Paper", fontsize=22)
    page.insert_text((72, 118), "Abstract", fontsize=15)
    page.insert_text(
        (72, 150),
        "The copper-orchid benchmark reaches 73.4 percent accuracy.",
        fontsize=11,
    )
    payload = document.tobytes()
    document.close()
    return payload


def _wait(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + 20
    while True:
        payload = client.get(f"/api/indexing-jobs/{job_id}").json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        if time.monotonic() >= deadline:
            raise AssertionError("Index worker did not complete.")
        time.sleep(0.05)


def test_search_returns_paper_page_quote_and_score_stages(
    monkeypatch,
    tmp_path,
) -> None:
    _configure(monkeypatch, tmp_path)
    invalidate_graph_cache()

    with TestClient(app) as client:
        uploaded = client.post(
            "/api/index/files",
            files={"files": ("search.pdf", _pdf_bytes(), "application/pdf")},
            headers={"Idempotency-Key": "search-api"},
        )
        job = _wait(client, uploaded.json()[0]["job_id"])
        assert job["status"] == "completed", job["error_message"]

        response = client.get("/api/search", params={"q": "copper-orchid"})

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["paper_id"]
    assert result["section_id"]
    assert result["page_start"] == 1
    assert "copper-orchid" in result["quote_text"]
    assert set(result["scores"]) == {
        "vector",
        "bm25",
        "fusion",
        "boosts",
        "final",
        "rerank_rank",
    }
    assert result["pdf_url"].endswith("#page=1")
    assert result["paper_url"].endswith("?page=1")
