import asyncio

from fastapi.testclient import TestClient

from api.db.database import create_chat_session, get_db
from api.main import app
from api.services.graph_cache import invalidate_graph_cache
from core.settings import load_settings
from tests.test_api import _configure_tmp_paths


def _settings(monkeypatch, tmp_path):
    _configure_tmp_paths(monkeypatch, tmp_path)
    invalidate_graph_cache()
    return load_settings(base_dir=tmp_path, env_file=tmp_path / ".env")


def test_chat_session_list_sorts_paginates_and_titles(monkeypatch, tmp_path):
    settings = _settings(monkeypatch, tmp_path)
    asyncio.run(
        create_chat_session(
            settings,
            session_id="older",
            messages=[{"role": "user", "content": "  older   question  "}],
            created_at="2026-08-01T00:00:00+00:00",
        )
    )
    asyncio.run(
        create_chat_session(
            settings,
            session_id="newer",
            messages=[{"role": "user", "content": "newer\nquestion"}],
            created_at="2026-08-02T00:00:00+00:00",
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/chat?limit=1&offset=0")
        page = client.get("/api/chat?limit=1&offset=1")

    assert response.status_code == 200
    assert response.json()["total"] == 2
    assert response.json()["items"][0]["session_id"] == "newer"
    assert response.json()["items"][0]["title"] == "newer question"
    assert response.json()["items"][0]["message_count"] == 1
    assert page.json()["items"][0]["session_id"] == "older"
    assert page.json()["items"][0]["title"] == "older question"
    assert page.json()["items"][0]["message_count"] == 1


def test_chat_session_list_handles_empty_and_corrupt_json(monkeypatch, tmp_path):
    settings = _settings(monkeypatch, tmp_path)
    asyncio.run(
        create_chat_session(
            settings,
            session_id="empty",
            messages=[],
            created_at="2026-08-01T00:00:00+00:00",
        )
    )
    asyncio.run(
        create_chat_session(
            settings,
            session_id="corrupt",
            messages=[{"role": "user", "content": "ignored"}],
            created_at="2026-08-02T00:00:00+00:00",
        )
    )
    asyncio.run(
        create_chat_session(
            settings,
            session_id="mixed",
            messages=[
                {"role": "user", "content": "mixed"},
                "not-a-message",
                {},
            ],
            created_at="2026-08-03T00:00:00+00:00",
        )
    )

    async def corrupt() -> None:
        async with get_db(settings) as db:
            await db.execute(
                "UPDATE chat_sessions SET messages = ? WHERE id = ?",
                ("{not-json", "corrupt"),
            )
            await db.commit()

    asyncio.run(corrupt())

    with TestClient(app) as client:
        response = client.get("/api/chat?limit=100")

    assert response.status_code == 200
    by_id = {item["session_id"]: item for item in response.json()["items"]}
    assert by_id["empty"]["title"] == "未命名会话"
    assert by_id["empty"]["message_count"] == 0
    assert by_id["corrupt"]["title"] == "未命名会话"
    assert by_id["corrupt"]["message_count"] == 0
    assert by_id["mixed"]["title"] == "mixed"
    assert by_id["mixed"]["message_count"] == 1


def test_post_chat_remains_compatible(monkeypatch, tmp_path):
    _settings(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"message": " hello "})
        listed = client.get("/api/chat?limit=100")

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"] == {"role": "user", "content": "hello"}
    assert payload["session_id"]
    assert listed.json()["items"][0]["session_id"] == payload["session_id"]
    assert listed.json()["items"][0]["message_count"] == 1
