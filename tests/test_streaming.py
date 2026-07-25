from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from api.main import create_app
from api.services.index_worker import IndexWorker
from core.settings import load_settings


def test_chat_stream_emits_only_progress_evidence_and_one_final_answer(
    tmp_path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("APP_DB_PATH", str(data_dir / "api" / "sessions.db"))
    monkeypatch.setenv("UPLOAD_ROOT", str(data_dir / "uploads"))
    monkeypatch.setenv("INDEX_ROOT", str(data_dir / "indexes"))
    monkeypatch.setenv("OFFLINE_MODE", "0")
    settings = load_settings(base_dir=tmp_path, env_file=tmp_path / "missing.env")

    async def no_start(self: IndexWorker) -> None:
        return None

    class FakeGraph:
        def invoke(self, input_state, config=None):
            assert config is not None
            return {
                "messages": [
                    *input_state["messages"],
                    AIMessage(content="verified final answer"),
                ],
                "routingDecision": "SECRET_ROUTING_TOKEN",
                "queryPlan": "SECRET_PLANNING_TOKEN",
            }

    monkeypatch.setattr(IndexWorker, "start", no_start)
    monkeypatch.setattr("api.routers.chat.get_cached_graph", lambda _: FakeGraph())
    monkeypatch.setattr(
        "api.routers.chat._extract_citations",
        lambda _: "verified evidence",
    )

    with TestClient(create_app(settings)) as client:
        created = client.post("/api/chat", json={"message": "question"})
        session_id = created.json()["session_id"]
        streamed = client.get(f"/api/chat/stream?session_id={session_id}")

    assert streamed.status_code == 200
    assert "event: progress" in streamed.text
    assert "event: evidence" in streamed.text
    assert streamed.text.count("event: answer.final") == 1
    assert "event: token" not in streamed.text
    assert "event: citations" not in streamed.text
    assert "SECRET_ROUTING_TOKEN" not in streamed.text
    assert "SECRET_PLANNING_TOKEN" not in streamed.text
    assert streamed.text.count("verified final answer") == 1
