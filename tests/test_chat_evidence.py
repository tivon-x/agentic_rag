from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.documents import Document

from api.db.database import create_chat_session
from api.main import create_app
from api.services.index_worker import IndexWorker
from api.routers.chat import _extract_chat_evidence, _normalize_evidence_item
from core.settings import load_settings


def _settings(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("APP_DB_PATH", str(data_dir / "api" / "sessions.db"))
    monkeypatch.setenv("UPLOAD_ROOT", str(data_dir / "uploads"))
    monkeypatch.setenv("INDEX_ROOT", str(data_dir / "indexes"))
    monkeypatch.setenv("OFFLINE_MODE", "0")
    return load_settings(base_dir=tmp_path, env_file=tmp_path / "missing.env")


def test_graph_evidence_is_persisted_per_answer_and_survives_reload(
    monkeypatch,
    tmp_path,
) -> None:
    settings = _settings(monkeypatch, tmp_path)

    async def no_start(self: IndexWorker) -> None:
        return None

    class FakeGraph:
        calls = 0

        def invoke(self, input_state, config=None):
            assert config is not None
            FakeGraph.calls += 1
            paper_id = f"paper-{FakeGraph.calls}"
            return {
                "messages": [
                    *input_state["messages"],
                    AIMessage(content=f"answer {FakeGraph.calls}"),
                ],
                "groundedAnswer": {
                    "evidence": [
                        {
                            "doc_id": f"doc-{FakeGraph.calls}",
                            "node_id": f"node-{FakeGraph.calls}",
                            # The answer model may omit or rewrite catalog IDs;
                            # the router must use the retrieval-owned artifact.
                            "paper_id": f"model-invented-{FakeGraph.calls}",
                            "paper_title": f"Paper {FakeGraph.calls}",
                            "source": f"C:\\library\\{paper_id}.pdf",
                            "section_path": ["3 Methods", "3.1 Setup"],
                            "page": FakeGraph.calls + 2,
                            "quote": f"Source quote {FakeGraph.calls}.",
                            "score": 0.8,
                            "relevance": "Directly answers the question.",
                        }
                    ]
                },
                "evidenceGroups": [
                    {
                        "subquery": f"question {FakeGraph.calls}",
                        "evidence": [
                            {
                                "doc_id": f"doc-{FakeGraph.calls}",
                                "node_id": f"node-{FakeGraph.calls}",
                                "paper_id": paper_id,
                                "paper_title": f"Paper {FakeGraph.calls}",
                                "source": f"C:\\library\\{paper_id}.pdf",
                                "section_path": ["3 Methods", "3.1 Setup"],
                                "page": FakeGraph.calls + 2,
                                "quote": f"Source quote {FakeGraph.calls}.",
                                "score": 0.8,
                            }
                        ],
                    }
                ],
            }

    monkeypatch.setattr(IndexWorker, "start", no_start)
    monkeypatch.setattr("api.routers.chat.get_cached_graph", lambda _: FakeGraph())

    with TestClient(create_app(settings)) as client:
        first = client.post("/api/chat", json={"message": "first question"})
        session_id = first.json()["session_id"]
        first_stream = client.get(f"/api/chat/stream?session_id={session_id}")

        second = client.post(
            "/api/chat",
            json={"message": "second question", "session_id": session_id},
        )
        second_stream = client.get(f"/api/chat/stream?session_id={session_id}")
        session = client.get(f"/api/chat/{session_id}")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first_stream.status_code == 200
    assert second_stream.status_code == 200
    assert first_stream.text.index("event: evidence") < first_stream.text.index(
        "event: answer.final"
    )
    assert '"paper_id": "paper-1"' in first_stream.text
    assert '"paper_id": "paper-2"' in second_stream.text
    assert session.status_code == 200
    messages = session.json()["messages"]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert messages[1]["evidence"][0]["paper_id"] == "paper-1"
    assert messages[3]["evidence"][0]["paper_id"] == "paper-2"
    assert messages[1]["evidence"][0]["quote"] == "Source quote 1."
    assert messages[3]["evidence"][0]["quote"] == "Source quote 2."
    assert "C:\\library" not in first_stream.text


def test_old_session_is_readable_and_empty_retrieval_is_not_persisted(
    monkeypatch,
    tmp_path,
) -> None:
    settings = _settings(monkeypatch, tmp_path)

    async def no_start(self: IndexWorker) -> None:
        return None

    class EmptyEvidenceGraph:
        def invoke(self, input_state, config=None):
            return {
                "messages": [*input_state["messages"], AIMessage(content="No evidence answer")],
                "groundedAnswer": {"evidence": []},
                "routingDecision": "retrieve",
                "evidenceGroups": [],
            }

    monkeypatch.setattr(IndexWorker, "start", no_start)
    monkeypatch.setattr(
        "api.routers.chat.get_cached_graph", lambda _: EmptyEvidenceGraph()
    )

    import asyncio

    asyncio.run(
        create_chat_session(
            settings,
            session_id="old-session",
            messages=[{"role": "user", "content": "旧问题"}],
            created_at="2026-08-02T00:00:00+00:00",
        )
    )

    with TestClient(create_app(settings)) as client:
        old = client.get("/api/chat/old-session")
        created = client.post("/api/chat", json={"message": "new question"})
        stream = client.get(
            f"/api/chat/stream?session_id={created.json()['session_id']}"
        )
        saved = client.get(f"/api/chat/{created.json()['session_id']}")

    assert old.status_code == 200
    assert old.json()["messages"] == [{"role": "user", "content": "旧问题"}]
    assert stream.status_code == 200
    assert "event: stream-error" in stream.text
    assert "event: answer.final" not in stream.text
    assert saved.json()["messages"] == [
        {"role": "user", "content": "new question"}
    ]


def test_retrieval_failure_streams_recovery_and_does_not_persist_answer(
    monkeypatch,
    tmp_path,
) -> None:
    settings = _settings(monkeypatch, tmp_path)

    async def no_start(self: IndexWorker) -> None:
        return None

    class BrokenRetriever:
        def invoke(self, question: str):
            raise RuntimeError("retrieval backend unavailable")

    monkeypatch.setattr(IndexWorker, "start", no_start)
    monkeypatch.setattr("api.routers.chat.get_cached_graph", lambda _: None)
    monkeypatch.setattr("api.routers.chat.build_retriever", lambda _: BrokenRetriever())

    with TestClient(create_app(settings)) as client:
        created = client.post("/api/chat", json={"message": "retrieval failure"})
        session_id = created.json()["session_id"]
        stream = client.get(f"/api/chat/stream?session_id={session_id}")
        saved = client.get(f"/api/chat/{session_id}")

    assert stream.status_code == 200
    stream_body = stream.content.decode("utf-8")
    assert "event: stream-error" in stream_body
    assert "检索失败，回答没有保存，请重试。" in stream_body
    assert "event: answer.final" not in stream_body
    assert saved.json()["messages"] == [
        {"role": "user", "content": "retrieval failure"}
    ]


def test_answer_save_failure_streams_recovery_and_does_not_report_success(
    monkeypatch,
    tmp_path,
) -> None:
    settings = _settings(monkeypatch, tmp_path)

    async def no_start(self: IndexWorker) -> None:
        return None

    class AnswerGraph:
        def invoke(self, input_state, config=None):
            return {
                "messages": [
                    *input_state["messages"],
                    AIMessage(content="A valid answer"),
                ],
                "groundedAnswer": {"evidence": []},
                "evidenceGroups": [],
            }

    async def fail_save(*args, **kwargs) -> bool:
        return False

    monkeypatch.setattr(IndexWorker, "start", no_start)
    monkeypatch.setattr("api.routers.chat.get_cached_graph", lambda _: AnswerGraph())
    monkeypatch.setattr("api.routers.chat._save_history", fail_save)

    with TestClient(create_app(settings)) as client:
        created = client.post("/api/chat", json={"message": "save failure"})
        session_id = created.json()["session_id"]
        stream = client.get(f"/api/chat/stream?session_id={session_id}")

    assert stream.status_code == 200
    stream_body = stream.content.decode("utf-8")
    assert "event: stream-error" in stream_body
    assert "回答保存失败，回答没有保存，请重试。" in stream_body
    assert "event: answer.final" not in stream_body


def test_missing_catalog_paper_id_is_not_invented_or_persisted(
    monkeypatch,
    tmp_path,
) -> None:
    settings = _settings(monkeypatch, tmp_path)

    async def no_start(self: IndexWorker) -> None:
        return None

    class NoCatalogGraph:
        def invoke(self, input_state, config=None):
            return {
                "messages": [
                    *input_state["messages"],
                    AIMessage(content="Answer without a catalog record"),
                ],
                "groundedAnswer": {
                    "evidence": [
                        {
                            "node_id": "node-no-catalog",
                            "paper_id": "model-invented-id",
                            "source": "unmanaged-paper.pdf",
                            "page": 4,
                            "quote": "Quote from a source outside the catalog.",
                        }
                    ]
                },
                "evidenceGroups": [
                    {
                        "evidence": [
                            {
                                "node_id": "node-no-catalog",
                                "paper_id": None,
                                "source": "unmanaged-paper.pdf",
                                "page": 4,
                                "quote": "Quote from a source outside the catalog.",
                            }
                        ]
                    }
                ],
            }

    monkeypatch.setattr(IndexWorker, "start", no_start)
    monkeypatch.setattr(
        "api.routers.chat.get_cached_graph", lambda _: NoCatalogGraph()
    )

    with TestClient(create_app(settings)) as client:
        created = client.post("/api/chat", json={"message": "uncataloged question"})
        session_id = created.json()["session_id"]
        stream = client.get(f"/api/chat/stream?session_id={session_id}")
        saved = client.get(f"/api/chat/{session_id}")

    assert stream.status_code == 200
    assert '"paper_id": null' in stream.text
    assert "model-invented-id" not in stream.text
    evidence = saved.json()["messages"][1]["evidence"][0]
    assert evidence.get("paper_id") is None

    import asyncio

    from api.db.database import get_chat_session_messages

    persisted = asyncio.run(
        get_chat_session_messages(settings, session_id=session_id)
    )
    assert persisted is not None
    assert persisted[1]["evidence"][0]["paper_id"] is None


def test_catalog_document_metadata_reaches_chat_evidence() -> None:
    from api.routers.chat import _documents_to_chat_evidence

    document = Document(
        page_content="catalog quote",
        metadata={
            "node_id": "catalog-node",
            "paper_id": "catalog-paper-id",
            "paper_title": "Catalog Paper",
            "source": "catalog-paper.pdf",
            "page": 2,
            "quote_text": "catalog quote",
        },
    )

    evidence = _documents_to_chat_evidence([document])

    assert evidence[0].paper_id == "catalog-paper-id"
    assert evidence[0].paper_title == "Catalog Paper"
    assert evidence[0].page == 2


def test_model_only_evidence_cannot_supply_a_paper_id() -> None:
    evidence = _extract_chat_evidence(
        {
            "groundedAnswer": {
                "evidence": [
                    {
                        "node_id": "model-node",
                        "paper_id": "hallucinated-paper-id",
                        "source": "unmanaged.pdf",
                        "quote": "model quote",
                    }
                ]
            }
        }
    )

    assert len(evidence) == 1
    assert evidence[0].paper_id is None


def test_evidence_normalization_requires_source_faithful_quote_and_stable_node() -> None:
    assert _normalize_evidence_item(
        {
            "node_id": "node-1",
            "source": "paper.pdf",
            "quote": "quoted text",
        }
    ).model_dump() == {
        "node_id": "node-1",
        "paper_id": None,
        "paper_title": None,
        "source": "paper.pdf",
        "section_path": [],
        "page": None,
        "quote": "quoted text",
        "score": None,
        "relevance": None,
    }
    assert _normalize_evidence_item(
        {"node_id": "node-2", "source": "paper.pdf", "quote": ""}
    ) is None
    assert _normalize_evidence_item(
        {"source": "paper.pdf", "quote": "quote"}
    ) is None
