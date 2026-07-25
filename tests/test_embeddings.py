"""Tests for embedding provider configuration."""

from indexing.embeddings import get_cloud_embeddings


def test_get_cloud_embeddings_sends_raw_text_to_provider(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class StubEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("indexing.embeddings.OpenAIEmbeddings", StubEmbeddings)

    get_cloud_embeddings(
        model="test-embedding-model",
        api_key="test-key",
        api_base="https://example.com/v1",
    )

    assert captured["check_embedding_ctx_length"] is False
