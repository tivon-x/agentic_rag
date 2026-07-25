"""Tests for embedding provider configuration."""

import pytest

from indexing.embeddings import FakeEmbeddings, LengthLimitedEmbeddings, get_cloud_embeddings


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


def test_length_limited_embeddings_rejects_oversized_raw_input() -> None:
    embeddings = LengthLimitedEmbeddings(
        delegate=FakeEmbeddings(dimensions=8),
        max_input_chars=5,
    )

    with pytest.raises(ValueError, match="EMBEDDING_MAX_INPUT_CHARS=5"):
        embeddings.embed_documents(["123456"])
