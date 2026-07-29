"""Tests for embedding provider configuration."""

from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
import pytest

from indexing.embeddings import (
    FakeEmbeddings,
    LengthLimitedEmbeddings,
    RetryingEmbeddings,
    get_cloud_embeddings,
)


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


class _TransientEmbeddingError(Exception):
    status_code = 400
    body = {
        "error": {
            "code": "InternalError",
            "message": "Receive batching backend response failed!",
        }
    }


@dataclass
class _FlakyEmbeddings(Embeddings):
    failures: int
    calls: int = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        if self.calls <= self.failures:
            raise _TransientEmbeddingError
        return [1.0]


def test_retrying_embeddings_retries_same_transient_query(monkeypatch) -> None:
    delegate = _FlakyEmbeddings(failures=2)
    monkeypatch.setattr("indexing.embeddings.time.sleep", lambda _: None)
    embeddings = RetryingEmbeddings(delegate=delegate)

    assert embeddings.embed_query("same query") == [1.0]
    assert delegate.calls == 3


def test_retrying_embeddings_does_not_retry_other_400(monkeypatch) -> None:
    class _BadRequest(Exception):
        status_code = 400
        body = {"error": {"code": "InvalidParameter"}}

    delegate = _FlakyEmbeddings(failures=0)

    def fail(_: str) -> list[float]:
        raise _BadRequest

    monkeypatch.setattr(delegate, "embed_query", fail)
    embeddings = RetryingEmbeddings(delegate=delegate)

    with pytest.raises(_BadRequest):
        embeddings.embed_query("invalid")
